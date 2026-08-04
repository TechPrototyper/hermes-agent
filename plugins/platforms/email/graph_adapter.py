"""
Microsoft Graph API Email platform adapter for the Hermes gateway.

Allows users to interact with Hermes by sending emails via Microsoft Graph API.
Uses Microsoft Graph API (OAuth2) to receive/send messages with SMTP fallback for sending.

Environment variables / secrets:
    EMAIL_AUTH_MODE     — 'graph' to use this adapter (default is 'imap')
    EMAIL_ADDRESS       — Email address for the agent (overrides graph_creds.json)
    EMAIL_POLL_INTERVAL — Seconds between mailbox checks (default: 15)
    EMAIL_ALLOWED_USERS — Comma-separated list of allowed sender addresses
    GRAPH_CREDS_PATH    — Path to graph_creds.json (default: ~/.hermes/graph_creds.json)
    GRAPH_TOKEN_PATH    — Path to mail_oauth_token.json (default: ~/.hermes/mail_oauth_token.json)
"""

import asyncio
import base64
import json
import logging
import os
import re
import smtplib
import socket
import ssl
import sys
import time
import uuid
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# Profile-scoped secret reader for multiplexing support
from agent.secret_scope import UnscopedSecretError as _UnscopedSecretError
from agent.secret_scope import get_secret as _scoped_get_secret
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_document_from_bytes,
    cache_image_from_bytes,
)
from utils import is_truthy_value

logger = logging.getLogger(__name__)

GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"
DEFAULT_CREDS_PATH = os.path.expanduser("~/.hermes/graph_creds.json")
DEFAULT_TOKEN_PATH = os.path.expanduser("~/.hermes/mail_oauth_token.json")
DEFAULT_WHITELIST_PATH = os.path.expanduser("~/.hermes/email_whitelist.txt")

# Automated sender patterns — emails from these are silently ignored
_NOREPLY_PATTERNS = (
    "noreply", "no-reply", "no_reply", "donotreply", "do-not-reply",
    "mailer-daemon", "postmaster", "bounce", "notifications@",
    "automated@", "auto-confirm", "auto-reply", "automailer",
)

# Supported image extensions for inline/attachment detection
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

# Max length per email body
MAX_MESSAGE_LENGTH = 50_000


def _get_esecret(name: str, default: str = "") -> str:
    """Scope-aware ``EMAIL_*`` / ``GRAPH_*`` secret read with fallback."""
    try:
        val = _scoped_get_secret(name, default)
    except _UnscopedSecretError:
        val = os.getenv(name)
    return val if val is not None else default


_get_secret = _get_esecret


def _esecret_int(name: str, default: int) -> int:
    """Scope-aware integer read."""
    raw = str(_get_esecret(name, "")).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


def _strip_html(html: str) -> str:
    """Naive HTML tag stripper for text extraction."""
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"<p[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_email_address(raw: str) -> str:
    """Extract bare email address from 'Name <addr>' format."""
    match = re.search(r"<([^>]+)>", raw)
    if match:
        return match.group(1).strip().lower()
    return raw.strip().lower()


def _is_automated_sender(address: str) -> bool:
    """Return True if this email is from an automated/noreply source."""
    addr = address.lower()
    return any(pattern in addr for pattern in _NOREPLY_PATTERNS)


def get_graph_token(creds_path: str = DEFAULT_CREDS_PATH, token_path: str = DEFAULT_TOKEN_PATH) -> str:
    """Get Graph API access token, auto-refreshing via OAuth2 if near expiry.
    
    Mirrors imap_token.py and process_inbox.py token management.
    """
    creds_file = Path(creds_path)
    token_file = Path(token_path)

    if not creds_file.exists():
        raise FileNotFoundError(f"Graph API credentials file not found at {creds_path}")

    with open(creds_file, "r", encoding="utf-8") as f:
        creds = json.load(f)

    tenant_id = creds["tenant_id"]
    client_id = creds["client_id"]
    client_secret = creds["client_secret"]

    if not token_file.exists():
        raise FileNotFoundError(f"OAuth2 token file not found at {token_path}")

    with open(token_file, "r", encoding="utf-8") as f:
        token_data = json.load(f)

    created = token_data.get("_created", 0)
    expires_in = token_data.get("expires_in", 3600)
    refresh_token = token_data.get("refresh_token")

    # Check if near expiry (5 minute buffer)
    if created and (time.time() - created > expires_in - 300):
        if not refresh_token:
            raise ValueError(f"Token at {token_path} is expired and has no refresh_token")

        logger.info("[GraphEmail] Token near expiry, refreshing via Microsoft OAuth2...")
        token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
        resp = requests.post(
            token_url,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=30,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"Token refresh failed ({resp.status_code}): {resp.text[:200]}")

        new_token_data = resp.json()
        new_token_data["_created"] = time.time()
        if "refresh_token" not in new_token_data:
            new_token_data["refresh_token"] = refresh_token

        with open(token_file, "w", encoding="utf-8") as f:
            json.dump(new_token_data, f, indent=2)

        logger.info("[GraphEmail] Token refreshed successfully.")
        return new_token_data["access_token"]

    return token_data["access_token"]


def check_graph_email_requirements() -> bool:
    """Check if Microsoft Graph API requirements are met."""
    creds_path = _get_secret("GRAPH_CREDS_PATH", DEFAULT_CREDS_PATH)
    token_path = _get_secret("GRAPH_TOKEN_PATH", DEFAULT_TOKEN_PATH)
    return Path(creds_path).exists() and Path(token_path).exists()


class GraphEmailAdapter(BasePlatformAdapter):
    """Email gateway adapter using Microsoft Graph API (receive/send) + SMTP fallback."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.EMAIL)

        extra = config.extra or {}
        self._creds_path = os.path.expanduser(_get_secret("GRAPH_CREDS_PATH", DEFAULT_CREDS_PATH))
        self._token_path = os.path.expanduser(_get_secret("GRAPH_TOKEN_PATH", DEFAULT_TOKEN_PATH))
        self._whitelist_path = os.path.expanduser(_get_secret("GRAPH_WHITELIST_PATH", DEFAULT_WHITELIST_PATH))

        # Load credentials if available
        self._creds: Dict[str, Any] = {}
        if Path(self._creds_path).exists():
            try:
                with open(self._creds_path, "r", encoding="utf-8") as f:
                    self._creds = json.load(f)
            except Exception as e:
                logger.warning("[GraphEmail] Failed to load credentials from %s: %s", self._creds_path, e)

        self._address = (
            _get_secret("EMAIL_ADDRESS", "")
            or self._creds.get("mail_account", "")
            or extra.get("address", "")
        ).strip()
        self._user_password = self._creds.get("user_password", "") or _get_secret("EMAIL_PASSWORD", "")

        self._smtp_host = (_get_secret("EMAIL_SMTP_HOST", "") or extra.get("smtp_host", "") or "outlook.office365.com").strip()
        self._smtp_port = _esecret_int("EMAIL_SMTP_PORT", 587)
        self._poll_interval = _esecret_int("EMAIL_POLL_INTERVAL", 15)

        self._skip_attachments = extra.get("skip_attachments", False)
        self._seen_ids: set = set()
        self._seen_ids_max: int = 2000
        self._poll_task: Optional[asyncio.Task] = None

        # Map chat_id (sender email) -> last subject + message-id for threading
        self._thread_context: Dict[str, Dict[str, str]] = {}

        logger.info("[GraphEmail] Adapter initialized for %s", self._address)

    def _trim_seen_ids(self) -> None:
        """Keep seen message IDs bounded."""
        if len(self._seen_ids) <= self._seen_ids_max:
            return
        keep = self._seen_ids_max // 2
        self._seen_ids = set(list(self._seen_ids)[-keep:])

    def _get_token(self) -> str:
        """Get valid OAuth2 access token."""
        return get_graph_token(self._creds_path, self._token_path)

    def _graph_get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Execute GET request to Microsoft Graph API."""
        token = self._get_token()
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        url = f"{GRAPH_API_BASE}{endpoint}"
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code != 200:
            logger.error("[GraphEmail] GET %s returned status %d: %s", endpoint, resp.status_code, resp.text[:200])
            return None
        return resp.json()

    def _graph_patch(self, endpoint: str, body: Dict[str, Any]) -> bool:
        """Execute PATCH request to Microsoft Graph API."""
        token = self._get_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = f"{GRAPH_API_BASE}{endpoint}"
        resp = requests.patch(url, headers=headers, json=body, timeout=30)
        if resp.status_code not in (200, 204):
            logger.error("[GraphEmail] PATCH %s returned status %d: %s", endpoint, resp.status_code, resp.text[:200])
            return False
        return True

    def _graph_post(self, endpoint: str, body: Optional[Dict[str, Any]] = None) -> Tuple[bool, int, str]:
        """Execute POST request to Microsoft Graph API."""
        token = self._get_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = f"{GRAPH_API_BASE}{endpoint}"
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        success = resp.status_code in (200, 201, 202, 204)
        return success, resp.status_code, resp.text

    def _load_whitelist(self) -> List[str]:
        """Load allowlist from EMAIL_ALLOWED_USERS secret and/or email_whitelist.txt file."""
        rules = []
        allowed_raw = _get_secret("EMAIL_ALLOWED_USERS", "").strip()
        if allowed_raw:
            for item in allowed_raw.split(","):
                if item.strip():
                    rules.append(item.strip())

        if Path(self._whitelist_path).exists():
            try:
                with open(self._whitelist_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and line not in rules:
                            rules.append(line)
            except Exception as e:
                logger.warning("[GraphEmail] Failed to read whitelist file %s: %s", self._whitelist_path, e)

        return rules

    def _is_whitelisted(self, sender: str, whitelist: List[str]) -> bool:
        """Check if sender matches whitelist rules."""
        if not whitelist:
            return True
        sl = sender.lower()
        for pattern in whitelist:
            pl = pattern.lower()
            if pl.startswith("@") and sl.endswith(pl):
                return True
            elif pl.startswith("*"):
                if pl.endswith("*") and pl[1:-1] in sl:
                    return True
                elif sl.endswith(pl[1:]):
                    return True
            elif sl == pl:
                return True
        return False

    @staticmethod
    def _allow_all_senders() -> bool:
        """Return True if open sender access is configured."""
        truthy = {"true", "1", "yes"}
        return (
            _get_secret("EMAIL_ALLOW_ALL_USERS", "").strip().lower() in truthy
            or os.getenv("GATEWAY_ALLOW_ALL_USERS", "").strip().lower() in truthy
        )

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect and test Microsoft Graph API & SMTP connections."""
        if not self._address:
            logger.error("[GraphEmail] Missing EMAIL_ADDRESS or mail_account in graph_creds.json")
            self._set_fatal_error("graph_email_missing_address", "No email address configured", retryable=False)
            return False

        try:
            # Test Graph API token and access — use /me/mailFolders (requires Mail.ReadWrite only)
            token = self._get_token()
            folders = self._graph_get("/me/mailFolders")
            if not folders:
                logger.error("[GraphEmail] Failed to connect to Graph API /me/mailFolders endpoint.")
                return False
            folder_names = [f["displayName"] for f in folders.get("value", [])]
            logger.info("[GraphEmail] Connected to Graph API — folders: %s", folder_names[:5])
        except Exception as e:
            logger.error("[GraphEmail] Graph API connection test failed: %s", e)
            return False

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        print(f"[GraphEmail] Connected as {self._address}")
        return True

    async def disconnect(self) -> None:
        """Stop polling and disconnect."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        logger.info("[GraphEmail] Disconnected.")

    async def _poll_loop(self) -> None:
        """Poll Graph API for unread messages at regular intervals."""
        while self._running:
            try:
                await self._check_inbox()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[GraphEmail] Poll error: %s", e)
            await asyncio.sleep(self._poll_interval)

    async def _check_inbox(self) -> None:
        """Check INBOX for unread messages and dispatch them."""
        loop = asyncio.get_running_loop()
        messages = await loop.run_in_executor(None, self._fetch_new_messages)
        for msg_data in messages:
            await self._dispatch_message(msg_data)

    def _fetch_new_messages(self) -> List[Dict[str, Any]]:
        """Fetch unread messages via GET /me/messages?$filter=isRead eq false."""
        results = []
        try:
            params = {
                "$filter": "isRead eq false",
                "$orderby": "receivedDateTime desc",
                "$top": 20,
                "$select": "id,from,subject,body,bodyPreview,receivedDateTime,isRead,internetMessageId,hasAttachments,conversationId",
            }
            data = self._graph_get("/me/messages", params=params)
            if not data or "value" not in data or not data["value"]:
                return results

            whitelist = self._load_whitelist()
            allow_all = self._allow_all_senders()

            for email in data["value"]:
                msg_id = email["id"]
                if msg_id in self._seen_ids:
                    continue

                self._seen_ids.add(msg_id)
                self._trim_seen_ids()

                from_info = email.get("from", {}).get("emailAddress", {})
                sender_addr = _extract_email_address(from_info.get("address", ""))
                sender_name = from_info.get("name", "")

                subject = email.get("subject") or "(no subject)"
                internet_msg_id = email.get("internetMessageId") or msg_id

                # Skip self-messages
                if sender_addr == self._address.lower():
                    self._graph_patch(f"/me/messages/{msg_id}", {"isRead": True})
                    continue

                # Skip automated senders
                if _is_automated_sender(sender_addr):
                    logger.debug("[GraphEmail] Skipping automated sender: %s", sender_addr)
                    self._graph_patch(f"/me/messages/{msg_id}", {"isRead": True})
                    continue

                # Whitelist filtering
                if not allow_all and not self._is_whitelisted(sender_addr, whitelist):
                    logger.warning("[GraphEmail] Dropping non-whitelisted sender: %s (subject: %s)", sender_addr, subject)
                    self._graph_patch(f"/me/messages/{msg_id}", {"isRead": True})
                    continue

                # Body extraction
                body_obj = email.get("body", {})
                content_type = body_obj.get("contentType", "").lower()
                content = body_obj.get("content", "")
                if content_type == "html":
                    body = _strip_html(content)
                else:
                    body = content or email.get("bodyPreview", "")

                # Attachment extraction
                attachments = []
                if email.get("hasAttachments") and not self._skip_attachments:
                    attachments = self._fetch_attachments(msg_id)

                # Mark as read
                self._graph_patch(f"/me/messages/{msg_id}", {"isRead": True})

                results.append({
                    "graph_id": msg_id,
                    "sender_addr": sender_addr,
                    "sender_name": sender_name,
                    "subject": subject,
                    "message_id": internet_msg_id,
                    "body": body,
                    "attachments": attachments,
                    "date": email.get("receivedDateTime", ""),
                })
        except Exception as e:
            logger.error("[GraphEmail] Fetch error: %s", e)
        return results

    def _fetch_attachments(self, message_id: str) -> List[Dict[str, Any]]:
        """Fetch attachment items via GET /me/messages/{id}/attachments."""
        attachments = []
        data = self._graph_get(f"/me/messages/{message_id}/attachments")
        if not data or "value" not in data:
            return attachments

        for item in data["value"]:
            if item.get("@odata.type") != "#microsoft.graph.fileAttachment":
                continue

            filename = item.get("name", "attachment.bin")
            content_type = item.get("contentType", "application/octet-stream")
            content_bytes_b64 = item.get("contentBytes")

            if not content_bytes_b64:
                continue

            try:
                payload = base64.b64decode(content_bytes_b64)
            except Exception as e:
                logger.warning("[GraphEmail] Failed to decode base64 attachment %s: %s", filename, e)
                continue

            ext = Path(filename).suffix.lower()
            if ext in _IMAGE_EXTS or content_type.startswith("image/"):
                try:
                    cached_path = cache_image_from_bytes(payload, ext or ".png")
                    attachments.append({
                        "path": cached_path,
                        "filename": filename,
                        "type": "image",
                        "media_type": content_type,
                    })
                except ValueError:
                    continue
            else:
                cached_path = cache_document_from_bytes(payload, filename)
                attachments.append({
                    "path": cached_path,
                    "filename": filename,
                    "type": "document",
                    "media_type": content_type,
                })

        return attachments

    async def _dispatch_message(self, msg_data: Dict[str, Any]) -> None:
        """Convert fetched message into MessageEvent and handle."""
        sender_addr = msg_data["sender_addr"]
        subject = msg_data["subject"]
        body = msg_data["body"].strip()
        attachments = msg_data["attachments"]

        text = body
        if subject and not subject.startswith("Re:"):
            text = f"[Subject: {subject}]\n\n{body}"

        media_urls = []
        media_types = []
        msg_type = MessageType.TEXT

        for att in attachments:
            media_urls.append(att["path"])
            media_types.append(att["media_type"])
            if att["type"] == "image" and msg_type == MessageType.TEXT:
                msg_type = MessageType.PHOTO
            elif att["type"] == "document":
                msg_type = MessageType.DOCUMENT

        # Store thread context for reply threading
        self._thread_context[sender_addr] = {
            "subject": subject,
            "message_id": msg_data["message_id"],
        }

        source = self.build_source(
            chat_id=sender_addr,
            chat_name=msg_data["sender_name"] or sender_addr,
            chat_type="dm",
            user_id=sender_addr,
            user_name=msg_data["sender_name"] or sender_addr,
        )

        event = MessageEvent(
            text=text or "(empty email)",
            message_type=msg_type,
            source=source,
            message_id=msg_data["message_id"],
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=None,
        )

        logger.info("[GraphEmail] New message from %s: %s", sender_addr, subject)
        await self.handle_message(event)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an email reply via Graph API sendMail with SMTP fallback."""
        try:
            loop = asyncio.get_running_loop()
            message_id = await loop.run_in_executor(
                None, self._send_email, chat_id, content, reply_to
            )
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            logger.error("[GraphEmail] Send failed to %s: %s", chat_id, e)
            return SendResult(success=False, error=str(e))

    def _message_id_domain(self) -> str:
        """Domain part for generated Message-IDs."""
        if "@" in self._address:
            return self._address.rsplit("@", 1)[-1] or "localhost"
        return "localhost"

    def _send_email(
        self,
        to_addr: str,
        body: str,
        reply_to_msg_id: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
    ) -> str:
        """Send email via POST /me/sendMail with SMTP fallback."""
        ctx = self._thread_context.get(to_addr, {})
        subject = ctx.get("subject", "Hermes Agent")
        if not subject.startswith("Re:"):
            subject = f"Re: {subject}"

        original_msg_id = reply_to_msg_id or ctx.get("message_id")
        generated_msg_id = f"<hermes-{uuid.uuid4().hex[:12]}@{self._message_id_domain()}>"

        # Prepare attachments for Graph API
        graph_attachments = []
        if file_paths:
            for fp in file_paths:
                p = Path(fp)
                if not p.exists():
                    continue
                try:
                    with open(p, "rb") as f:
                        b64_content = base64.b64encode(f.read()).decode("utf-8")
                    graph_attachments.append({
                        "@odata.type": "#microsoft.graph.fileAttachment",
                        "name": p.name,
                        "contentType": "application/octet-stream",
                        "contentBytes": b64_content,
                    })
                except Exception as e:
                    logger.warning("[GraphEmail] Could not read attachment %s for Graph API: %s", fp, e)

        # Build Graph sendMail payload
        headers_list = []
        if original_msg_id:
            headers_list.append({"name": "X-In-Reply-To", "value": original_msg_id})
            headers_list.append({"name": "X-References", "value": original_msg_id})

        message_payload: Dict[str, Any] = {
            "subject": subject,
            "body": {
                "contentType": "Text",
                "content": body,
            },
            "toRecipients": [
                {
                    "emailAddress": {
                        "address": to_addr,
                    }
                }
            ],
        }

        if headers_list:
            message_payload["internetMessageHeaders"] = headers_list
        if graph_attachments:
            message_payload["attachments"] = graph_attachments

        payload = {
            "message": message_payload,
            "saveToSentItems": "true",
        }

        # Try Graph API sendMail
        success, status, err_text = self._graph_post("/me/sendMail", payload)
        if success:
            logger.info("[GraphEmail] Sent reply via Graph API to %s (subject: %s)", to_addr, subject)
            return generated_msg_id

        logger.warning(
            "[GraphEmail] POST /me/sendMail failed (status %d: %s). Falling back to SMTP...",
            status, err_text[:200]
        )

        # Fallback to SMTP
        return self._send_email_smtp_fallback(to_addr, subject, body, original_msg_id, generated_msg_id, file_paths)

    def _send_email_smtp_fallback(
        self,
        to_addr: str,
        subject: str,
        body: str,
        original_msg_id: Optional[str],
        generated_msg_id: str,
        file_paths: Optional[List[str]] = None,
    ) -> str:
        """Fallback SMTP sender using Office 365 or configured SMTP host."""
        if not self._user_password:
            raise RuntimeError("SMTP fallback failed: user_password not found in credentials/env")

        msg = MIMEMultipart()
        msg["From"] = self._address
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = generated_msg_id

        if original_msg_id:
            msg["In-Reply-To"] = original_msg_id
            msg["References"] = original_msg_id

        if body:
            msg.attach(MIMEText(body, "plain", "utf-8"))

        if file_paths:
            for fp in file_paths:
                p = Path(fp)
                if not p.exists():
                    continue
                try:
                    with open(p, "rb") as f:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(f.read())
                        encoders.encode_base64(part)
                        part.add_header("Content-Disposition", f"attachment; filename={p.name}")
                        msg.attach(part)
                except Exception as e:
                    logger.warning("[GraphEmail] Failed to attach %s for SMTP: %s", fp, e)

        server = smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=30)
        try:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(self._address, self._user_password)
            server.send_message(msg)
        finally:
            try:
                server.quit()
            except Exception:
                server.close()

        logger.info("[GraphEmail] Sent reply via SMTP fallback to %s (subject: %s)", to_addr, subject)
        return generated_msg_id

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Email has no typing indicator — no-op."""

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send image URL as part of email body."""
        text = caption or ""
        text += f"\n\nImage: {image_url}"
        return await self.send(chat_id, text.strip(), reply_to)

    async def send_multiple_images(
        self,
        chat_id: str,
        images: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        human_delay: float = 0.0,
    ) -> None:
        """Send batch of images."""
        if not images:
            return

        from urllib.parse import unquote as _unquote

        body_parts: List[str] = []
        local_paths: List[str] = []
        for image_url, alt_text in images:
            if alt_text:
                body_parts.append(alt_text)
            if image_url.startswith("file://"):
                local_path = _unquote(image_url[7:])
                if Path(local_path).exists():
                    local_paths.append(local_path)
            else:
                body_parts.append(f"Image: {image_url}")

        if not local_paths and not body_parts:
            return

        body = "\n\n".join(body_parts)

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                self._send_email,
                chat_id,
                body,
                None,
                local_paths,
            )
        except Exception as e:
            logger.error("[GraphEmail] Multi-image send failed: %s", e, exc_info=True)
            await super().send_multiple_images(chat_id, images, metadata, human_delay)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a file as an email attachment."""
        try:
            loop = asyncio.get_running_loop()
            message_id = await loop.run_in_executor(
                None,
                self._send_email,
                chat_id,
                caption or "",
                reply_to,
                [file_path],
            )
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            logger.error("[GraphEmail] Send document failed: %s", e)
            return SendResult(success=False, error=str(e))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic chat info."""
        ctx = self._thread_context.get(chat_id, {})
        return {
            "name": chat_id,
            "type": "dm",
            "chat_id": chat_id,
            "subject": ctx.get("subject", ""),
        }


async def _standalone_send(
    pconfig,
    chat_id,
    message,
    *,
    thread_id=None,
    media_files=None,
    force_document=False,
):
    """Standalone send function for Microsoft Graph API email platform."""
    try:
        creds_path = _get_secret("GRAPH_CREDS_PATH", DEFAULT_CREDS_PATH)
        token_path = _get_secret("GRAPH_TOKEN_PATH", DEFAULT_TOKEN_PATH)
        token = get_graph_token(creds_path, token_path)

        extra = getattr(pconfig, "extra", {}) or {}
        address = extra.get("address") or _get_secret("EMAIL_ADDRESS", "")

        payload = {
            "message": {
                "subject": "Hermes Agent",
                "body": {"contentType": "Text", "content": message},
                "toRecipients": [{"emailAddress": {"address": chat_id}}],
            },
            "saveToSentItems": "true",
        }
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        r = requests.post(f"{GRAPH_API_BASE}/me/sendMail", headers=headers, json=payload, timeout=30)
        if r.status_code in (200, 201, 202, 204):
            return {"success": True, "platform": "email", "chat_id": chat_id}
        return {"error": f"Graph API sendMail failed ({r.status_code}): {r.text[:200]}"}
    except Exception as e:
        return {"error": f"Graph API standalone send failed: {e}"}


def _is_connected(config) -> bool:
    """Graph email is connected when graph_creds & token exist or EMAIL_ADDRESS is set."""
    return check_graph_email_requirements()


def _build_adapter(config):
    """Factory wrapper for GraphEmailAdapter."""
    return GraphEmailAdapter(config)


def register(ctx) -> None:
    """Plugin entry point for Microsoft Graph Email Adapter."""
    ctx.register_platform(
        name="email",
        label="Email (Microsoft Graph API)",
        adapter_factory=_build_adapter,
        check_fn=check_graph_email_requirements,
        is_connected=_is_connected,
        required_env=[],
        install_hint="Email Graph API requires requests and ~/.hermes/graph_creds.json",
        allowed_users_env="EMAIL_ALLOWED_USERS",
        allow_all_env="EMAIL_ALLOW_ALL_USERS",
        cron_deliver_env_var="EMAIL_HOME_ADDRESS",
        standalone_sender_fn=_standalone_send,
        max_message_length=50_000,
        pii_safe=True,
        emoji="📧",
        allow_update_command=True,
    )
