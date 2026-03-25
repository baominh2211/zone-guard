"""Alert dispatcher — Telegram, Slack, Email with rate limiting."""
import time
import logging
import smtplib
from email.message import EmailMessage
from typing import Optional
import httpx

logger = logging.getLogger(__name__)


class TelegramSender:
    def __init__(self, bot_token, chat_id):
        self._url = f"https://api.telegram.org/bot{bot_token}"
        self._chat_id = chat_id

    async def send(self, text, photo_path=None):
        if not self._chat_id:
            return False
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                if photo_path:
                    with open(photo_path, "rb") as f:
                        r = await c.post(f"{self._url}/sendPhoto",
                            data={"chat_id": self._chat_id, "caption": text[:1024]},
                            files={"photo": ("snap.jpg", f, "image/jpeg")})
                else:
                    r = await c.post(f"{self._url}/sendMessage",
                        json={"chat_id": self._chat_id, "text": text})
                r.raise_for_status()
                return True
        except Exception as e:
            logger.error("Telegram failed: %s", e)
            return False


class SlackSender:
    def __init__(self, webhook_url):
        self._url = webhook_url

    async def send(self, text):
        if not self._url:
            return False
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(self._url, json={"text": text})
                r.raise_for_status()
                return True
        except Exception as e:
            logger.error("Slack failed: %s", e)
            return False


class EmailSender:
    def __init__(self, host, port, user, password, from_addr, to_addrs):
        self._host, self._port = host, port
        self._user, self._pw = user, password
        self._from, self._to = from_addr, to_addrs

    async def send(self, subject, body, photo_path=None):
        if not self._host or not self._to:
            return False
        try:
            msg = EmailMessage()
            msg["Subject"], msg["From"], msg["To"] = subject, self._from, ", ".join(self._to)
            msg.set_content(body)
            if photo_path:
                with open(photo_path, "rb") as f:
                    msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename="snap.jpg")
            with smtplib.SMTP(self._host, self._port) as s:
                s.starttls()
                if self._user:
                    s.login(self._user, self._pw)
                s.send_message(msg)
            return True
        except Exception as e:
            logger.error("Email failed: %s", e)
            return False


class AlertDispatcher:
    def __init__(self, config=None):
        cfg = config or {}
        g = lambda k, d: cfg.get(k, d) if isinstance(cfg, dict) else getattr(cfg, k, d)
        self._rate = g("rate_limit_per_zone_seconds", 60)
        self._last: dict[str, float] = {}
        self._tg = TelegramSender(g("telegram_bot_token",""), g("telegram_chat_id",""))
        self._slack = SlackSender(g("slack_webhook_url",""))
        self._email = EmailSender(g("smtp_host",""), g("smtp_port",587), g("smtp_user",""),
                                   g("smtp_password",""), g("smtp_from",""), g("smtp_to",[]))

    async def dispatch(self, event, channels):
        now = time.time()
        if (now - self._last.get(event.zone_id, 0)) < self._rate:
            return {}
        self._last[event.zone_id] = now
        text = self._fmt(event)
        photo = event.snapshot_path or None
        results = {}
        for ch in channels:
            if ch == "telegram":
                results[ch] = await self._tg.send(text, photo)
            elif ch == "slack":
                results[ch] = await self._slack.send(text)
            elif ch == "email":
                results[ch] = await self._email.send(
                    f"[ZoneGuard] {event.event_type}: {event.zone_name}", text, photo)
            logger.info("Alert %s via %s: %s", event.zone_name, ch,
                         "OK" if results.get(ch) else "FAIL")
        return results

    @staticmethod
    def _fmt(event):
        ts = event.created_at.strftime("%Y-%m-%d %H:%M:%S")
        return (f"🚨 ZoneGuard Alert\n\nType: {event.event_type.replace('_',' ').title()}\n"
                f"Zone: {event.zone_name}\nCamera: {event.camera_id}\nTime: {ts}\n"
                f"Confidence: {event.confidence:.0%}\nTrack: #{event.track_id}\n"
                f"Occupancy: {event.occupancy_count}")

    async def close(self):
        pass
