import asyncio
import json
import logging
import os

import requests
from telegram import Bot
from typing import Optional, Dict, Any
from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException
from dotenv import load_dotenv

load_dotenv()


class Config:
    ERROR_TELEGRAM_BOT_TOKEN = os.getenv("ERROR_TELEGRAM_BOT_TOKEN")
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    BASE_URL = "https://api.astroguru.ru/api/v1"
    TG_CHAT_ID = "@astro_errors"
    DOC_PATH = "./tgbot_data/numero_db.docx"
    CHROMA_DIR = "./tgbot_data/chroma"
    LOG_FILE = "tgbot_logs/tgbot.log"
    VIDEO_PATH = "./tgbot_data/fffx.mp4"
    TECH_SUPPORT_ID = -1002011240000


class AccessChecker:
    def __init__(
        self, base_url: str, tg_bot_token: str, tg_chat_id: str, timeout: int = 10
    ):
        self.base_url = base_url
        self.tg_bot_token = tg_bot_token
        self.tg_chat_id = tg_chat_id
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self.bot = Bot(token=tg_bot_token)

    async def _send_error_to_telegram(self, error: str, bot_name: str):
        error_message = json.dumps({"bot": bot_name, "error": error}, indent=2)
        if len(error_message) > 4096:
            error_message = error_message[:4093] + "..."
        try:
            await self.bot.send_message(
                chat_id=self.tg_chat_id,
                text=f"```json\n{error_message}\n```",
                parse_mode="MarkdownV2",
            )
            self.logger.info("Error message sent to Telegram.")
        except Exception as err:
            self.logger.error(f"Failed to send error message to Telegram: {err}")

    def _post_request(
        self, endpoint: str, payload: Dict[str, Any], bot_name: str
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        self.logger.info(f"Sending POST request to {url} with payload: {payload}")
        try:
            response = requests.post(
                url, data=json.dumps(payload), timeout=self.timeout
            )
            response.raise_for_status()
            self.logger.info(f"Received response: {response.json()}")
            return response.json()
        except (HTTPError, ConnectionError, Timeout, RequestException) as req_err:
            self.logger.error(f"Request error occurred: {req_err}")
            asyncio.create_task(
                self._send_error_to_telegram(
                    error=json.dumps(
                        {"RequestException": str(req_err) + payload["email"]}
                    ),
                    bot_name=bot_name,
                )
            )
            return {"status": "error", "message": f"Request error occurred: {req_err}"}
        except Exception as err:
            self.logger.error(f"An unexpected error occurred: {err}")
            asyncio.create_task(
                self._send_error_to_telegram(
                    error=json.dumps({"Exception": str(err)}), bot_name=bot_name
                )
            )
            return {
                "status": "error",
                "message": f"An unexpected error occurred: {err}",
            }

    def check_email(
        self, tg_id: int, email: str, bot: str, bot_id: int
    ) -> Dict[str, Any]:
        payload = {"email": email, "tg_id": tg_id, "bot_id": bot_id}
        print(payload)
        return self._post_request(
            endpoint="/check_email", payload=payload, bot_name=bot
        )

    def get_limits(self, tg_id: int, bot: str) -> int:
        payload = {"tg_id": tg_id, "bot_id": 2}
        print(payload)
        return self._post_request(
            endpoint="/veda.get_messages_limit", payload=payload, bot_name=bot
        )

    def check_access(
        self, tg_id: int, bot_id: int, bot: str, task: Optional[str] = None
    ) -> Dict[str, Any]:
        payload = {"tg_id": tg_id, "bot_id": bot_id}
        if task:
            payload["task"] = task
        return self._post_request(
            endpoint="/check_access", payload=payload, bot_name=bot
        )

    def minus_message(self, tg_id: int, bot: str) -> dict:
        payload = {"tg_id": tg_id, "bot_id": 2}
        return self._post_request(
            endpoint="/veda.minus_message", payload=payload, bot_name=bot
        )

    def add_cost(self, tg_id: int, price: float, bot_id: int, bot: str):
        payload = {"tg_id": tg_id, "price": price, "bot_id": bot_id}
        return self._post_request(endpoint="/token.add", payload=payload, bot_name=bot)


access_checker = AccessChecker(
    base_url=Config.BASE_URL,
    tg_bot_token=Config.ERROR_TELEGRAM_BOT_TOKEN,
    tg_chat_id=Config.TG_CHAT_ID,
)
