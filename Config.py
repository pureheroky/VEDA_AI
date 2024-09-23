import os
import dotenv

dotenv.load_dotenv()


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
