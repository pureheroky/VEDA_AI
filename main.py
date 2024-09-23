# бот VEDA
import os
import time
import asyncio
import logging
from logging.handlers import RotatingFileHandler
import psycopg2
from psycopg2 import sql
from pinecone import Pinecone, PodSpec
from pinecone_text.sparse import BM25Encoder
from telegram import Update, ForceReply, Bot
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackContext,
    CallbackQueryHandler,
)
from langchain_community.document_loaders import Docx2txtLoader
import bs4
from langchain import hub
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException
from dotenv import load_dotenv
import json
import re
from typing import Optional, Dict, Any
import requests
import tiktoken


# Загрузка переменных окружения из .env файла
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


embedding = OpenAIEmbeddings(api_key=Config.OPENAI_API_KEY)


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


print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")


# Эта функция возвращает текущую временную метку в удобочитаемом формате
def current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


# Добавлен токен для Telegram бота
telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")


# база данных
def check_or_add_user(user_id, username, update_counts=False):
    try:
        # Подключение к базе данных...
        logger.info(
            f"Попытка подключения к базе данных для пользователя: {username}, ID: {user_id}"
        )
        # Подключение к базе данных
        conn = psycopg2.connect(
            dbname="telegram_bot_db",  # Имя вашей базы данных
            user="postgres",  # Имя пользователя базы данных
            password="kn7BOa0e1WF*",  # Пароль пользователя базы данных
            host="5.35.88.158",  # Адрес сервера
            port="5432",  # Порт, на котором слушает PostgreSQL
        )
        logger.info("Успешное подключение к базе данных")

        cur = conn.cursor()
        logger.info("Курсор успешно создан")

        # Проверка, существует ли пользователь
        cur.execute(
            "SELECT telegram_id, messages_limit FROM users WHERE telegram_id = %s;",
            (user_id,),
        )
        user = cur.fetchone()

        if user:
            _, messages_limit = user

            if update_counts and messages_limit > 0:
                # Уменьшаем лимит сообщений и увеличиваем количество отправленных сообщений
                cur.execute(
                    """
                    UPDATE users
                    SET messages_limit = messages_limit - 1, messages_sent = messages_sent + 1
                    WHERE telegram_id = %s;
                """,
                    (user_id,),
                )
                conn.commit()  # Добавлен вызов commit для сохранения изменений
                messages_limit -= 1

            return messages_limit
        else:
            # Пользователя нет, добавляем его с начальным лимитом 20 сообщений
            cur.execute(
                "INSERT INTO users (telegram_id, username, messages_limit, messages_sent, can_send) VALUES (%s, %s, %s, %s, %s);",
                (user_id, username, 10, 0, True),
            )
            conn.commit()
            logger.info(f"Новый пользователь добавлен: {username}, ID: {user_id}")
            return 10  # начальный лимит сообщений

    except Exception as e:
        logger.error(f"Ошибка при работе с базой данных: {e}")
        return None  # В случае ошибки
    finally:
        # Закрываем соединения
        cur.close()
        logger.info("Курсор успешно закрыт")
        conn.close()
        logger.info("Соединение закрыто")


# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Установка уровня логирования


# Создание фильтра
class CustomFilter(logging.Filter):
    def filter(self, record):
        # Ваши условия фильтрации
        if record.name == "httpx" and "getUpdates" in record.getMessage():
            return False
        if (
            record.name == "urllib3.connectionpool"
            and "ReadTimeoutError" in record.getMessage()
        ):
            return False
        if (
            record.name == "langchain_core.tracers.langchain"
            and "409 Client Error" in record.getMessage()
        ):
            return False
        return True


# Создание и настройка RotatingFileHandler
handler = RotatingFileHandler(
    "tgbot_logs/tgbot.log",
    maxBytes=1 * 1024 * 1024,  # Максимальный размер файла 1 МБ
    backupCount=5,  # Количество сохраняемых копий файлов логов
    encoding="utf-8",
)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
handler.addFilter(CustomFilter())  # Добавление фильтра к обработчику

# Добавление обработчика к логгеру
logger.addHandler(handler)


def load_documents(file_path):
    logger.info(f"Документы загружены из файла: {file_path}")
    loader = Docx2txtLoader(file_path)
    return loader.load()


# Проверка значения переменной окружения OPENAI_API_KEY
logger.info(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")

# Функция для разделения текста на блоки


def split_text_into_blocks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=160, add_start_index=True
    )
    logger.info("Текст успешно разбит на блоки.")
    return text_splitter.split_documents(docs)


docs = load_documents("./tgbot_data/db_veda.docx")

bm25_encoder = BM25Encoder().default()
all_splits = split_text_into_blocks(docs)
corpus = all_splits
text_corpus = [doc.page_content for doc in corpus]
bm25_encoder.fit(text_corpus)

# Функция для инициализации BM25Retriever


def initialize_pinecone_retriever(bm25_encoder):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "tests"
    index = pc.Index(index_name)
    # Проверка, существует ли файл с сохраненными значениями BM25
    if os.path.exists("bm25_values.json"):
        logger.info("Загрузка сохраненных значений BM25.")
        # Загрузка сохраненных значений, если файл существует
        bm25_encoder = BM25Encoder().load("bm25_values.json")
    else:
        logger.info("Файл BM25 не найден, начинается обучение и сохранение.")
        # Обучение BM25Encoder, если файл не существует

        # Сохранение обученных значений в файл
        bm25_encoder.dump("bm25_values.json")

    retriever = PineconeHybridSearchRetriever(
        embeddings=embedding,  # Например, embedding
        sparse_encoder=bm25_encoder,
        index=index,
        top_k=2,
    )
    # retriever.add_texts(text_corpus) #<-- добавлять один раз при первом запуске
    return retriever


# Функция для инициализации и загрузки Chroma


def initialize_chroma(all_splits, persist_directory, embedding):
    if os.path.exists(persist_directory):
        vectorstore = Chroma(
            persist_directory=persist_directory, embedding_function=embedding
        )
        logger.info("Chroma успешно загружена из существующего файла.")
    else:
        vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=embedding,
            persist_directory=persist_directory,
        )
        logger.info("Создан новый файл Chroma.")
    return vectorstore

    # Функция для инициализации EnsembleRetriever


def initialize_ensemble_retriever(retriever, faiss_retriever):
    return EnsembleRetriever(
        retrievers=[retriever, faiss_retriever], weights=[0.9, 0.1]
    )


def get_texts_from_docs(docs):
    return [doc.page_content for doc in docs]


def format_docs(docs):
    # Проверяем, является ли docs строкой и возвращаем ее напрямую,
    # иначе выполняем исходную логику.
    if isinstance(docs, str):
        return docs
    else:
        return "\n\n".join(doc.page_content for doc in docs)


# Глобальная переменная для хранения состояния запроса на помощь
help_request_state = {}


# Функция для обработки команды /help
async def help_command(update: Update, _: CallbackContext) -> None:
    """Отправляет запрос пользователю написать сообщение для техподдержки."""
    user_id = update.message.from_user.id
    help_request_state[user_id] = True  # Устанавливаем флаг в True
    await update.message.reply_text(
        "Введите сообщение, которое мы отправим в техподдержку:"
    )


# ID технического сотрудника
TECH_SUPPORT_ID = -1002011240000


# Функция для обработки текстового сообщения после команды /help
async def handle_help_message(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    user_id = update.message.from_user.id  # Telegram ID пользователя
    username = update.message.from_user.username  # Имя пользователя в Telegram
    # Формируем сообщение для технического сотрудника
    message_for_support = f"Запрос на помощь от пользователя @{username} (ID: {user_id}):\n\n{user_message}"
    logger.info(
        f"Пользователь @{username} (ID: {user_id}) отправил запрос в техподдержку: {user_message}"
    )

    # Отправляем сообщение техническому сотруднику
    await context.bot.send_message(TECH_SUPPORT_ID, message_for_support)

    # Отправляем пользователю уведомление о принятии запроса
    await update.message.reply_text(
        "Ваш запрос помощи отправлен, с вами свяжутся в ближайшее время."
    )

    if context.user_data.get("email_verified", False):
        # Пользователь уже прошел верификацию почты, возвращаем его в диалог с ботом
        await update.message.reply_text(
            "Вы вернулись в чат с ботом. Можете задавать ваши вопросы об астрологии."
        )
    else:
        # Пользователь еще не прошел верификацию почты
        await update.message.reply_text(
            "Вы вернулись на этап подтверждения почты. Пожалуйста, введите вашу почту с GetCourse."
        )
        context.user_data["awaiting_email"] = True


# Функция для потоковой обработки запроса с использованием RAG chain
def count_tokens(text: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


# Функция для расчета стоимости
def calculate_cost(
    input_tokens: int, output_tokens: int, model: str = "gpt-4"
) -> float:
    # Примерные тарифы (уточняйте в документации OpenAI)
    if model == "gpt-4":
        cost_per_1000_input_tokens = 0.03
        cost_per_1000_output_tokens = 0.06
    elif model == "gpt-3.5-turbo-0125":
        cost_per_1000_input_tokens = 0.002
        cost_per_1000_output_tokens = 0.002
    else:
        raise ValueError("Unknown model")

    total_cost = (input_tokens / 1000) * cost_per_1000_input_tokens + (
        output_tokens / 1000
    ) * cost_per_1000_output_tokens
    return total_cost


async def process_query_with_rag_chain(
    ensemble_retriever, user_query, model_name, prompt_id, user_id
):
    llm = ChatOpenAI(
        model_name=model_name, temperature=0.1, openai_api_key=Config.OPENAI_API_KEY
    )
    prompt = hub.pull(prompt_id)

    # Подсчет входных токенов
    input_tokens = count_tokens(user_query)

    rag_chain = (
        {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response_text = ""
    for chunk in rag_chain.stream(user_query):
        response_text += chunk

    # Подсчет выходных токенов
    output_tokens = count_tokens(response_text)

    # Расчет стоимости
    cost = calculate_cost(input_tokens, output_tokens)

    # Логирование стоимости
    logger.info(f"Стоимость запроса: {cost} USD")

    # Обновление стоимости в базе данных
    access_checker.add_cost(tg_id=user_id, price=cost, bot_id=2, bot="VEDA")

    return response_text


# Функция для создания RunnableLambda с предыдущим ответом


def create_context_runnable(response_text):
    return RunnableLambda(lambda input: response_text)


async def start(update: Update, context: CallbackContext) -> None:
    """Отправляет сообщение при получении команды /start."""
    await update.message.reply_text(
        "Пожалуйста введите ваш Email на котором вы зарегистрированы на GetCourse."
    )
    # Устанавливаем флаг, который говорит, что мы ожидаем ввода электронной почты
    # Сброс флагов для начала нового процесса верификации
    context.user_data["awaiting_email"] = True
    context.user_data["email_verified"] = False


async def handle_email(update: Update, context: CallbackContext) -> None:
    user_id = update.effective_user.id

    # Проверяем, ожидается ли ввод email
    if context.user_data.get("awaiting_email", False):
        # Если пользователь отправил запрос на помощь
        if help_request_state.get(user_id, False):
            await handle_help_message(update, context)
            help_request_state[user_id] = False  # Сбрасываем флаг запроса на помощь
            return  # Важно добавить return здесь, чтобы предотвратить дальнейшую обработку текста как email

        else:
            # Обработка ввода email
            email = update.message.text
            print(email)
            if check_email_in_database(email, user_id):
                context.user_data["awaiting_email"] = False
                context.user_data["email_verified"] = (
                    True  # Переходим в состояние EMAIL_VERIFIED
                )
                waiting_for_button_click[user_id] = (
                    False  # Сбрасываем флаг ожидания нажатия кнопки
                )

                await send_welcome_message(update, context)
            else:
                await update.message.reply_text(
                    "Электронная почта не найдена. Пожалуйста, введите корректный адрес."
                )
    else:
        # Если не ожидается ввода почты, передаем управление следующему обработчику
        await handle_message(update, context)


def check_email_in_database(email, tg_id):
    email_response = access_checker.check_email(
        tg_id=tg_id, email=email, bot="VEDA AI", bot_id=2
    )
    access = access_checker.check_access(tg_id=tg_id, bot_id=2, bot="VEDA AI")
    print(access)

    access_granted = False  # Инициализируем переменную по умолчанию как False

    if email_response["status"] == "success":
        email_access = email_response["response"].get("email_exist", False)
        order_exist = access["response"].get("order_exist", False)
        if email_access and order_exist:
            access_granted = True
    elif (
        email_response["status"] == "error"
        or email_response["message"]
        == "HTTP error 401: Unauthorized access. Please check your credentials."
        or not access["response"].get("order_exist", True)
    ):
        access_granted = False

    return access_granted


# Функция для обработки команды /start
async def send_welcome_message(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    # Пользователь может отправлять сообщения
    await update.message.reply_markdown_v2(rf"Привет, {user.mention_markdown_v2()}\!")
    # Отправка дополнительного информационного сообщения
    await update.message.reply_text(
        "Добро пожаловать в онлайн-школу астрологии! Здесь вы можете задавать вопросы "
        "по астрологии и получать развернутые ответы. Просто напишите свой вопрос."
    )

    # Отправка текстового сообщения с инструкцией
    await update.message.reply_text(
        "Ознакомьтесь с текстовой и видео-инструкцией к нашему боту"
    )

    # Отправка видео
    video_path = "./tgbot_data/fffx.mp4"  # путь к вашему видеофайлу
    await context.bot.send_video(
        chat_id=update.effective_chat.id, video=open(video_path, "rb")
    )
    # Теперь отправим форматированное текстовое сообщение
    formatted_message = (
        "<b>Как правильно задавать вопросы боту по астрологии?</b>\n"
        "1. Определите цель вопроса, о чем Вы хотите узнать.\n"
        "2. Вопрос должен содержать как можно больше деталей о том, что Вы хотите узнать.\n"
        "3. Вопрос должен состоять из целого предложения, исключая неоднозначные фразы.\n"
        "<b>Максимальное количество запросов в бот – 10 штук в сутки.</b>\n"
        "<b>Примеры некорректных запросов:</b>\n"
        "❌<s>Что такое дома в астрологии?<s>\n"
        "❌<s>Расскажите о планете Меркурий.<s>\n"
        "<b>Примеры корректных запросов:\n"
        "✅Какие основные характеристики 7-го дома в астрологии и как они связаны с партнерскими отношениями?\n"
        "✅Как транзит Сатурна через 10-й дом влияет на карьерные аспекты в астрологии?\n"
        "⚠️ В видеоинструкции показаны примеры неправильных и правильных запросов. Пожалуйста, обязательно ознакомьтесь с ними, чтобы не исчерпать свой лимит впустую.\n"
        "P.S. Если вы хотите перезапустить бота или что-то пошло не так, отправьте команду /start без пробелов или нажмите в левом нижнем углу кнопку меню и выберите команду /start\n"
        "Если вы хотите проверить количество запросов, которое у вас осталось, отправьте команду /limit без пробелов или нажмите в левом нижнем углу кнопку меню и выберите команду /limit\n"
        "Если вы хотите задать вопрос по работе с ботом или вы столкнулись с какой-то проблемой, вы можете отправить команду /help без пробелов или нажмите в левом нижнем углу кнопку меню и выберите команду /help\n"
    )
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=formatted_message, parse_mode="HTML"
    )


# Функция для следующего вопроса пользователя


async def ask_question(update: Update, _: CallbackContext) -> None:
    """Обрабатывает нажатие кнопки 'Отправить еще один вопрос'."""
    user_id = update.callback_query.from_user.id
    waiting_for_button_click[user_id] = False  # Сбрасываем состояние ожидания
    try:
        # Уведомление о получении callback-запроса
        await update.callback_query.answer()
        # Отправка сообщения
        await update.callback_query.message.reply_text("Жду от вас следующие вопросы")
    except BadRequest as e:
        logger.error(f"Ошибка при обработке callback-запроса: {e}")
        # Если возникла ошибка, переводим пользователя в состояние ожидания вопроса
        waiting_for_button_click[user_id] = False
        await update.callback_query.message.reply_text(
            "Что то пошло не так. Но вы можете продолжить свой диалог с ботом VEDA"
        )


waiting_for_button_click = {}


# Функция для обработки текстовых сообщений от пользователей
async def handle_message(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    user_id = user.id
    username = user.username

    # Проверяем, находится ли пользователь в процессе создания запроса помощи
    if help_request_state.get(user_id, False):
        # Обработка сообщения запроса на помощь
        await handle_help_message(update, context)
        help_request_state[user_id] = False  # Сбрасываем флаг
        return  # Завершаем функцию, чтобы предотвратить дальнейшую обработку текста

    # Проверяем, ожидается ли нажатие кнопки от пользователя
    if waiting_for_button_click.get(user_id, False):
        await update.message.reply_text(
            "Для продолжения диалога с ботом, пожалуйста, нажмите кнопку."
        )
        return

    # Проверяем, находится ли пользователь в процессе создания запроса помощи
    if help_request_state.get(user_id, False):
        # Обработка сообщения запроса на помощь
        await handle_help_message(update, context)
        help_request_state[user_id] = False  # Сбрасываем флаг
        return

    # Проверка лимита сообщений перед обработкой запроса
    messages_limit = access_checker.check_access(user_id, 2, "VEDA")["response"][
        "message_limit"
    ]
    print(messages_limit)
    if not (messages_limit):
        await update.message.reply_text(
            "Извините, но Ваш лимит сообщений на сегодня исчерпан. Лимит востановиться в 6:00 (Мск)."
        )
        return

    """Обрабатывает текстовое сообщение от пользователя."""
    user_query = update.message.text  # Извлекаем текст запроса пользователя
    logger.info(
        f"Пользователь @{username} (ID: {user_id}) отправил вопрос: {user_query}"
    )
    try:
        # Отправляем промежуточное сообщение пользователю
        await update.message.reply_text(
            f"""Анализирую Ваш запрос, @{username}.
        Обычно это занимает не более трех минут.
        💭 Уделите это время дыханию: раскройте плечи, понаблюдайте свой спокойный вдох и выдох.
        """
        )
        # Передаем текст пользователя в функцию обработки
        response = await process_query(user_query, user_id, username)
        reply_markup = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "👉Отправить еще один вопрос👈", callback_data="ask_question"
                    )
                ]
            ]
        )
        await update.message.reply_text(response, reply_markup=reply_markup)
        waiting_for_button_click[user_id] = True  # Устанавливаем флаг в True

        logger.info(
            f"Обработан запрос от пользователя: @{username} (ID: {user_id}), Вопрос: {user_query}"
        )
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")


# Функция команды /help
async def limit_command(update: Update, context: CallbackContext) -> None:
    """Отправляет пользователю информацию о количестве доступных сообщений."""
    user = update.effective_user
    user_id = user.id

    # Сброс ожидания нажатия кнопки
    waiting_for_button_click[user_id] = False

    # Проверяем, подтверждена ли электронная почта пользователя
    if context.user_data.get("email_verified", False):

        messages_limit = access_checker.check_access(user_id, 2, "VEDA")["response"][
            "message_limit"
        ]
        limit_count = access_checker.get_limits(user_id, "VEDA")
        if limit_count["status"] == "success":
            limit_count = limit_count["response"]
            limit_string = f"У вас осталось {limit_count} сообщений."
        else:
            limit_string = f"Произошла ошибка в получении коллличества сообщений, обратитесь в поддержку с указанием email"
        print(messages_limit)
        if messages_limit:
            await update.message.reply_text(limit_string)
        else:
            await update.message.reply_text(
                "Извините, но ваш лимит сообщений на сегодня исчерпан. Попробуйте снова завтра."
            )
    else:
        await update.message.reply_text(
            "Пожалуйста, сначала подтвердите вашу электронную почту."
        )


# Функция для обработки запроса пользователя


async def process_query(user_query: str, user_id: int, username: str) -> str:
    start_load_docs = time.time()
    docs = load_documents("./tgbot_data/db_veda.docx")
    all_splits = split_text_into_blocks(docs)
    end_load_docs = time.time()
    logger.info(
        f"Время загрузки и разбиения документов: {end_load_docs - start_load_docs} секунд"
    )

    start_bm25 = time.time()
    retriever = initialize_pinecone_retriever(bm25_encoder)
    end_bm25 = time.time()
    logger.info(f"Время инициализации BM25Retriever: {end_bm25 - start_bm25} секунд")

    start_chroma = time.time()
    persist_directory = "./tgbot_data/chroma"
    embedding = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
    vectorstore = initialize_chroma(all_splits, persist_directory, embedding)
    end_chroma = time.time()
    logger.info(f"Время инициализации Chroma: {end_chroma - start_chroma} секунд")

    start_ensemble = time.time()
    faiss_retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    ensemble_retriever = initialize_ensemble_retriever(retriever, faiss_retriever)
    end_ensemble = time.time()
    logger.info(
        f"Время инициализации EnsembleRetriever: {end_ensemble - start_ensemble} секунд"
    )

    start_query_model1 = time.time()
    response_text = await process_query_with_rag_chain(
        ensemble_retriever, user_query, "gpt-3.5-turbo-0125", "nemov186124/www", user_id
    )
    end_query_model1 = time.time()
    logger.info(
        f"Время обработки запроса первой моделью: {end_query_model1 - start_query_model1} секунд"
    )

    start_query_model2 = time.time()
    context_runnable = create_context_runnable(response_text)
    final_response = await process_query_with_rag_chain(
        context_runnable, user_query, "gpt-4o", "nemov186124/x", user_id
    )
    end_query_model2 = time.time()
    logger.info(
        f"Время обработки запроса второй моделью: {end_query_model2 - start_query_model2} секунд"
    )

    if final_response:
        access_checker.minus_message(user_id, "VEDA")
        logger.info(f"Лимит сообщений для пользователя {username} уменьшен на 1.")

    logger.info(f"===ОТВЕТ МОДЕЛИ=== Ответ: {final_response}")
    return final_response


# Основная функция, запускающая бота
async def main() -> None:
    application = Application.builder().token(telegram_bot_token).build()

    # Обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("limit", limit_command))

    # Обработка callback-запросов (например, от кнопок)
    application.add_handler(
        CallbackQueryHandler(ask_question, pattern="^ask_question$")
    )

    # Регистрация обработчиков текстовых сообщений
    # Важно поместить обработчик handle_email перед handle_message, чтобы корректно обработать ввод электронной почты
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_email)
    )
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    try:
        await application.initialize()
        await application.start()
        logger.info("Запуск бота...")
        await application.updater.start_polling()
        # Вставьте здесь другие асинхронные задачи, если они есть
        logger.info("Бот запущен.")
        # Например, await other_application()

        # Бесконечный цикл для поддержания работы
        while True:
            # Ожидание 10 секунд перед следующейитерацией
            await asyncio.sleep(10)
    finally:
        # Остановка обновления перед завершением приложения
        await application.updater.stop()
        await application.stop()
        await application.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
