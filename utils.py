import os
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from telegram import Update
from AccessChecker import access_checker
from telegram.ext import (
    CallbackContext,
)
from langchain_community.document_loaders import Docx2txtLoader
import Config
from dotenv import load_dotenv
from langchain_chroma import Chroma
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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


def check_email_in_database(email, tg_id):
    email_response = access_checker.check_email(
        tg_id=tg_id, email=email, bot="VEDA AI", bot_id=2
    )
    access = access_checker.check_access(tg_id=tg_id, bot_id=2, bot="VEDA AI")
    print(access)

    access_granted = False 

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
        "❌<s>Что такое дома в астрологии?</s>\n"
        "❌<s>Расскажите о планете Меркурий.</s>\n"
        "<b>Примеры корректных запросов:</b>\n"
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

def merge_similar_docs(docs, top_n=3):
    """
    Объединяет содержимое нескольких релевантных документов в единый контекст.
    """
    merged_content = "\n".join([doc.page_content for doc in docs[:top_n]])
    return merged_content

def rank_documents_by_relevance(docs, query, top_n=3):
    """
    Ранжирует документы по релевантности на основе TF-IDF и семантической близости.
    Возвращает top_n наиболее релевантных документов.
    """
    # Получаем тексты документов
    doc_texts = [doc.page_content for doc in docs]

    # Инициализация TF-IDF векторизатора
    vectorizer = TfidfVectorizer().fit_transform(doc_texts + [query])
    vectors = vectorizer.toarray()

    # Последний вектор — это вектор запроса
    query_vector = vectors[-1]

    # Считаем косинусное сходство между запросом и каждым документом
    cosine_similarities = np.dot(vectors[:-1], query_vector) / (
        np.linalg.norm(vectors[:-1], axis=1) * np.linalg.norm(query_vector)
    )

    # Ранжируем документы по косинусному сходству
    ranked_doc_indices = np.argsort(-cosine_similarities)[:top_n]
    ranked_docs = [docs[i] for i in ranked_doc_indices]

    return ranked_docs

def load_documents(file_path):
    loader = Docx2txtLoader(file_path)
    return loader.load()


def split_text_into_blocks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=10, add_start_index=True
    )
    return text_splitter.split_documents(docs)


def initialize_local_vectorstore(all_splits):
    embedding = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
    persist_directory = "./tgbot_data/chroma"
    if os.path.exists(persist_directory):
        vectorstore = Chroma(
            persist_directory=persist_directory, embedding_function=embedding
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=embedding,
            persist_directory=persist_directory,
        )
    return vectorstore

def initialize_local_vectorstore_from_two_source(all_splits, persist_directory):
    embedding = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
    if os.path.exists(persist_directory):
        vectorstore = Chroma(
            persist_directory=persist_directory, embedding_function=embedding
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=embedding,
            persist_directory=persist_directory,
        )
    return vectorstore