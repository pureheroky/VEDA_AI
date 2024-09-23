import asyncio
import os
import re
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)
import openai
from pinecone_text.sparse import BM25Encoder
from tiktoken import encoding_for_model
from AccessChecker import access_checker
from utils import (
    check_email_in_database,
    initialize_local_vectorstore_from_two_source,
    send_welcome_message,
    load_documents,
    split_text_into_blocks,
    rank_documents_by_relevance,
    merge_similar_docs,
)
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

bm25_encoder = BM25Encoder().default()

waiting_for_button_click = {}
help_request_state = {}


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


logging.basicConfig(
    level=logging.INFO,  # Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Config.LOG_FILE, encoding="utf-8"),  # Логировать в файл
        logging.StreamHandler(),  # Логировать в консоль
    ],
)
logger = logging.getLogger(__name__)


# Функция для подсчета токенов
def count_tokens(text: str, model: str = "gpt-4") -> int:
    encoding = encoding_for_model(model)
    return len(encoding.encode(text))


# Функция для расчета стоимости
def calculate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4") -> float:
    pricing = {
        "gpt-4o-mini": (0.00015, 0.0006),  # Исправленные значения для gpt-4o-mini
        "gpt-4": (0.00003, 0.00006),  # Исправленные значения для gpt-4
        "gpt-3.5-turbo-0125": (0.000002, 0.000002),  # Значения для gpt-3.5-turbo-0125
    }
    if model not in pricing:
        raise ValueError("Unknown model")
    cost_input, cost_output = pricing[model]

    return (input_tokens/1000) * cost_input + (output_tokens / 1000) * cost_output



async def process_query(user_query: str, user_id: int, username: str) -> str:
    # Загрузка и обработка документов
    docs = load_documents("./tgbot_data/db_veda.docx")
    all_splits = split_text_into_blocks(docs)

    if len(user_query.split()) <= 3:
        print("used short")
        vectorstore = initialize_local_vectorstore_from_two_source(all_splits, "./tgbot_data/chroma_short")
        docs = vectorstore.similarity_search(user_query, k=30)
        ranked_docs = rank_documents_by_relevance(docs, user_query, top_n=30)
        merged_docs = merge_similar_docs(ranked_docs, top_n=30)
    else:
        print("used long")
        vectorstore = initialize_local_vectorstore_from_two_source(all_splits, "./tgbot_data/chroma")
        docs = vectorstore.similarity_search(user_query, k=5)
        logger.info(docs)
        ranked_docs = rank_documents_by_relevance(docs, user_query, top_n=5)
        merged_docs = merge_similar_docs(ranked_docs, top_n=5)
    if not docs:
        logger.info(
            "Список документов пуст. Возможно, поиск не сработал или данные отсутствуют."
        )
        message_content = "Контекст не найден."
    else:
        message_content = re.sub(
            r"\s+",
            " ",
            "\n ".join(
                [
                    f"\nКонтекст-инструкция\n===============" + merged_docs + "\n"
                ]
            ),
        )

    logger.info(message_content)

    # Инструкции и системные сообщения
    system = "Вы — продвинутый AI-ассистент, специализирующийся на астрологии. Ваша задача — предоставлять точные, детализированные и содержательные ответы на вопросы пользователей по астрологии. Вы должны выдавать чёткие, информативные и полезные ответы, основывающиеся исключительно на переданной инструкции. Поддерживайте профессиональный, но доступный тон, обеспечивая, чтобы ваши объяснения были понятны как новичкам, так и продвинутым пользователям астрологии."
    instructions = """
    1. Вы должны формировать ответ ТОЛЬКО из той информации, которая была вам приведена в качестве контекст-инсрукции.
    2. Если информация из контекст-инструкции кажется вам неполной, ее все равно нельзя дополнять. Исключением является запрос, описанный в пункте 5.
    3. Будьте вежливы по отношению к пользователю, в ответе должна быть информация соответствующая запросу пользователя.
    4. Нельзя отступать от вопроса пользователя.
    """

    # Формирование сообщения для модели
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"Контекст: {message_content}\n\nИнструкции: {instructions} \n\nОтвет на задание: {user_query}",
        },
    ]
    logger.info(messages)
    # Запрос к OpenAI API
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    )

    final_response = completion.choices[0].message.content

    # Логирование и учет затрат
    input_tokens = count_tokens(system + message_content + user_query, model="gpt-4o-mini")
    output_tokens = count_tokens(final_response, model="gpt-4o-mini")
    cost = calculate_cost(input_tokens, output_tokens, model="gpt-4o-mini")
    logger.info(f"Стоимость запроса: {cost} USD")
    access_checker.add_cost(tg_id=user_id, price=cost, bot_id=2, bot="VEDA")

    if final_response:
        access_checker.minus_message(user_id, "VEDA")
        logger.info(f"Лимит сообщений для пользователя {username} уменьшен на 1.")

    logger.info(f"===ОТВЕТ МОДЕЛИ=== Ответ: {final_response}")
    return final_response


# Функция для создания RunnableLambda с предыдущим ответом
def create_context_runnable(response_text):
    return RunnableLambda(lambda input: response_text)


# Функция для обработки команды /start
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(
        "Пожалуйста, введите ваш Email на котором вы зарегистрированы на GetCourse."
    )
    context.user_data["awaiting_email"] = True
    context.user_data["email_verified"] = False


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
    await context.bot.send_message(Config.TECH_SUPPORT_ID, message_for_support)

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


# Функция для обработки ввода email
async def handle_email(update: Update, context: CallbackContext) -> None:
    user_id = update.effective_user.id

    if context.user_data.get("awaiting_email", False):
        if help_request_state.get(user_id, False):
            await handle_help_message(update, context)
            help_request_state[user_id] = False
            return

        email = update.message.text
        if check_email_in_database(email, user_id):
            context.user_data["awaiting_email"] = False
            context.user_data["email_verified"] = True
            waiting_for_button_click[user_id] = False
            await send_welcome_message(update, context)
        else:
            await update.message.reply_text(
                "Электронная почта не найдена. Пожалуйста, введите корректный адрес."
            )
    else:
        await handle_message(update, context)


# Функция для обработки текстовых сообщений от пользователей
async def handle_message(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    user_id = user.id

    if help_request_state.get(user_id, False):
        await handle_help_message(update, context)
        help_request_state[user_id] = False
        return

    if waiting_for_button_click.get(user_id, False):
        await update.message.reply_text(
            "Для продолжения диалога с ботом, пожалуйста, нажмите кнопку."
        )
        return

    messages_limit = access_checker.check_access(user_id, 2, "VEDA")["response"][
        "message_limit"
    ]
    if not messages_limit:
        await update.message.reply_text(
            "Извините, но Ваш лимит сообщений на сегодня исчерпан. Лимит восстановится в 6:00 (Мск)."
        )
        return

    user_query = update.message.text
    logger.info(
        f"Пользователь @{user.username} (ID: {user_id}) отправил вопрос: {user_query}"
    )

    try:
        await update.message.reply_text(
            f"Анализирую Ваш запрос, @{user.username}. Обычно это занимает не более трех минут. "
            "💭 Уделите это время дыханию: раскройте плечи, понаблюдайте свой спокойный вдох и выдох."
        )

        response = await process_query(user_query, user_id, user.username)
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
        waiting_for_button_click[user_id] = True

        logger.info(
            f"Обработан запрос от пользователя: @{user.username} (ID: {user_id}), Вопрос: {user_query}"
        )
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")


# Функция команды /limit
async def limit_command(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    user_id = user.id

    waiting_for_button_click[user_id] = False

    if context.user_data.get("email_verified", False):
        messages_limit = access_checker.check_access(user_id, 2, "VEDA")["response"][
            "message_limit"
        ]
        limit_count = access_checker.get_limits(user_id, "VEDA")
        if limit_count["status"] == "success":
            limit_count = limit_count["response"]
            limit_string = f"У вас осталось {limit_count} сообщений."
        else:
            limit_string = "Произошла ошибка в получении количества сообщений, обратитесь в поддержку с указанием email"
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


async def ask_question(update: Update, _: CallbackContext) -> None:
    """Обрабатывает нажатие кнопки 'Отправить еще один вопрос'."""
    user_id = update.callback_query.from_user.id
    waiting_for_button_click[user_id] = False  # Сбрасываем состояние ожидания
    try:
        # Уведомление о получении callback-запроса
        await update.callback_query.answer()
        # Отправка сообщения
        await update.callback_query.message.reply_text("Жду от вас следующие вопросы")
    except Exception as e:
        logger.error(f"Ошибка при обработке callback-запроса: {e}")
        # Если возникла ошибка, переводим пользователя в состояние ожидания вопроса
        waiting_for_button_click[user_id] = False
        await update.callback_query.message.reply_text(
            "Что то пошло не так. Но вы можете продолжить свой диалог с ботом VEDA"
        )


# Основная функция, запускающая бота
async def main() -> None:
    application = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()

    # Обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("limit", limit_command))

    application.add_handler(
        CallbackQueryHandler(ask_question, pattern="^ask_question$")
    )
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

        while True:
            await asyncio.sleep(10)
    finally:
        await application.updater.stop()
        await application.stop()
        await application.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
