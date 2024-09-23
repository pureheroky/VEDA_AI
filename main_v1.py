import shutil
import os
import re
import time
import openai
import logging
from logging.handlers import RotatingFileHandler
import sys
import requests
import asyncio
import json
from typing import Optional, Dict, Any
from aiogram import F, Bot, Dispatcher, types
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.filters.command import Command
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import Docx2txtLoader
from langchain.docstore.document import Document

from dotenv import load_dotenv
from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException
import tiktoken

user_data = {}

# Загрузка переменных среды
load_dotenv()
OPENAI_API_KEY = os.environ['YOUR_API_TOKEN']
TELEGRAM_BOT_TOKEN = os.getenv('YOUR_BOT_TOKEN')

class Config:
    ERROR_TELEGRAM_BOT_TOKEN = os.getenv("ERROR_TELEGRAM_BOT_TOKEN")
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    YOUR_API_TOKEN = os.getenv("OPENAI_API_KEY")
    BASE_URL = "https://api.astroguru.ru/api/v1"
    TG_CHAT_ID = "@astro_errors"
    DOC_PATH = "./tgbot_data/numero_db.docx"
    CHROMA_DIR = "./tgbot_data/chroma"
    LOG_FILE = "tgbot_logs/tgbot.log"
    VIDEO_PATH = "./tgbot_data/fffx.mp4"
    TECH_SUPPORT_ID = -1002011240000


class AccessChecker:
    def __init__(self, base_url: str, tg_bot_token: str, tg_chat_id: str, timeout: int = 10):
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
            await self.bot.send_message(chat_id=self.tg_chat_id, text=f"```json\n{error_message}\n```",
                                        parse_mode="MarkdownV2")
            self.logger.info("Error message sent to Telegram.")
        except Exception as err:
            self.logger.error(f"Failed to send error message to Telegram: {err}")

    def _post_request(self, endpoint: str, payload: Dict[str, Any], bot_name: str) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        self.logger.info(f"Sending POST request to {url} with payload: {payload}")
        try:
            response = requests.post(url, data=json.dumps(payload), timeout=self.timeout)
            response.raise_for_status()
            self.logger.info(f"Received response: {response.json()}")
            return response.json()
        except (HTTPError, ConnectionError, Timeout, RequestException) as req_err:
            self.logger.error(f"Request error occurred: {req_err}")
            asyncio.create_task(
                self._send_error_to_telegram(error=json.dumps({"RequestException": str(req_err) + payload["email"]}),
                                             bot_name=bot_name))
            return {"status": "error", "message": f"Request error occurred: {req_err}"}
        except Exception as err:
            self.logger.error(f"An unexpected error occurred: {err}")
            asyncio.create_task(
                self._send_error_to_telegram(error=json.dumps({"Exception": str(err)}), bot_name=bot_name))
            return {"status": "error", "message": f"An unexpected error occurred: {err}"}

    def check_email(self, tg_id: int, email: str, bot: str, bot_id: int) -> Dict[str, Any]:
        payload = {"email": email, "tg_id": tg_id, "bot_id": 3}
        return self._post_request(endpoint="/check_email", payload=payload, bot_name=bot)

    def get_limits(self, tg_id: int, bot: str) -> dict[str, Any]:
        payload = {"tg_id": tg_id, "bot_id": 3}
        return self._post_request(endpoint="/veda.get_messages_limit", payload=payload, bot_name=bot)

    def check_access(self, tg_id: int, bot_id: int, bot: str, task: Optional[str] = None) -> Dict[str, Any]:
        payload = {"tg_id": tg_id, "bot_id": 3}
        if task:
            payload["task"] = task
        return self._post_request(endpoint="/check_access", payload=payload, bot_name=bot)

    def get_tasks(self, tg_id: int, bot_id: int, bot: str):
        payload = {"tg_id": tg_id, "bot_id": 3}
        return self._post_request(endpoint="/kurator.get_tasks", payload=payload, bot_name=bot)

    def add_task(self, tg_id: int, bot_id: int, task_number: int, bot: str):
        payload = {"tg_id": tg_id, "bot_id": 3, "task_number": task_number}
        return self._post_request(endpoint="/kurator.add_task", payload=payload, bot_name=bot)

    def add_cost(self, tg_id: int, price: float, bot_id: int, bot: str):
        payload = {"tg_id": tg_id, "price": price, "bot_id": 3}
        return self._post_request(endpoint="/token.add", payload=payload, bot_name=bot)

access_checker = AccessChecker(base_url=Config.BASE_URL, tg_bot_token=Config.ERROR_TELEGRAM_BOT_TOKEN,
                               tg_chat_id=Config.TG_CHAT_ID)

# Создание объекта logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Установка уровня логирования

# Настройка обработчика для записи логов в файл
file_handler = RotatingFileHandler(
    'app.log', maxBytes=1048576, backupCount=5, encoding='utf-8')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Настройка обработчика для вывода логов в терминал
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Добавление обработчиков к логгеру
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

class CustomFilter(logging.Filter):
    def filter(self, record):
        if record.name == 'httpx' and 'getUpdates' in record.getMessage():
            return False
        if record.name == 'urllib3.connectionpool' and 'ReadTimeoutError' in record.getMessage():
            return False
        if record.name == 'langchain_core.tracers.langchain' and '409 Client Error' in record.getMessage():
            return False
        return True

def load_document_text(url: str) -> str:
    match_ = re.search('/document/d/([a-zA-Z0-9-_]+)', url)
    if match_ is None:
        raise ValueError('Invalid Google Docs URL')
    doc_id = match_.group(1)
    response = requests.get(
        f'https://docs.google.com/document/d/{doc_id}/export?format=txt')
    response.raise_for_status()
    return response.text

async def answer_index(system, topic, search_index, user_message, temp=0.2, verbose=0):
    logger.info(topic)
    docs = search_index.similarity_search(topic, k=1)

    if not docs:
        logger.info("Список документов пуст. Возможно, поиск не сработал или данные отсутствуют.")
        message_content = "Контекст не найден."
    else:
        message_content = re.sub(r'\{2}+', ' ', '\n '.join(
            [f'\nКонтекст-инструкция\n===============' + doc.page_content + '\n' for i, doc in enumerate(docs)]))

    message_content = re.sub(r'\s+', ' ', '\n '.join(
        [f'\nКонтекст-инструкция\n===============' + doc.page_content + '\n' for i, doc in enumerate(docs)]))

    for i, doc in enumerate(docs):
        if not doc.page_content.strip():  # Проверка на пустоту контента
            logger.info(f"Документ {i} пустой или содержит только пробелы.")
        else:
            logger.info(f"Документ {i} контент: {doc.page_content[:100]}")  # Вывод первых 100 символов для проверки

    logger.info(message_content)
    # Логирование отправляемых данных
    logger.info(
        f"Отправка следующего текста в модель GPT: {message_content} | Текст студента: {user_message} ")

    old_instructions = """
    Подробная инструкция по проверке домашних заданий:
        ОБРАЩАЙ БОЛЬШЕ ВНИМАНИЕ НА ТЕКСТОВОЕ ОПИСАНИЕ КАРТЫ
        1. Получение данных
        Шаг 1.1: Получить текст задания от ученика. Текст задания может содержать несколько частей: описание задания, что должен был сделать студент, пример выполненного задания, текст лекций.
        Шаг 1.2: Получить правильные ответы и критерии оценки. Эти ответы и критерии могут быть заранее подготовлены учителем или взяты из учебных материалов.
        2. Разбор задания
        Шаг 2.1: Разделить текст задания на части:
        Что должен был выполнить студент.
        Пример выполненного задания.
        Текст лекций.
        Шаг 2.2: Извлечь важную информацию из каждой части для последующей проверки.
        3. Проверка каждого ответа студента
        Шаг 3.1: Начать с проверки первого аспекта задания (например, описание кармы по положению в знаке хозяина 10-го дома).
        Шаг 3.2: Сравнить ответ студента с правильным ответом из лекций и примера выполненного задания.
        Шаг 3.3: Если ответ студента совпадает с правильным, пометить его как "правильный".
        Шаг 3.4: Если ответ студента не совпадает с правильным, пометить его как "неправильный" и сохранить причину ошибки.
        Шаг 3.5: Повторить шаги 3.1 - 3.4 для всех аспектов задания.
        4. Формирование ответа ученику
        Шаг 4.1: После проверки всех аспектов задания, бот составляет два списка:
        Список правильных ответов.
        Список неправильных ответов с объяснением ошибки.
        Шаг 4.2: Бот проверяет, чтобы один и тот же аспект не оказался одновременно в списке правильных и неправильных ответов.
        5. Валидация и проверка на противоречия
        Шаг 5.1: Бот проверяет каждый аспект из списка правильных ответов.
        Шаг 5.2: Бот убеждается, что этот аспект не находится в списке неправильных ответов.
        Шаг 5.3: Если бот находит противоречие, он повторно проверяет этот аспект:
        Если ответ действительно правильный, бот удаляет его из списка неправильных ответов.
        Если ответ действительно неправильный, бот удаляет его из списка правильных ответов.
        Шаг 5.4: Бот повторяет шаги 5.1 - 5.3 для всех аспектов.
        6. Итоговый вывод
        Шаг 6.1: Бот составляет итоговый отчет для ученика:
        В отчете перечислены все правильные ответы ученика.
        В отчете перечислены все неправильные ответы с объяснением ошибок.
        В отчете указаны все отсутствующие или недоделанные задания
        В отчете в конце пишется итог по всему выполненному домашнему заданию
        Шаг 6.2: Бот отправляет отчет ученику.

    Обязательно выполнить все 5 шагов указанных выше. Стоит учитывать что ученик не может делать опечаток, надо воспринимать информацию в контексте того что ученик не мог сделать опечатку, его ответы либо правильные либо не правильные
    """

    instructions = """
    Вы - эксперт по проверке домашних заданий в области астрологии. Ваша основная задача - предоставить точный и поддерживающий отзыв студенту, основываясь исключительно на представленном контексте и материалах лекций. Ответы студента должны строго соответствовать содержимому лекций.

    Структура ответа:
    1. Позитивная обратная связь: Подчеркните, что студент сделал правильно в соответствии с лекционными материалами.
    2. Обнаруженные ошибки: Укажите на ошибки и несоответствия, но в мягкой форме, предложите пересмотреть соответствующие материалы лекций.
    3. Грубые ошибки: Если в ответе студента присутствует информация, относящаяся к другому заданию, или материал, не связанный с лекцией, подчеркните это как грубую ошибку и укажите на недопустимость таких ошибок.
    4. Рекомендации: Дайте советы для улучшения работы на основе лекционных материалов.

    Пример ответа:
    "Ваш анализ положения планет в домах выполнен правильно. Отличная работа с интерпретацией Юпитера. Обратите внимание на влияние Луны, где есть небольшие несоответствия лекциям. Рекомендую пересмотреть тему 'Знаки и дома'. Продолжайте в том же духе!"

    Важно:
    - Ответ должен быть структурирован, ясным и поддерживающим.
    - Ответы студента должны полностью соответствовать контексту лекций; любые отклонения от контекста лекций рассматриваются как грубые ошибки.
    - Если студент даёт ответ на другое задание или использует информацию вне контекста лекций, обязательно отметьте это как грубую ошибку и подчеркните важность строгого следования материалам лекций.
    - Избегайте перегрузки информацией и не предоставляйте прямых ответов на задания.
    - Четко говори об ошибках, если в карте другие знаки, выноси как ошибку, но не говори какие знаки должны быть
    - Если тектовое описание карты не соответствует ответу ученика, скажи об этом
    - Не используй выражения такие как: очень грубая ошибка, говори просто ошибка и т.д.
    """


    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"Контекст: {message_content}\n\nИнструкции: {instructions} \n\nОтвет на задание: {user_message}",
        },
    ]

    input_tokens = enc.encode(system + message_content + topic)
    global input_token_count
    input_token_count = len(input_tokens)

    completion = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        temperature=temp
    )

    output_tokens = enc.encode(completion.choices[0].message.content)
    global output_token_count
    output_token_count = len(output_tokens)

    global total_token_count
    total_token_count = input_token_count + output_token_count

    return completion.choices[0].message.content

async def add_completed_task(user_id: int, task_number: int):
    access_checker.add_task(tg_id=user_id, bot_id=3, task_number=task_number, bot="ASTRO KURATOR")


async def check_user_email_in_db(user_email: str, tg_id: int):
    email_response = access_checker.check_email(tg_id=tg_id, email=user_email, bot="ASTRO KURATOR", bot_id=3)
    order_exist = False
    access_granted = False

    if email_response["status"] == "success":
        email_access = email_response["response"].get("email_exist", False)
        access = access_checker.check_access(tg_id=tg_id, bot_id=1, bot="VEDA")
        order_exist = access['response'].get('order_exist', False)
        access_granted = email_access and order_exist
    else:
        if email_response["status"] == "error" or "Unauthorized access" in email_response["message"]:
            access_granted = False

    return access_granted


async def check_access(user_id: int):
    has_active_subscription = False
    active_subscription = access_checker.check_access(user_id, 3, "VEDA")['response']['order_exist']
    if active_subscription:
        has_active_subscription = True
    return has_active_subscription

async def confirm_email(call: types.CallbackQuery) -> None:
    user_id = call.from_user.id
    if user_data[user_id].get('WAITING_FOR_CONFIRM_EMAIL', False):
        email = user_data[user_id].get('email', None)
        email_exists = await check_user_email_in_db(user_email=email, tg_id=user_id)
        if email_exists:
            await bot.send_message(chat_id=user_id, text=f"{email} подтвержден. Добро пожаловать!")
            await get_task(user_id)
        else:
            await bot.send_message(chat_id=user_id, text='К сожалению, этот email не найден. Пожалуйста, проверьте правильность введенного email или используйте другой.')


async def start(message: types.Message) -> None:
    user_id = message.from_user.id
    user_data[user_id] = {}
    user_data[user_id]['CHANGE_EMAIL'] = False
    user_data[user_id]['WAITING_FOR_CONFIRM_EMAIL'] = False
    user_data[user_id]['WAITING_FOR_EMAIL'] = True
    user_data[user_id]['WAITING_FOR_TASK'] = False
    await bot.send_message(chat_id=user_id,
                           text='Пожалуйста введите Ваш Email по которому Вы зарегистрированы на GetCourse.')


async def main_message_handler(message: types.Message):
    user_id = message.from_user.id
    user_id_db = user_data[user_id].get('user_id_db')
    if user_data[user_id].get('WAITING_FOR_TASK', False):
        try:
            logger.info("ВНИМАНИЕ! СТУДЕНТ ОТПРАВИЛ НОВОЕ ДОМАШНЕЕ ЗАДАНИЕ")
            user_message = message.text
            await message.answer(
                "Ваше задание получено и отправлено на проверку, пожалуйста, ожидайте. \n\n Как только оно будет проверено, я пришлю Вам обратную связь.")
            logger.info("Студенту отправлено уведомление о получении задания.")

            task_title = user_data[user_id]['TITLE']
            combined_message = task_title
            user_email = user_data[user_id].get('email', 'Неизвестно')
            task_number = user_data[user_id]['TASK_NUMBER']
            logger.info(
                f"текст по которому произошел поиск в ВекБД:: {combined_message}")
            feedback = await answer_index(system, combined_message, db, user_message)
          
            logger.info("Ответ от GPT получен: %s", feedback)

            input_cost = (input_token_count / 1000) * INPUT_TOKEN_PRICE
            output_cost = (output_token_count / 1000) * OUTPUT_TOKEN_PRICE
            total_cost = input_cost + output_cost
            logger.info(f"Стоимость входных токенов: {input_cost}")
            logger.info(f"Стоимость выходных токенов: {output_cost}")
            logger.info(f"Общая стоимость запроса: {total_cost}")
            access_checker.add_cost(tg_id=user_id, price=total_cost, bot_id=3, bot="ASTRO KURATOR")
            await message.answer(feedback)

        except Exception as e:
            logger.error("Ошибка при обработке сообщения: %s", e)
            await message.answer('Произошла ошибка при обработке вашего задания.')

        task_number_int = int(user_data[user_id]['TASK_NUMBER'])
        await add_completed_task(user_id, task_number_int)
        await send_task_again_option(user_id)

        user_data[user_id]['WAITING_FOR_TASK'] = False
        return
    if user_data[user_id].get('WAITING_FOR_EMAIL', False):
        user_data[user_id]['email'] = message.text.lower()
        builder = InlineKeyboardBuilder()
        builder.add(types.InlineKeyboardButton(
            text="Подтвердить",
            callback_data="confirm_email"))
        builder.add(types.InlineKeyboardButton(
            text="Изменить",
            callback_data="change_email"))
        builder.adjust(1)
        email = user_data[user_id]['email']
        await message.answer(f'Подтвердите, пожалуйста, свой email {email}, или измените его',
                             reply_markup=builder.as_markup())
        user_data[user_id]['WAITING_FOR_EMAIL'] = False
        user_data[user_id]['WAITING_FOR_CONFIRM_EMAIL'] = True
        user_data[user_id]['CHANGE_EMAIL'] = False
        return

async def change_email(call: types.CallbackQuery) -> None:
    user_id = call.from_user.id
    if user_data[user_id].get('WAITING_FOR_CONFIRM_EMAIL', False):
        user_data[user_id]['CHANGE_EMAIL'] = True
        user_data[user_id]['WAITING_FOR_EMAIL'] = True
        await bot.send_message(chat_id=user_id, text='Введите email заново:')

async def get_user_lesson(call: types.CallbackQuery) -> None:
    user_id = call.from_user.id
    user_data[user_id]['WAITING_FOR_TASK'] = True
    num_task = call.data.split('_')[-1]
    user_data[user_id]['TITLE'] = TASK_TITLES[call.data]
    user_data[user_id]['TASK_NUMBER'] = num_task
    await bot.send_message(chat_id=user_id,
                           text=f'''Отлично! Вы выбрали Задание №{num_task}.\nТеперь напишите текст Вашего выполненного домашнего задания ОДНИМ сообщением.\nВ формате:\n-Ваши интерпретации и анализ представленной астрологической информации''')
    return

async def get_task(user_id):
    has_access = await check_access(user_id)

    if not has_access:
        await bot.send_message(chat_id=user_id, text="Ваша подписка истекла, необходимо продлить.")
        return

    completed_tasks_response = access_checker.get_tasks(tg_id=user_id, bot_id=3, bot="ASTRO KURATOR")
    completed_tasks = completed_tasks_response['response']

    # Отладка: выводим тип и содержимое completed_tasks
    logger.info(f"Тип completed_tasks: {type(completed_tasks)}, содержимое: {completed_tasks}")

    if not isinstance(completed_tasks, list):
        logger.error("completed_tasks должен быть списком")
    # Если completed_tasks - это список целых чисел, используем его напрямую
    if completed_tasks is None:
        available_task_numbers = set()
    else:
        available_task_numbers = set(completed_tasks)

    logger.info(f"Доступные номера заданий: {available_task_numbers}")

    builder = InlineKeyboardBuilder()
    for task_key, task_title in TASK_TITLES.items():
        task_number = int(task_key.split('_')[-1])
        if task_number not in available_task_numbers:
            builder.add(types.InlineKeyboardButton(
                text=f'№{task_number}', callback_data=task_key))
    builder.adjust(3)
    await bot.send_message(chat_id=user_id,
                           text='Выберите, пожалуйста, № задания по которому у Вас готово домашнее задание',
                           reply_markup=builder.as_markup())

async def send_task_again_option(user_id):
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="Отправить еще одно задание", callback_data="new_task"))
    await asyncio.sleep(10)  # Задержка перед отправкой кнопки
    await bot.send_message(chat_id=user_id, text="Для отправки еще одного задания нажмите кнопку ниже⬇️",
                           reply_markup=builder.as_markup())

async def new_task(call: types.CallbackQuery) -> None:
    user_id = call.from_user.id
    if user_id in user_data:
        user_data[user_id]['WAITING_FOR_TASK'] = False
        await get_task(user_id)
    else:
        logger.warning(f"Ключ {user_id} отсутствует в словаре user_data")
        await bot.send_message(chat_id=user_id, text="Пожалуйста, начните с команды /start")

async def main():
    dp.message.register(start, Command("start"))
    dp.message.register(main_message_handler, F.text & ~F.command)

    # dp.callback_query.register(start_bot, F.data == 'start')
    dp.callback_query.register(get_user_lesson, F.data.startswith("task"))
    dp.callback_query.register(confirm_email, F.data == 'confirm_email')
    dp.callback_query.register(change_email, F.data == 'change_email')
    dp.callback_query.register(new_task, F.data == 'new_task')
    dp.callback_query.register(
        send_task_again_option, F.data == 'send_task_again_option')

    await dp.start_polling(bot)

# Заменить модель, если будет использоваться другая модель, например gpt-4
enc = tiktoken.encoding_for_model("gpt-4o")

# Замените на реальную цену за 1000 входных токенов определенной модели
INPUT_TOKEN_PRICE = 0.010
# Замените на реальную цену за 1000 выходных токенов определенной модели
OUTPUT_TOKEN_PRICE = 0.030

user_data = {}

GOOGLE_FORM_URL = 'https://docs.google.com/forms/d/e/1FAIpQLSchZbkkpfGfeZB6EJym_FU2u2yVB-iOeE76X3wRyPCPoEDxXw/formResponse'

TASK_TITLES = {
    'task_1': 'ЗАДАНИЕ №1 «Анализ Личности через Астрологические Знаки» | ТЕКСТОВОЕ ОПИСАНИЕ КАРТЫ: <<Восходящий знак - Рак, Знак Луны - Рыбы, Накшатра Луны - Уттара Бхадрапада, Планета в доме личности - Юпитер>>',
    'task_2': 'ЗАДАНИЕ №2 «Определение Сильных и Слабых Планет в Личностном Развитии» | ТЕКСТОВОЕ ОПИСАНИЕ КАРТЫ: <<Солнце сильное с высокой иштой (хорошего качества), Луна сильная с высокой иштой (хорошего качества), Марс сильный с высокой иштой (хорошего качества), Меркурий сильный с высокой иштой (хорошего качества), Юпитер сильный с высокой иштой (хорошего качества), Венера слабая с высокой каштой (плохого качества), Сатурн сильный с высокой каштой (плохого качества)>>',
    'task_3': 'ЗАДАНИЕ №3 «Расчет Варны и Анализ Талантов через Астрологические Дома» | ТЕКСТОВОЕ ОПИСАНИЕ КАРТЫ: <<Знак в лагне - Козерог, Планет в лагне нет, 100%-е аспекты на лагну: Венера, Лагнеша - Сатурн, Знак лагнеши - Весы, Знак Солнца - Телец, Знак Луны - Рыбы, Сильнейшая планета по шад бале - Солнце, Знак Венеры - Рак, Знак Юпитера - Скорпион>>',
    'task_4': 'ЗАДАНИЕ №4 «Кармический Путь к Успеху через Положение Планет» | ТЕКСТОВОЕ ОПИСАНИЕ КАРТЫ: <<Управитель 10 дома Меркурий в Тельце - в знаке большого друга, Влияние планет на 10-й дом - нахождение Кету в 10 доме > аспекты Раху > аспекты Юпитера > аспекты Марса, Управитель 1 дома - Юпитер, Влияние планет на 1-й дом - нахождение Луны в 1 доме > аспект Марса, Накшатра управителя 10-го дома Меркурия - Рохини, Накшатра Луны - Мула>>',
    'task_5': 'ЗАДАНИЕ №5 «Источники Дохода и Финансовая Карма через SAV 2-го и 11-го Домов» | ТЕКСТОВОЕ ОПИСАНИЕ КАРТЫ: <<2 дом – 27 бинду - управитель в 8 доме, 11 дом – 25 бинду - управитель в 7 доме>>',
    'task_6': 'ЗАДАНИЕ №6 «Финансовая Ёмкость и Коррекция Юпитера для Процветания» | ТЕКСТОВОЕ ОПИСАНИЕ КАРТЫ: <<Юпитер ретроградный, сильный, хорошего качества, в падении, в 9 доме>>',
    'task_7': 'ЗАДАНИЕ №7 «АНАЛИЗ КАРМИЧЕСКИХ АСПЕКТОВ ЧЕРЕЗ ХОЗЯИНА 7-ГО ДОМА И РЕТРОГРАДНЫЕ ПЛАНЕТЫ» | ТЕКСТОВОЕ ОПИСАНИЕ КАРТЫ: <<Хозяин 7-го дома - Меркурий в Раке в 8-ом доме. Хозяин 7-го дома - ретроградный Меркурий. Влияние ретроградных планет на 7-й дом - аспект ретроградного Марса и Меркурия. Венера в знаке Льва в 9-м доме.>>',
    'task_8': '<<ЗАДАНИЕ №8 «Совместимости Партнеров в Астрологических Картах: Эмоциональная, Внешняя, Сексуальная, Финансовая и Кармическая Совместимости» | ТЕКСТОВОЕ ОПИСАНИЕ КАРТЫ: <<Расстояние между Лунами - 8/6. Расстояние между восходящими знаками - 6/8. Расстояние между Венерой и Марсом - 4/10. Расстояние между Юпитерами - 4/10. Кету при наложении карт не имеет соединений с другими планетами, попадает в 5-й дом. Раху при наложении карт не имеет соединений с другими планетами, попадает в 11-й дом.>>',
    'task_9': 'ЗАДАНИЕ №9 «Привлекательные Качества партнера, Внешности Партнера и Места Встречи Согласно Венере и 7-му Дому в Астрологической Карте» | ТЕКСТОВОЕ ОПИСАНИЕ КАРТЫ: <<Венера в знаке Овна, Накшатра хозяина 7-го дома - Рохини, 7-й дом от Венеры - 11-й, хозяин дома – Весы>>',
    'task_10': 'ЗАДАНИЕ №10 «Сила личности и болезней по SAV 1-го и 6-го домов, вероятные болезни и рекомендации для улучшения здоровья» | ТЕКСТОВОЕ ОПИСАНИЕ КАРТЫ: <<Сила 1-го дома - 24 SAV, Сила 6-го дома - 35 SAV, Сила 6-го дома сильнее силы 1-го дома, В 6-ом доме Раху и аспект Юпитера в знаке Тельца>>',
    'task_11': 'ЗАДАНИЕ №11 «Периоды и подпериоды по дате рождения, события, сферы жизни по домам, Персональный год и рекомендации» | ТЕКСТОВОЕ ОПИСАНИЕ КАРТЫ: <<Период - Меркурий, Подпериод - Раху Меркурий находится в 8-м доме эзотерики, владеет 9 домом религий и паломнических путешествий и 12-м домом потерь, заграничных путешествий Меркурий в знаке большого друга. Меркурий средней силы хорошего качества Раху находится в 10-м доме карьеры Число Персонального года - 4>>'
}

openai.api_key = OPENAI_API_KEY
persist_directory = 'astro_ai_data/chroma'
chroma_file_path = os.path.join(persist_directory)
logger.info(chroma_file_path)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

handler = RotatingFileHandler(
    'bot.log',
    maxBytes=1 * 1024 * 1024,  # Максимальный размер файла 1 МБ
    backupCount=5,  # Количество сохраняемых копий файлов логов
    encoding='utf-8')
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levellevel)s - %(message)s'))
handler.addFilter(CustomFilter())  # Добавление фильтра к обработчику
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Установка уровня логирования

# Настройка обработчика для записи логов в файл
file_handler = RotatingFileHandler(
    'app.log', maxBytes=1048576, backupCount=5, encoding='utf-8')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Настройка обработчика для вывода логов в терминал
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Добавление обработчиков к логгеру
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
if os.path.exists(chroma_file_path):
    shutil.rmtree(chroma_file_path)
    logger.info("Существующий каталог базы данных Chroma был удален.")

    # Создание нового экземпляра Chroma
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    logger.info("Создан пустой индекс Chroma.")

    # Загрузка и разделение документов
    loader = Docx2txtLoader("db_neuro.docx")
    try:
        documents = loader.load()
        logger.info(f"Загружено {len(documents)} документов из 'db_neuro.docx'.")
        text_splitter = CharacterTextSplitter(separator="*", chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        logger.info(f"Документы успешно разделены на {len(docs)} частей.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке или разделении документов: {e}")
        docs = []

    # Преобразование документов в формат, который принимает Chroma
    # Преобразуем содержимое каждого документа в строку, если это не так
    formatted_docs = [Document(page_content=str(doc)) for doc in docs]

    # Добавление документов в Chroma
    try:
        db.add_documents(formatted_docs)
        logger.info("Документы добавлены в индекс Chroma.")
    except Exception as e:
        logger.error(f"Ошибка при добавлении документов в индекс Chroma: {e}")
else:
    loader = Docx2txtLoader("db_neuro.docx")
    documents = loader.load()
    logger.info("Документы успешно загружены.")
    logger.info(f"Загружено {len(documents)} документов из 'db_neuro.docx'.")

    text_splitter = CharacterTextSplitter(separator="*", chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    db = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)
    logger.info("Создан новый файл Chroma.")
logger.info(f"Методы и атрибуты объекта db: {dir(db)}")

system = """
Роль: Вы - нейро-куратор домашних заданий по астрологии в онлайн-школе. Ваша задача - проверять работы студентов, обеспечивая обратную связь в элегантной, приятной и дружелюбной форме. Большинство ваших студентов - новички в астрологии, стремящиеся изучить эту область.

Инструкция:
1. Проверьте правильность ответов студентов, сравнивая их с предоставленными лекционными материалами и образцами.
2. Укажите, что сделано правильно, подчеркните положительные моменты.
3. В случае ошибок, укажите их в мягкой форме, дайте рекомендации по улучшению и укажите на темы, которые нужно повторить.
4. Если студент что-то упустил, направьте его к соответствующим материалам без чрезмерной критики.
5. Формируйте отчет, используя не менее 15 предложений, с фокусом на поддержке и мотивации.
6. Учитывай колличество ошибок человека и от этого выбирай градацию из трех вариантов отлично/хорошо/очень хорошо
7. Будь придельно честен с учеником, хвали, но не перехваливай его
8. Если студент сделал ошибку, обзятельно распеши ему, как должно быть

Пример обратной связи:
"Ваш анализ положения планет в домах выполнен отлично/хорошо/очень хорошо, особенно правильное определение влияния Юпитера на карму. Обратите внимание на влияние Луны, здесь было небольшое несоответствие лекциям. Рекомендую пересмотреть тему о домах и знаках, чтобы улучшить понимание."

- Правиьлыным ответом считай ПРИМЕР ВЫПОЛНЕННОГО ЗАДАНИЯ (ОБРАЗЕЦ) и отталкивайся от него, если ответ ученика не соответствует указывай все ошибка
- Пиши так, будто пишет реальный человек(без использования сторонних символов ##)
- Нельзя ссылаться на текстовое описание карт, нужно говорить просто о карте
- Общайся в дружелюбном и веживом тоне
- Если мысли правильные, но не относияся к задании, скажи, что студент мыслит верно, но задание решено не верно, и укажи где не верно


Формат ответа:
1. "Вы правильно определили ...".
2. "Возможно, вы упустили ...".
3. "Для улучшения, обратите внимание на ...".
"""

TELEGRAM_TOKEN = os.getenv('YOUR_BOT_TOKEN')
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

if __name__ == "__main__":
    asyncio.run(main())
