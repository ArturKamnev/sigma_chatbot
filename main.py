# import logging
# from telegram import Update
# from telegram.constants import ChatAction  # Добавляем импорт ChatAction
# from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
# from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# # Настройка логирования
# logging.basicConfig(
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
#     level=logging.INFO
# )
# logger = logging.getLogger(__name__)

# # Инициализация модели
# print("Загружаем модель...")
# tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
# model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
# set_seed(42)

# # Настройка специальных токенов
# tokenizer.pad_token = tokenizer.eos_token
# model.config.pad_token_id = model.config.eos_token_id

# def generate_answer(user_text: str) -> str:
#     """Генерирует ответ с улучшенной фокусировкой на теме"""
#     try:
#         # Усиленный промпт с явными инструкциями
#         prompt = (
#             "Ответь кратко и точно на русском языке. "
#             f"Вопрос: {user_text}\n"
#             "Четкий ответ:"
#         )
        
#         inputs = tokenizer(
#             prompt,
#             return_tensors='pt',
#             max_length=256,  # Уменьшено для фокусировки
#             truncation=True
#         )
        
#         outputs = model.generate(
#             inputs.input_ids,
#             attention_mask=inputs.attention_mask,
#             max_new_tokens=100,           # Уменьшено для краткости
#             repetition_penalty=2.0,       # Увеличено против повторов
#             temperature=0.3,              # Понижено для точности
#             top_k=20,                     # Более строгий отбор
#             top_p=0.7,
#             do_sample=True,
#             pad_token_id=tokenizer.eos_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#             num_return_sequences=1,
#             no_repeat_ngram_size=2        # Блокировка повторяющихся фраз
#         )
        
#         full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         # Улучшенная пост-обработка ответа
#         if "Четкий ответ:" in full_response:
#             response = full_response.split("Четкий ответ:")[-1]
#             # Удаляем все после первого непунктуационного символа
#             return response.split("\n")[0].strip().split(".")[0] + "."
#         return full_response.split("\n")[0].strip()
    
#     except Exception as e:
#         logger.error(f"Ошибка генерации: {e}")
#         return "Извините, не удалось обработать запрос"

# async def answer_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     """Обработчик сообщений с фильтрацией"""
#     user_text = update.message.text.strip()
    
#     if len(user_text) < 3:
#         await update.message.reply_text("Пожалуйста, задайте более развернутый вопрос")
#         return
    
#     # Исправленный способ отправки действия "печатает"
#     await context.bot.send_chat_action(
#         chat_id=update.effective_chat.id, 
#         action=ChatAction.TYPING
#     )
    
#     answer = generate_answer(user_text)
    
#     if len(answer) > 3000:
#         answer = answer[:3000] + "... [сообщение сокращено]"
        
#     await update.message.reply_text(answer)

# async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     """Обработка команды /start."""
#     await update.message.reply_text("Привет! Задай свой вопрос.")

# def main():
#     application = ApplicationBuilder().token("7602048756:AAGCQrULx5hfGsVDeHqNfqy4MaOVGKyEOJg").build()

#     application.add_handler(CommandHandler("start", start))
#     application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, answer_message))

#     application.run_polling()

# if __name__ == '__main__':
#     main()
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# from openai import OpenAI

# client = OpenAI(
#   api_key="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"
# )

# completion = client.chat.completions.create(
#   model="gpt-4o-mini",
#   store=True,
#   messages=[
#     {"role": "user", "content": "Can you say me what is DeepSeek?"}
#   ]
# )

# print(completion.choices[0].message);
# 8066853463:AAGEUpF-kpBu8he8zBX9oS0HkBUBlLlwh48
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
import os
import io
import uuid
import logging
import asyncio
import base64
import tempfile
from typing import Tuple

# --- Дополнительные импорты для Flask и потока ---
from flask import Flask, request
from threading import Thread
import time
import requests
# -------------------------------------------------

from gtts import gTTS
from mutagen.mp3 import MP3  # <-- для чтения длительности MP3
from openai import OpenAI
from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackQueryHandler,
)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Глобальная память для хранения последнего текстового сообщения пользователя (для контекста)
chat_memory = {}
# Глобальное состояние голосового режима для каждого чата
voice_mode_enabled = {}  # ключ: chat_id, значение: bool

class Config:
    TELEGRAM_TOKEN = TELEGRAM_TOKEN
    OPENAI_API_KEY = OPENAI_API_KEY
    OPENAI_MODEL_TEXT = "gpt-4o-mini"
    WHISPER_MODEL = "whisper-1"

# ------------------------------------------------------
# Логирование
# ------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------
# Генерация MP3-файла через gTTS
# ------------------------------------------------------
def generate_voice_answer_gtts(text: str) -> str:
    """
    Генерирует MP3-файл через gTTS в текущей папке.
    Возвращает имя созданного файла (путь).
    """
    filename = f"voice_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=text, lang='ru')
    tts.save(filename)
    return filename

# ------------------------------------------------------
# Класс-обработчик для взаимодействия с OpenAI
# ------------------------------------------------------
class ChatGPTHandler:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    async def get_text_response(self, prompt: str, context_msgs: list = None) -> str:
        messages = []
        if context_msgs:
            for msg in context_msgs:
                messages.append({"role": "user", "content": f"Предыдущее сообщение: {msg}"})
        messages.append({"role": "user", "content": "\n Use in your message emojis" + prompt })
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                messages=messages,
                model=Config.OPENAI_MODEL_TEXT,
            )
            logger.info(f"API text output: {response}")
            answer = response.choices[0].message.content
            return answer
        except Exception as e:
            logger.error(f"Ошибка OpenAI API (текст): {e}")
            return "Произошла ошибка при обращении к OpenAI API. Попробуйте позже."

    async def transcribe_audio(self, audio_bytes: bytes) -> str:
        try:
            file_obj = io.BytesIO(audio_bytes)
            transcription = await asyncio.to_thread(
                self.client.audio.transcriptions.create,
                model=Config.WHISPER_MODEL,
                file=file_obj,
            )
            logger.info(f"Транскрипция: {transcription.text}")
            return transcription.text
        except Exception as e:
            logger.error(f"Ошибка транскрипции: {e}")
            return None

    async def process_voice_message(self, audio_bytes: bytes) -> Tuple[str, str]:
        """
        1. Транскрибирует входной голос (audio_bytes -> текст).
        2. Генерирует ответ (текст).
        3. Генерирует MP3-файл ответа (через gTTS).
        Возвращает (текст, путь_к_MP3).
        """
        transcribed = await self.transcribe_audio(audio_bytes)
        if transcribed is None:
            return None, None
        answer_text = await self.get_text_response(transcribed)
        mp3_path = await asyncio.to_thread(generate_voice_answer_gtts, answer_text)
        return answer_text, mp3_path

# ------------------------------------------------------
# Команды /start /help и переключение режима
# ------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    voice_mode_enabled[chat_id] = False
    keyboard = [
        [InlineKeyboardButton("Ответ голосовым", callback_data="enable_voice_mode")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Привет! Я мульти-модальный бот, использующий OpenAI.\n"
        "Отправь мне текст, голосовое сообщение или изображение.\n\n"
        "Нажми кнопку ниже, чтобы включить голосовой режим.",
        reply_markup=reply_markup
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "Я мульти-модальный бот, использующий OpenAI для генерации ответов.\n"
        "Отправь мне текстовое сообщение, голосовое сообщение или изображение, и я постараюсь ответить.\n\n"
        "Доступные команды:\n"
        "/start - начало работы\n"
        "/help - справка\n\n"
        "Также ты можешь включить голосовой режим, нажав кнопку «Ответ голосовым»."
    )
    await update.message.reply_text(help_text)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    chat_id = query.message.chat_id
    data = query.data
    if data == "enable_voice_mode":
        voice_mode_enabled[chat_id] = True
        keyboard = [[InlineKeyboardButton("Ответ текстом", callback_data="disable_voice_mode")]]
        await query.edit_message_text(
            text="Голосовой режим включён. Все последующие ответы будут в виде голосовых сообщений.",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    elif data == "disable_voice_mode":
        voice_mode_enabled[chat_id] = False
        keyboard = [[InlineKeyboardButton("Ответ голосовым", callback_data="enable_voice_mode")]]
        await query.edit_message_text(
            text="Голосовой режим отключён. Ответы будут отправляться текстом.",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    await query.answer()

# ------------------------------------------------------
# Обработчик ТЕКСТА
# ------------------------------------------------------
async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    user_message = update.message.text
    chat_handler: ChatGPTHandler = context.bot_data.get("chat_handler")
    if not chat_handler:
        await update.message.reply_text("Сбой в работе бота. Попробуйте позже.")
        return

    context_msgs = []
    if chat_id in chat_memory:
        context_msgs.append(chat_memory[chat_id])

    response_text = await chat_handler.get_text_response(user_message, context_msgs)
    chat_memory[chat_id] = user_message  # Сохраняем для контекста

    if voice_mode_enabled.get(chat_id, False):
        mp3_filename = generate_voice_answer_gtts(response_text)
        audio_info = MP3(mp3_filename)
        duration = int(audio_info.info.length) if audio_info and audio_info.info else 0

        try:
            with open(mp3_filename, "rb") as f:
                await update.message.reply_voice(
                    voice=InputFile(f),
                    duration=duration
                )
        except Exception as e:
            logger.error(f"Ошибка при отправке голосового ответа: {e}")
            await update.message.reply_text("Не удалось отправить голосовой ответ.")
        finally:
            os.remove(mp3_filename)
    else:
        await update.message.reply_text(response_text)

# ------------------------------------------------------
# Обработчик ГОЛОСОВОГО
# ------------------------------------------------------
async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    chat_handler: ChatGPTHandler = context.bot_data.get("chat_handler")
    if not chat_handler:
        await update.message.reply_text("Сбой в работе бота. Попробуйте позже.")
        return

    try:
        file = await update.message.voice.get_file()
        file_bytes = await file.download_as_bytearray()
    except Exception as e:
        logger.error(f"Ошибка при получении голосового файла: {e}")
        await update.message.reply_text("Не удалось загрузить голосовое сообщение.")
        return

    transcribed_text, mp3_path = await chat_handler.process_voice_message(file_bytes)
    if transcribed_text is None:
        await update.message.reply_text("Ошибка при распознавании речи.")
        return
    if mp3_path is None:
        await update.message.reply_text("Ошибка при генерации голосового ответа.")
        return

    await update.message.reply_text(f"Распознано: {transcribed_text}")

    try:
        audio_info = MP3(mp3_path)
        duration = int(audio_info.info.length) if audio_info and audio_info.info else 0
        with open(mp3_path, "rb") as f:
            await update.message.reply_voice(
                voice=InputFile(f),
                duration=duration
            )
    except Exception as e:
        logger.error(f"Ошибка при отправке голосового ответа: {e}")
        await update.message.reply_text("Не удалось отправить голосовой ответ.")
    finally:
        os.remove(mp3_path)

# ------------------------------------------------------
# Обработчик ИЗОБРАЖЕНИЙ (пример из вашего кода)
# ------------------------------------------------------
async def handle_image_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    chat_handler: ChatGPTHandler = context.bot_data.get("chat_handler")
    if not chat_handler:
        await update.message.reply_text("Сбой в работе бота. Попробуйте позже.")
        return

    try:
        photo = update.message.photo[-1]
        file = await photo.get_file()
        file_bytes = await file.download_as_bytearray()
        base64_image = base64.b64encode(file_bytes).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{base64_image}"
    except Exception as e:
        logger.error(f"Ошибка при получении изображения: {e}")
        await update.message.reply_text("Не удалось обработать изображение.")
        return

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Опиши, что изображено на этой картинке. Ответь на русском языке."},
                    {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}}
                ]
            }
        ]
        response = await asyncio.to_thread(
            chat_handler.client.chat.completions.create,
            messages=messages,
            model=Config.OPENAI_MODEL_TEXT,
            max_tokens=300,
        )
        answer = response.choices[0].message.content
        await update.message.reply_text(answer)
    except Exception as e:
        logger.error(f"Ошибка OpenAI API (изображение): {e}")
        await update.message.reply_text("Произошла ошибка при анализе изображения. Попробуйте позже.")

# ------------------------------------------------------
# Код Flask keep_alive
# ------------------------------------------------------
app = Flask('')

@app.route('/')
def home():
    return "I'm alive"

def run():
    # Используем порт, предоставленный Replit (или 8080 по умолчанию)
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run)
    t.start()

# ------------------------------------------------------
# Основная точка входа
# ------------------------------------------------------
def main():
    # Запускаем Flask-сервер (keep_alive)
    keep_alive()

    application = Application.builder().token(Config.TELEGRAM_TOKEN).build()
    chat_handler = ChatGPTHandler(Config.OPENAI_API_KEY)
    application.bot_data["chat_handler"] = chat_handler

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image_message))

    logger.info("Бот запущен. Начинается опрос сервера Telegram...")
    application.run_polling()

if __name__ == "__main__":
    main()
