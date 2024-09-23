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
from langchain_community.vectorstores import Chroma
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def chunk_data(data, chunk_size):
    for i in range(5001, len(data), chunk_size):
        yield data[i:i + chunk_size]

def load_documents(file_path):
    loader = Docx2txtLoader(file_path)
    return loader.load()
def split_text_into_blocks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=512, add_start_index=True
    )
    return text_splitter.split_documents(docs)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
persist_directory = "./tgbot_data/chroma"
db_file = os.path.join(persist_directory, "chroma.sqlite3")
print("step1")
docs = load_documents("./tgbot_data/db_veda.docx")
print("step2")
all_splits = split_text_into_blocks(docs)
print("step3")
if os.path.exists(db_file):
    os.remove(db_file)
    os.sync()


vectorstore = Chroma.from_documents(
    embedding=embedding,
    persist_directory=persist_directory,
    documents=all_splits[0:5000]
)
print("step4")
for batch in chunk_data(all_splits, 5000):
    print("step N")
    vectorstore.add_documents(batch)

print("finish")
