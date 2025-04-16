import asyncio
import re
from time import time

import httpx
import yaml
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utilities.colorize import color


class TextRequest(BaseModel):
    text: str
    id: int
    sub_id: int | None = None


class MessageWithDocs:
    def __init__(self, message, documents):
        self.message = message
        self.documents = documents

    def embed_self(self, vectorizer):
        if not vectorizer:
            return []
        full_text = self.message.content
        if self.documents:
            full_text += "\n" + docs_to_string(self.documents)
        return vectorizer.transform([full_text]).toarray()[0]


class ChatHistory:
    def __init__(self, limit: int = 0):
        self.messages: list[MessageWithDocs] = []
        self.vectorizer = None
        self.limit = limit

    def limit_history(self):
        if self.limit != 0:
            self.messages = self.messages[
                -self.limit :
            ]  # lascio solo gli ultimi messaggi

    def add_message_from_user(
        self, user_input: dict
    ):  # * ste funzioni vanno chiamate dopo che il modello ha finito di rispondere
        content = user_input.get("messages", "")
        if content:
            content = content[1]
            message = MessageWithDocs(
                message=HumanMessage(content=content), documents=[]
            )
            self.messages.append(message)
            self.limit_history()

    def add_message_from_response(self, response: dict):
        message = MessageWithDocs(
            message=AIMessage(content=response.get("answer", "")),
            documents=response.get("documents", []),
        )
        self.messages.append(message)
        self.limit_history()

    def train_vectorizer(self):
        all_texts = []
        ai_messagges = [
            msg for msg in self.messages if isinstance(msg.message, AIMessage)
        ]
        for msg in ai_messagges:
            all_texts.append(msg.message.content + "\n" + docs_to_string(msg.documents))
        if all_texts:
            self.vectorizer = TfidfVectorizer(encoding="utf-8").fit(all_texts)

    def get_old_messages_ctx(self, threshold: float):
        if not self.vectorizer:
            return []
        user_message_vector = self.messages[-1].embed_self(self.vectorizer)
        ctx = []
        ai_messagges = [
            msg for msg in self.messages if isinstance(msg.message, AIMessage)
        ]
        for msg in ai_messagges:
            vector = msg.embed_self(self.vectorizer)
            try:
                similarity = cosine_similarity([user_message_vector], [vector])[0][0]
            except Exception as e:
                raise e
            if similarity > threshold:
                ctx.extend(msg.documents)
        if not ctx:  # Se non ho trovato nessun contesto, prendo l'ultimo contesto
            ctx.extend(ai_messagges[-1].documents)
        return ctx

    def get_followup_ctx(self, threshold: float):
        self.train_vectorizer()
        if self.vectorizer is None:
            return []
        return self.get_old_messages_ctx(threshold)

    def get_all_messages(self):
        if self.messages == []:
            return []
        return [msg.message for msg in self.messages]

    def get_last_messages(self, n: int):
        if self.messages == []:
            return []
        if n > len(self.messages):
            n = len(self.messages)
        return [msg.message for msg in self.messages[-n:]]

    def clear(self):
        self.messages = []
        self.vectorizer = None


class StdOutHandler:
    """
    Class to manage token's stream
    """

    def __init__(self, config, audio=True, debug=False):
        self.containers = None
        self.text = ""
        self.chunks = []
        self.completed_chunks = []
        self.time = 0
        self.config = config
        self.debug = debug
        self.audio = audio

    def set_audio(self, audio: bool):
        self.audio = audio

    def start(self, containers=None):
        self.time = time()
        self.text = ""
        self.containers = containers
        self.chunks = []
        self.completed_chunks = []

    async def on_new_token(self, token: dict) -> None:
        token = token.content if isinstance(token, AIMessageChunk) else ""
        if self.debug:
            if token:
                print(token, sep="", end="", flush=True)
        if token:
            self.text += token
            if self.audio:
                try:
                    await self.generate_audio_stream()
                except Exception as e:
                    print(color("[STDOUTHANDLER]", True, "red"), f"Error: {e}")
                    self.error(e)
            if self.containers:
                self.containers[0].markdown(self.text)

    def sanitize_text(self, text: str) -> str:
        return text.translate({ord(i): None for i in "*!\n\t"})

    def chunk_text(self, text: str) -> list[str]:
        stripped_text = self.sanitize_text(text)
        return [
            chunk.strip() for chunk in re.split(r"[.]", stripped_text) if chunk.strip()
        ]

    async def generate_audio_stream(self):
        if self.text:
            self.chunks = self.chunk_text(self.text)

        if len(self.chunks) > 1:
            async with httpx.AsyncClient() as client:
                for i in range(len(self.chunks) - 1):
                    if self.chunks[i] and i not in self.completed_chunks:
                        self.completed_chunks.append(i)
                        response = await client.post(
                            "http://localhost:8000/store_text",
                            json=TextRequest(text=self.chunks[i], id=i).model_dump(),
                        )
                        if response.json().get("status", "error") == "error":
                            self.error(Exception("Errore nell'invio del chunk"))

    async def end(self):
        if self.audio and self.text:
            self.chunks = self.chunk_text(self.text)
            if self.chunks:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "http://localhost:8000/store_text",
                        json=TextRequest(
                            text=self.chunks[-1], id=len(self.chunks) - 1
                        ).model_dump(),
                    )
                    if response.json().get("status", "error") == "error":
                        self.error(Exception("Errore nell'invio del chunk"))

                    # Controllo finale per il completamento
                    while True:
                        final_response = await client.get("http://localhost:8000/")
                        status = final_response.json().get("status", "error")
                        if status == "ok":
                            print("Risposta finale ricevuta")
                            self.time = time() - self.time
                            text_time = f"⏱ Tempo di risposta: {self.time:.2f} secondi"
                            if self.debug:
                                print("\n" + text_time)
                            if self.containers:
                                self.containers[1].markdown(text_time)
                            break
                        elif status == "error":
                            print("Errore nella risposta finale")
                            self.error(Exception("Errore nella risposta finale"))
                        else:
                            print("Risposta finale in elaborazione")
                            await asyncio.sleep(1)
        self.text = ""
        self.chunks = []
        self.completed_chunks = []

    def error(self, error: Exception):
        self.text = ""
        self.chunks = []
        self.completed_chunks = []
        self.time = time() - self.time
        text_time = f"⏱ Tempo di risposta: {self.time:.2f} secondi"
        if self.debug:
            print(color("[STDOUTHANDLER]", True, "red"), f"Error: {error}")
            print("\n" + text_time)
        if self.containers:
            self.containers[0].markdown(f"Errore: {error}")
            self.containers[1].markdown(text_time)
        raise error


def load_config(file_path="config.yaml"):
    """
    Load configuration file

    Args:
        file_path (str): Configuration file path

    Returns:
        dict: Configuration file
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def docs_to_string(docs, sep="\n\n"):
    if docs:
        return f"{sep}".join([d.page_content for d in docs])
    return ""
