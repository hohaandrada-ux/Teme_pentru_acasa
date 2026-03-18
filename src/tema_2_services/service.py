import json
import os
import hashlib

from dotenv import load_dotenv
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import faiss

load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "data_chunks.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
FAISS_META_PATH = os.path.join(DATA_DIR, "faiss.index.meta")
USE_MODEL_URL = os.environ.get(
    "USE_MODEL_URL",
    "https://tfhub.dev/google/universal-sentence-encoder/4",
)

WEB_URLS = [u for u in os.environ.get("WEB_URLS", "").split(";") if u]

class RAGAssistant:
    """Asistent cu RAG din surse web si un LLM pentru raspunsuri."""

    def __init__(self) -> None:
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Seteaza GROQ_API_KEY in variabilele de mediu.")

        self.client = OpenAI(
            api_key=self.groq_api_key,
            base_url=os.environ.get("GROQ_BASE_URL")
        )

        os.makedirs(DATA_DIR, exist_ok=True)
        self.embedder = None

        # RELEVANCE (important pentru filtrare)
        self.relevance = self._embed_texts(
            "Aceasta este o intrebare relevanta despre un brand de lenjerie intima pentru femei, produse, clienti si marketing."
        )[0]

        # SYSTEM PROMPT
        self.system_prompt = (
            "Esti un asistent AI pentru un brand de lenjerie intima pentru femei. "
            "Raspunzi doar la intrebari despre produse, clienti, marketing si brand. "
            "Ofera raspunsuri clare, concise si relevante. "
            "Daca intrebarea nu are legatura cu domeniul, refuza politicos."
        )

    def _load_documents_from_web(self) -> list[str]:
        if os.path.exists(CHUNKS_JSON_PATH):
            try:
                with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if isinstance(cached, list) and cached:
                    return cached
            except (OSError, json.JSONDecodeError):
                pass

        all_chunks = []
        for url in WEB_URLS:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                for doc in docs:
                    chunks = self._chunk_text(doc.page_content)
                    all_chunks.extend(chunks)
            except Exception:
                continue

        if all_chunks:
            with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False)

        return all_chunks

    def _send_prompt_to_llm(self, user_input: str, context: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"""
Intrebare utilizator: {user_input}

Context relevant:
{context}

Raspunde clar si concis doar pe baza contextului.
"""
            },
        ]

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model="openai/gpt-oss-20b",
            )
            return response.choices[0].message.content
        except Exception:
            return "Asistent: Nu pot ajunge la model acum. Incearca din nou."

    def _embed_texts(self, texts: str | list[str], batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if self.embedder is None:
            self.embedder = hub.load(USE_MODEL_URL)

        embeddings = self.embedder(texts)
        return np.asarray(embeddings, dtype="float32")

    def _chunk_text(self, text: str) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
        )
        return splitter.split_text(text or "")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _build_faiss_index_from_chunks(self, chunks: list[str]) -> faiss.IndexFlatIP:
        embeddings = self._embed_texts(chunks).astype("float32")
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, FAISS_INDEX_PATH)

        return index

    def _retrieve_relevant_chunks(self, chunks: list[str], user_query: str, k: int = 5) -> list[str]:
        if not chunks:
            return []

        query_embedding = self._embed_texts(user_query).astype("float32")
        index = self._build_faiss_index_from_chunks(chunks)

        faiss.normalize_L2(query_embedding)

        _, indices = index.search(query_embedding, k=min(k, len(chunks)))
        return [chunks[i] for i in indices[0]]

    def calculate_similarity(self, text: str) -> float:
        embedding = self._embed_texts(text.strip())[0]
        return self._cosine_similarity(embedding, self.relevance)

    def is_relevant(self, user_input: str) -> bool:
        return self.calculate_similarity(user_input) >= 0.5

    def assistant_response(self, user_message: str) -> str:
        if not user_message:
            return "Te rog pune o intrebare despre brandul nostru de lenjerie intima."

        if not self.is_relevant(user_message):
            return "Imi pare rau, pot raspunde doar la intrebari legate de brandul nostru de lenjerie intima."

        chunks = self._load_documents_from_web()
        relevant_chunks = self._retrieve_relevant_chunks(chunks, user_message)
        context = "\n\n".join(relevant_chunks)

        return self._send_prompt_to_llm(user_message, context)


if __name__ == "__main__":
    assistant = RAGAssistant()

    print(assistant.assistant_response("Ce produse oferiti?"))
    print(assistant.assistant_response("Care este capitala Frantei?"))
