"""RAG sobre ChromaDB (persistido en carpeta chroma_db/)."""
import os
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from core.config import CHROMA_COLLECTION, CHROMA_DIR, OPENAI_API_KEY


def _get_embeddings() -> OpenAIEmbeddings:
    """Instancia lazy: solo se crea al usar el vectorstore."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Falta la API key de OpenAI. Define DEMO_OPENAI_API_KEY u "
            "OPENAI_API_KEY en el entorno o en el archivo .env"
        )
    os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=_get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def retrieve(query: str, k: int = 3, tipo: Optional[str] = None) -> List[Document]:
    """Busca los chunks más relevantes; 'tipo' permite filtrar (faq, servicio...)."""
    where = {"tipo": tipo} if tipo else None
    return get_vectorstore().similarity_search(query, k=k, filter=where)
