"""Indexador de la base de conocimiento (knowledge.md) en ChromaDB.

Estrategia de chunking estructural:
  1. Se extrae el bloque que contiene las secciones (entre líneas '====').
  2. Se divide por secciones ('----' + 'SECCIÓN N: TÍTULO').
  3. Sub-chunking semántico por sección:
       - Sección 3 (servicios)   → un chunk por servicio numerado.
       - Sección 8 (testimonios) → un chunk por testimonio.
       - Sección 9 (FAQ)         → un chunk por par PREGUNTA/RESPUESTA.
       - Resto                   → la sección completa (split recursivo si excede).
  4. Cada chunk lleva un encabezado de contexto (empresa + sección)
     para enriquecer el embedding, y metadatos filtrables.

Sin dependencias pesadas: incluye un splitter recursivo propio,
equivalente ligero a RecursiveCharacterTextSplitter.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from core.config import KNOWLEDGE_PATH
from core.rag import get_vectorstore

logger = logging.getLogger(__name__)

EMPRESA = "Manzzo y Cía"
MAX_CHARS_CHUNK = 1800  # umbral para sub-dividir con el splitter recursivo
CHUNK_OVERLAP = 150     # solape entre chunks sub-divididos

SECTION_HEADER_RE = re.compile(r"SECCIÓN\s+(\d+)\s*[:\-]\s*(.+)", re.IGNORECASE)
EQUALS_LINE_RE = re.compile(r"(?m)^\s*={10,}\s*$")          # líneas de '===='
FAQ_RE = re.compile(r"(?=PREGUNTA\s+\d+\s*:)", re.IGNORECASE)
SERVICE_RE = re.compile(r"(?m)(?=^\d+\.\s+[A-ZÁÉÍÓÚÑÜ][^:\n]{3,}:)")
TESTIMONY_RE = re.compile(r"(?=Testimonio\s+\d+)", re.IGNORECASE)

SEPARATORS = ["\n\n", "\n", ". ", " "]


# ---------------------------------------------------------------------------
# Splitter propio (sin dependencias pesadas)
# Equivalente ligero a RecursiveCharacterTextSplitter: divide por
# separadores en orden de prioridad y aplica solape entre chunks.
# ---------------------------------------------------------------------------
def _recursive_split(text: str, max_chars: int, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []

    sep = next((s for s in SEPARATORS if s in text), None)
    if sep is None:  # sin separadores → corte deslizante
        step = max_chars - overlap
        return [text[i:i + max_chars] for i in range(0, len(text), step)]

    chunks, current = [], ""
    for piece in text.split(sep):
        candidate = piece if not current else current + sep + piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(piece) > max_chars:
                chunks.extend(_recursive_split(piece, max_chars, overlap))
                current = ""
            else:
                current = piece
    if current:
        chunks.append(current)

    # Solape: antepone la cola del chunk anterior al siguiente
    if overlap and len(chunks) > 1:
        merged = [chunks[0]]
        for prev, cur in zip(chunks, chunks[1:]):
            merged.append(prev[-overlap:] + cur)
        chunks = merged
    return chunks


def _maybe_split(text: str) -> List[str]:
    """Sub-divide solo si el chunk excede el umbral."""
    if len(text) <= MAX_CHARS_CHUNK:
        return [text]
    return [c for c in _recursive_split(text, MAX_CHARS_CHUNK) if c.strip()]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
@dataclass
class Section:
    numero: int
    titulo: str
    cuerpo: str


def _clean_document(text: str) -> str:
    """Devuelve solo el bloque que contiene las secciones.

    El documento está envuelto en líneas '====' (cabecera y pie). En vez de
    eliminar bloques por posición, separamos por esas líneas y nos quedamos
    con el bloque que contiene encabezados 'SECCIÓN N:'. Así el parser es
    inmune a cambios en la estructura de la cabecera.
    """
    text = text.replace("\r\n", "\n")
    for block in EQUALS_LINE_RE.split(text):
        if SECTION_HEADER_RE.search(block):
            return block.strip()
    logger.warning("No se hallaron encabezados de sección; se usará el documento crudo.")
    return text.strip()


def _parse_sections(text: str) -> List[Section]:
    """Devuelve la lista de secciones (número, título, cuerpo)."""
    parts = [p.strip() for p in re.split(r"-{10,}", _clean_document(text)) if p.strip()]
    sections: List[Section] = []
    i = 0
    while i < len(parts):
        m = SECTION_HEADER_RE.match(parts[i])
        if m and i + 1 < len(parts):
            sections.append(Section(int(m.group(1)), m.group(2).strip(), parts[i + 1]))
            i += 2
        else:  # contenido huérfano → se anexa a la sección anterior
            if sections:
                sections[-1].cuerpo += "\n\n" + parts[i]
            i += 1
    return sections


# ---------------------------------------------------------------------------
# Construcción de Documents
# ---------------------------------------------------------------------------
def parse_knowledge_markdown(path: Optional[Path] = None) -> List[Document]:
    path = path or KNOWLEDGE_PATH
    raw = path.read_text(encoding="utf-8-sig")  # utf-8-sig: tolerante a BOM (Windows)
    sections = _parse_sections(raw)
    logger.info("Secciones detectadas: %d de %s", len(sections), path.name)

    docs: List[Document] = []

    for sec in sections:
        # Sub-chunking semántico según la sección
        if sec.numero == 9:
            pieces, tipo = FAQ_RE.split(sec.cuerpo), "faq"
        elif sec.numero == 3:
            pieces, tipo = SERVICE_RE.split(sec.cuerpo), "servicio"
        elif sec.numero == 8:
            pieces, tipo = TESTIMONY_RE.split(sec.cuerpo), "testimonio"
        else:
            pieces, tipo = [sec.cuerpo], "seccion"

        for piece in pieces:
            for sub in _maybe_split(piece.strip()):
                if len(sub) < 40:        # ignora fragmentos residuales
                    continue
                docs.append(Document(
                    page_content=f"[{EMPRESA} | Sección {sec.numero}: {sec.titulo}]\n{sub}",
                    metadata={
                        "seccion": sec.numero,
                        "seccion_titulo": sec.titulo,
                        "tipo": tipo,
                        "fuente": path.name,
                        "empresa": EMPRESA,
                    },
                ))
    return docs


# ---------------------------------------------------------------------------
# Indexación en ChromaDB
# ---------------------------------------------------------------------------
def index_knowledge(path: Optional[Path] = None, reset: bool = True) -> int:
    path = path or KNOWLEDGE_PATH
    docs = parse_knowledge_markdown(path)
    if not docs:
        raise ValueError(f"No se generaron chunks desde {path}; revisa el formato.")

    vs = get_vectorstore()
    if reset:
        try:
            vs.delete_collection()
            logger.info("Colección anterior eliminada (reset).")
        except Exception:
            pass
        vs = get_vectorstore()  # re-crear instancia tras el borrado

    ids = [f"s{d.metadata['seccion']:02d}-{d.metadata['tipo']}-{i:03d}"
           for i, d in enumerate(docs, start=1)]
    vs.add_documents(docs, ids=ids)
    logger.info("Indexados %d chunks desde %s", len(docs), path)
    return len(docs)
