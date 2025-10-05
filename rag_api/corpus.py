"""Text corpus and retrieval index for the prototype RAG service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    tags: Sequence[str]


def load_corpus() -> List[Document]:
    """Returns a small hand-authored corpus of product knowledge snippets (12 entries)."""

    snippets = [
        Document(
            doc_id="doc_001",
            title="Getting a library card",
            text="New members bring a photo ID and proof of address to the front desk. The card is printed on the spot and works for both physical and digital checkouts.",
            tags=("membership", "getting-started"),
        ),
        Document(
            doc_id="doc_002",
            title="Renewing borrowed books",
            text="Most books renew twice online from the My Account page. If an item has a hold, returns are due on the original date to avoid late fees.",
            tags=("borrowing", "due-dates"),
        ),
        Document(
            doc_id="doc_003",
            title="Late fee policy",
            text="Books accrue 25 cents per day after the due date with a five dollar maximum. Fees clear automatically once the item is returned and paid online or at the desk.",
            tags=("borrowing", "fees"),
        ),
        Document(
            doc_id="doc_004",
            title="Storytime schedule",
            text="Children's storytime runs every Tuesday and Thursday at 10 a.m. in the community room. Spots are first come, first served with a thirty family capacity.",
            tags=("events", "family"),
        ),
        Document(
            doc_id="doc_005",
            title="Computer access",
            text="Public computers are available during open hours. Sessions last 60 minutes and can be extended at the desk if no one is waiting.",
            tags=("technology", "access"),
        ),
        Document(
            doc_id="doc_006",
            title="Printing and copying",
            text="Black-and-white prints cost ten cents per page and color prints cost fifty cents. Pay with cash or a preloaded print card at the release station.",
            tags=("technology", "services"),
        ),
        Document(
            doc_id="doc_007",
            title="Meeting room reservations",
            text="Groups can reserve the meeting room up to four hours per week. Book online two weeks in advance or call the front desk for same-day availability.",
            tags=("rooms", "reservations"),
        ),
        Document(
            doc_id="doc_008",
            title="E-book help",
            text="Download the LibReader app, sign in with your library card number, and tap Borrow to send titles to your device. Staff can help reset a PIN if the login fails.",
            tags=("ebooks", "support"),
        ),
        Document(
            doc_id="doc_009",
            title="Volunteer program",
            text="Volunteers sort donations, assist at events, and help shelve books. Fill out the online interest form and attend the monthly orientation to get started.",
            tags=("community", "volunteers"),
        ),
        Document(
            doc_id="doc_010",
            title="Book donation guidelines",
            text="The library accepts gently used books published within the last five years. Drop donations at the rear entrance between 9 a.m. and noon on Saturdays.",
            tags=("donations", "policies"),
        ),
        Document(
            doc_id="doc_011",
            title="Library hours",
            text="Monday through Thursday 9 a.m. to 8 p.m., Friday and Saturday 9 a.m. to 5 p.m., closed on Sundays and major holidays.",
            tags=("visiting", "hours"),
        ),
        Document(
            doc_id="doc_012",
            title="Wi-Fi and seating",
            text="Free Wi-Fi is available throughout the building; the password is posted at each table. Quiet study tables sit upstairs, and the cafe seating is near the entrance.",
            tags=("technology", "spaces"),
        ),
    ]
    return snippets


class CorpusIndex:
    """Simple TF-IDF index supporting cosine and raw dot-product ranking."""

    def __init__(self, documents: Iterable[Document]):
        self.documents: List[Document] = list(documents)
        if not self.documents:
            raise ValueError("CorpusIndex requires at least one document")
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._doc_matrix_raw = self.vectorizer.fit_transform(doc.text for doc in self.documents)
        # Precompute a normalized copy for cosine similarity comparisons
        self._doc_matrix_cosine = normalize(self._doc_matrix_raw)

    def search(self, query: str, top_k: int = 3, similarity: str = "cosine") -> List[tuple[Document, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_vec = self.vectorizer.transform([query])
        similarity = similarity.lower()

        if similarity == "cosine":
            query_norm = normalize(query_vec)
            scores = query_norm @ self._doc_matrix_cosine.T
        elif similarity == "dot":
            scores = query_vec @ self._doc_matrix_raw.T
        else:
            raise ValueError(f"Unsupported similarity metric '{similarity}'. Choose 'cosine' or 'dot'.")

        dense_scores = scores.toarray().ravel()
        if dense_scores.size == 0:
            return []
        top_indices = np.argsort(dense_scores)[::-1][:top_k]
        return [(self.documents[i], float(dense_scores[i])) for i in top_indices]


__all__ = ["Document", "load_corpus", "CorpusIndex"]
