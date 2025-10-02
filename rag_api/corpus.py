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
            title="Password reset workflow",
            text="Users can reset their workspace password via Settings > Security > Reset Password. A confirmation email is sent and the reset link stays valid for 30 minutes.",
            tags=("account", "security"),
        ),
        Document(
            doc_id="doc_002",
            title="Mobile offline access policy",
            text="Offline mode caches the last 48 hours of messages. If a device is lost, revoke the session from the Admin Console Security tab to invalidate cached data.",
            tags=("mobile", "security"),
        ),
        Document(
            doc_id="doc_003",
            title="Audit log export",
            text="Admins can download audit logs as CSV by navigating to Admin Console > Compliance > Audit Logs. Exports include actor, timestamp, and ip columns.",
            tags=("compliance", "reporting"),
        ),
        Document(
            doc_id="doc_004",
            title="Burst rate limits",
            text="Webhook integrations may hit hidden burst caps. Implement exponential backoff and retry with jitter to avoid cascading failures.",
            tags=("integrations", "rate-limits"),
        ),
        Document(
            doc_id="doc_005",
            title="External workspace invites",
            text="To invite emails outside your domain, add the partner domain to the External Access allowlist and resend pending invitations.",
            tags=("access", "collaboration"),
        ),
        Document(
            doc_id="doc_006",
            title="Budget guardrail",
            text="Daily token budgets are enforced per workspace. After the soft cap we throttle completions and send an alert to billing contacts; spend does not auto-increase.",
            tags=("billing", "usage"),
        ),
        Document(
            doc_id="doc_007",
            title="Data retention policy",
            text="Deleted resources remain recoverable for 30 days before being purged from primary and backup storage.",
            tags=("compliance", "retention"),
        ),
        Document(
            doc_id="doc_008",
            title="Dark mode does not affect billing",
            text="UI preferences such as dark mode or compact view do not impact token consumption or billing calculations.",
            tags=("ui", "billing"),
        ),
        Document(
            doc_id="doc_009",
            title="Manual upgrade process",
            text="Spend above the committed plan requires manual approval. Contact your account manager to issue a contract amendment before extra usage is billed.",
            tags=("billing", "contracts"),
        ),
        Document(
            doc_id="doc_010",
            title="Workspace activity alerts",
            text="Security alerts trigger when a device is revoked, when throttling occurs, and when admin roles change. Alerts post to the #security-updates channel by default.",
            tags=("security", "notifications"),
        ),
        Document(
            doc_id="doc_011",
            title="CSV export schema",
            text="The audit CSV includes columns: actor_email, action, resource, ip, location, metadata. Large exports are streamed in 10k row chunks.",
            tags=("compliance", "reporting"),
        ),
        Document(
            doc_id="doc_012",
            title="Workspace roles overview",
            text="Workspace admins can manage security policies, billing owners manage spend approvals, and project maintainers configure integrations.",
            tags=("roles", "governance"),
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