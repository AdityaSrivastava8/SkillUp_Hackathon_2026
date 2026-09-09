"""
rag/retriever.py

Pure-Python TF/cosine similarity retriever — zero external dependencies.
"""

import os
import json
import math
import re
from typing import List, Dict, Any, Optional


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _tf(tokens: List[str]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    total = len(tokens) or 1
    return {token: count / total for token, count in counts.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    mag_a = math.sqrt(sum(value * value for value in a.values()))
    mag_b = math.sqrt(sum(value * value for value in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class CaseRetriever:

    def __init__(self, cases_dir: Optional[str] = None, db_path: str = "./chroma_db"):
        if cases_dir is None:
            self.cases_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "cases")
            )
        elif os.path.isabs(cases_dir):
            self.cases_dir = cases_dir
        else:
            self.cases_dir = os.path.abspath(cases_dir)
        self.db_path = db_path
        self._docs: List[Dict[str, Any]] = []
        self._index_cases()

    def reload(self) -> int:
        """Rebuild the in-memory index after case files change."""
        self._docs = []
        self._index_cases()
        return len(self._docs)

    @staticmethod
    def _case_text(case: Dict[str, Any]) -> str:
        """Flatten common case fields so uploaded evidence is searchable."""
        def flatten(value: Any) -> str:
            if isinstance(value, list):
                return ", ".join(flatten(item) for item in value)
            if isinstance(value, dict):
                return ", ".join(
                    f"{key}: {flatten(item)}"
                    for key, item in value.items()
                )
            return str(value or "")

        fields = (
            "title", "summary", "crime_type", "location", "status",
            "modus_operandi", "personality_disorder", "common_traits",
            "evidence", "suspects", "case_notes", "investigation_notes",
        )
        return " ".join(
            f"{field.replace('_', ' ').title()}: {flatten(case.get(field, ''))}."
            for field in fields
            if case.get(field)
        )

    def _index_cases(self) -> None:
        if not os.path.isdir(self.cases_dir):
            return

        for file in os.listdir(self.cases_dir):
            if not file.lower().endswith(".json"):
                continue

            file_path = os.path.join(self.cases_dir, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if not content:
                    continue

                case = json.loads(content)
                if not isinstance(case, dict):
                    continue

                doc_text = self._case_text(case)

                tokens = _tokenize(doc_text)
                if not tokens:
                    continue

                case_id = str(case.get("case_id", file))
                title = str(case.get("title", "Unknown"))
                location = str(case.get("location", "Unknown"))
                crime_type = str(case.get("crime_type", "Unknown"))

                self._docs.append({
                    "case_id": case_id,
                    "text": doc_text,
                    "tf": _tf(tokens),
                    "metadata": {
                        "case_id": case_id,
                        "title": title,
                        "case_title": title,
                        "location": location,
                        "crime_type": crime_type,
                    },
                })

            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                continue

    def search_similar_cases(
        self,
        suspect_query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        # Every call creates a fresh query vector from the CURRENT suspect.
        if not suspect_query or not str(suspect_query).strip():
            return []
        if not self._docs:
            return []

        try:
            top_k = max(1, int(top_k))
        except (TypeError, ValueError):
            top_k = 3

        query_tokens = _tokenize(str(suspect_query))
        if not query_tokens:
            return []

        query_tf = _tf(query_tokens)
        scored = []

        for doc in self._docs:
            similarity = _cosine(query_tf, doc["tf"])
            if similarity <= 0:
                continue
            distance = max(0.0, 1.0 - similarity)
            scored.append((similarity, distance, doc))

        # Highest similarity = best match.
        scored.sort(key=lambda item: item[0], reverse=True)

        results: List[Dict[str, Any]] = []
        for similarity, distance, doc in scored[:top_k]:
            metadata = dict(doc["metadata"])
            results.append({
                "case_id": doc["case_id"],
                "metadata": metadata,
                "case_title": metadata.get("case_title", metadata.get("title", "Unknown Case")),
                "location": metadata.get("location", "Unknown"),
                "crime_type": metadata.get("crime_type", "Unknown"),
                "snippet": doc["text"][:300],
                "distance": distance,
                "similarity": similarity,
            })

        return results
 
