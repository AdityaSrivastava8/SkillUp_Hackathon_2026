from typing import Dict, Any, List
from rag.retriever import CaseRetriever


class DetectiveAgent:

    def __init__(self):
        # Create the RAG retriever when the agent is initialized.
        self.retriever = CaseRetriever()

    def reload_cases(self) -> int:
        """Refresh indexed cases after a new JSON file is uploaded."""
        return self.retriever.reload()

    def evaluate_suspect(
        self,
        name: str,
        behavior: str,
        mo_suspected: str,
        personality_notes: str
    ) -> Dict[str, Any]:

        # Build one query while avoiding duplicate text from the frontend.
        query_parts = []
        for value in (behavior, mo_suspected, personality_notes):
            value_text = str(value or "").strip()
            if value_text and value_text not in query_parts:
                query_parts.append(value_text)
        query_str = "; ".join(query_parts)

        # Search the RAG case database for the most similar historical cases.
        retrieved_cases = self.retriever.search_similar_cases(
            query_str,
            top_k=3
        )

        # CaseRetriever returns distance = 1 - cosine similarity.
        # Smaller distance means stronger similarity.
        MATCH_DISTANCE_THRESHOLD = 0.80

        matched_cases: List[Dict[str, Any]] = []
        if retrieved_cases:
            for case in retrieved_cases:
                dist = case.get("distance")
                if dist is not None and float(dist) <= MATCH_DISTANCE_THRESHOLD:
                    matched_cases.append(case)

        # Dynamic Scoring Logic
        # Use actual similarity instead of only counting the number of matches.
        base_score = 15

        if matched_cases:
            similarities = [
                float(case.get("similarity", 0.0))
                for case in matched_cases
            ]

            strongest_similarity = max(similarities, default=0.0)
            average_similarity = (
                sum(similarities) / len(similarities)
                if similarities
                else 0.0
            )

            similarity_score = (
                (strongest_similarity * 60)
                + (average_similarity * 25)
            )

            score = round(
                min(95, max(15, base_score + similarity_score))
            )

            strongest_similarity = max(similarities, default=0.0)
            if strongest_similarity >= 0.55:
                match_quality = "Strong case-index similarity"
            elif strongest_similarity >= 0.35:
                match_quality = "Moderate case-index similarity"
            else:
                match_quality = "Weak case-index similarity"

        else:
            # Fallback heuristic: analyze keyword severity if no vector match is found.
            combined_text = (
                f"{behavior} {mo_suspected} {personality_notes}"
            ).lower()

            severe_keywords = [
                "murder", "kill", "weapon", "assault", "robbery",
                "theft", "break-in", "crime", "stolen", "force", "threat"
            ]

            kw_matches = sum(
                1 for kw in severe_keywords
                if kw in combined_text
            )

            if kw_matches > 0:
                score = min(
                    95,
                    max(
                        25,
                        base_score + (kw_matches * 15)
                    )
                )
            else:
                score = 15
            match_quality = "No reliable case-index match"

        # Convert the numerical score into a risk category.
        if score >= 70:
            risk_level = "HIGH RISK"
        elif score >= 40:
            risk_level = "MEDIUM RISK"
        else:
            risk_level = "LOW RISK"

        return {
            "suspect_name": name,
            "tendency_score": f"{score}%",
            "risk_level": risk_level,
            "match_quality": match_quality,
            "summary": (
                f"Suspect pattern aligns with {len(matched_cases)} "
                f"historical cases in the vector database."
                if matched_cases else
                "No direct vector match in database. Risk evaluated from behavioral indicators."
            ),
            "similar_cases": matched_cases
        }
 
