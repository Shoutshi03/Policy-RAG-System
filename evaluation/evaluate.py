from __future__ import annotations

import json
from pathlib import Path

from policyrag.demo import demo_documents
from policyrag.retrieval import HybridRetriever


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    benchmark = json.loads((root / "evaluation" / "golden_set.json").read_text(encoding="utf-8"))
    documents = demo_documents()
    chunks = [chunk for document in documents for chunk in document.chunks]
    retriever = HybridRetriever()
    answerable = [row for row in benchmark if row["answerable"]]
    hits_at_3 = 0
    reciprocal_ranks: list[float] = []
    abstentions_ok = 0
    details: list[dict[str, object]] = []

    for row in benchmark:
        results = retriever.search(chunks, [row["query"]], top_k=3)
        if row["answerable"]:
            matching = next((index for index, result in enumerate(results, start=1)
                             if result.chunk.file_name.startswith(row["expected_document"])
                             and result.chunk.page == row["expected_page"]), None)
            if matching:
                hits_at_3 += 1
                reciprocal_ranks.append(1 / matching)
            else:
                reciprocal_ranks.append(0.0)
            details.append({"id": row["id"], "hit_at_3": bool(matching), "rank": matching})
        else:
            abstained = not retriever.has_sufficient_evidence(results)
            abstentions_ok += int(abstained)
            details.append({"id": row["id"], "abstention_correct": abstained})

    metrics = {
        "recall_at_3": round(hits_at_3 / max(len(answerable), 1), 3),
        "mrr_at_3": round(sum(reciprocal_ranks) / max(len(answerable), 1), 3),
        "unsupported_abstention_accuracy": round(abstentions_ok / max(len(benchmark) - len(answerable), 1), 3),
        "test_cases": len(benchmark),
        "note": "Deterministic local retrieval only; no LLM calls. Synthetic demonstration corpus.",
        "details": details,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
