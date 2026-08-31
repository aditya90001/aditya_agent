import json
import time
from pathlib import Path

from rag import retrieve_bis_knowledge


def load_dataset(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def simple_recall_at_k(query: str, expected_keywords, k=5):
    start = time.time()
    result = retrieve_bis_knowledge(query, k=k)
    latency = (time.time() - start) * 1000.0

    text = result.lower()

    hit = any(kw.lower() in text for kw in expected_keywords)

    return hit, latency, result


def run():
    ds = load_dataset(Path(__file__).parent / "dataset.json")

    stats = {
        "queries": len(ds),
        "hits": 0,
        "latencies_ms": [],
        "no_results": 0,
    }

    for sample in ds:
        hit, latency, _ = simple_recall_at_k(sample["question"], sample["expected_keywords"], k=5)
        if hit:
            stats["hits"] += 1
        else:
            stats["no_results"] += 1
        stats["latencies_ms"].append(latency)

    avg_latency = sum(stats["latencies_ms"]) / len(stats["latencies_ms"]) if stats["latencies_ms"] else 0.0

    print("Evaluation Report")
    print("Queries:", stats["queries"])
    print("Hits@5:", stats["hits"])
    print("No result count:", stats["no_results"])
    print("Avg latency ms:", round(avg_latency, 2))


if __name__ == "__main__":
    run()
