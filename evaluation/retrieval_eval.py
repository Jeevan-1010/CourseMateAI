import argparse
import json
import statistics
import urllib.error
import urllib.request
from pathlib import Path


def request_answer(base_url: str, filename: str, question: str, timeout: int):
    payload = json.dumps({
        "filename": filename,
        "question": question,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API returned HTTP {error.code}: {detail}") from error


def unique_pages(retrieved_context):
    pages = []
    seen = set()
    for item in retrieved_context:
        page = item.get("page")
        if page is not None and page not in seen:
            seen.add(page)
            pages.append(page)
    return pages


def score_retrieval(retrieved_pages, relevant_pages, k: int):
    relevant = set(relevant_pages)
    top_k = retrieved_pages[:k]
    hits = [page for page in top_k if page in relevant]

    precision = len(hits) / k
    recall = len(hits) / len(relevant) if relevant else 0.0

    reciprocal_rank = 0.0
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant:
            reciprocal_rank = 1.0 / rank
            break

    return {
        f"precision_at_{k}": precision,
        f"recall_at_{k}": recall,
        "reciprocal_rank": reciprocal_rank,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CourseMateAI retrieval using labeled relevant pages."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--dataset",
        default="evaluation/retrieval_questions.json",
    )
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not cases:
        raise SystemExit("The evaluation dataset is empty.")

    results = []
    for index, case in enumerate(cases, start=1):
        question = case["question"]
        filename = case["filename"]
        relevant_pages = sorted(set(case["relevant_pages"]))
        if not relevant_pages:
            raise SystemExit(
                f"Test case {index} has no relevant_pages labels."
            )

        response = request_answer(
            args.base_url,
            filename,
            question,
            args.timeout,
        )
        retrieved = unique_pages(response.get("retrieved_context", []))
        metrics = score_retrieval(retrieved, relevant_pages, args.k)

        results.append({
            "question": question,
            "relevant_pages": relevant_pages,
            "retrieved_pages": retrieved,
            **metrics,
            "retrieval_latency_ms": response.get("retrieval_latency_ms"),
            "reranking_latency_ms": response.get("reranking_latency_ms"),
            "end_to_end_latency_ms": response.get("end_to_end_latency_ms"),
            "deduplicated_chunks_count": response.get(
                "deduplicated_chunks_count"
            ),
        })

        print(
            f"[{index}/{len(cases)}] P@{args.k}="
            f"{metrics[f'precision_at_{args.k}']:.3f} "
            f"R@{args.k}={metrics[f'recall_at_{args.k}']:.3f} "
            f"RR={metrics['reciprocal_rank']:.3f} "
            f"rerank={response.get('reranking_latency_ms')} ms"
        )

    summary = {
        f"precision_at_{args.k}": statistics.mean(
            item[f"precision_at_{args.k}"] for item in results
        ),
        f"recall_at_{args.k}": statistics.mean(
            item[f"recall_at_{args.k}"] for item in results
        ),
        "mrr": statistics.mean(
            item["reciprocal_rank"] for item in results
        ),
    }

    for key in [
        "retrieval_latency_ms",
        "reranking_latency_ms",
        "end_to_end_latency_ms",
        "deduplicated_chunks_count",
    ]:
        values = [
            item[key] for item in results
            if isinstance(item.get(key), (int, float))
        ]
        if values:
            summary[f"average_{key}"] = statistics.mean(values)

    report = {
        "dataset": str(dataset_path),
        "question_count": len(results),
        "k": args.k,
        "summary": summary,
        "results": results,
    }

    rendered = json.dumps(report, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(f"Saved evaluation report: {output_path}")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
