import argparse
import json
import math
import statistics
import urllib.error
import urllib.request
from pathlib import Path


def request_answer(
    base_url: str,
    filename: str,
    question: str,
    timeout: int,
    include_retrieval_debug: bool = False,
):
    payload = json.dumps({
        "filename": filename,
        "question": question,
        "include_retrieval_debug": include_retrieval_debug,
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

    # Unanswerable questions have no relevant pages and are excluded from
    # retrieval-ranking averages rather than being treated as retrieval misses.
    if not relevant:
        return {
            f"precision_at_{k}": None,
            f"recall_at_{k}": None,
            "reciprocal_rank": None,
        }

    precision = len(hits) / k
    recall = len(hits) / len(relevant)

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


def rank_candidate_pages(candidates, candidate_k=None):
    selected = candidates if candidate_k is None else candidates[:candidate_k]
    ranked = sorted(
        selected,
        key=lambda item: float(item["rerank_score"]),
        reverse=True,
    )
    return unique_pages(ranked)


def retrieval_study_metrics(retrieved_pages, relevant_pages, k):
    metrics = score_retrieval(retrieved_pages, relevant_pages, k)
    relevant = set(relevant_pages)
    top_k = retrieved_pages[:k]
    hit_at_1 = int(bool(retrieved_pages and retrieved_pages[0] in relevant))

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, page in enumerate(top_k, start=1)
        if page in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(
        1.0 / __import__("math").log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return {
        **metrics,
        "hit_at_1": hit_at_1 if relevant else None,
        f"ndcg_at_{k}": dcg / idcg if idcg else None,
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
    parser.add_argument(
        "--candidate-study",
        action="store_true",
        help="Cache pre-reranking candidates and compare retrieval/reranking K values offline.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    # Support both the simple list format and the project's existing
    # {"document": "...", "tests": [...]} evaluation dataset.
    if isinstance(dataset, dict):
        document = dataset.get("document")
        tests = dataset.get("tests")
        if not document or not isinstance(tests, list):
            raise SystemExit(
                "Dataset object must contain a document name and a tests list."
            )
        cases = [
            {
                **test,
                "filename": test.get("filename", document),
            }
            for test in tests
        ]
    elif isinstance(dataset, list):
        cases = dataset
    else:
        raise SystemExit("Dataset must be a list or an object containing document and tests.")

    if not cases:
        raise SystemExit("The evaluation dataset is empty.")

    results = []
    for index, case in enumerate(cases, start=1):
        question = case["question"]
        filename = case["filename"]
        relevant_pages = sorted(set(case.get("relevant_pages", [])))
        response = request_answer(
            args.base_url,
            filename,
            question,
            args.timeout,
            include_retrieval_debug=args.candidate_study,
        )
        retrieved = unique_pages(response.get("retrieved_context", []))
        metrics = score_retrieval(retrieved, relevant_pages, args.k)

        result_item = {
            "id": case.get("id"),
            "question": question,
            "expected_answerable": case.get("expected_answerable", bool(relevant_pages)),
            "relevant_pages": relevant_pages,
            "retrieved_pages": retrieved,
            **metrics,
            "retrieval_latency_ms": response.get("retrieval_latency_ms"),
            "reranking_latency_ms": response.get("reranking_latency_ms"),
            "generation_latency_ms": response.get("generation_latency_ms"),
            "end_to_end_latency_ms": response.get("end_to_end_latency_ms"),
            "deduplicated_chunks_count": response.get(
                "deduplicated_chunks_count"
            ),
            "candidate_chunks_count": response.get("candidate_chunks_count"),
            "unique_candidate_chunks_count": response.get(
                "unique_candidate_chunks_count"
            ),
            "reranker_input_chunks_count": response.get(
                "reranker_input_chunks_count"
            ),
        }

        if args.candidate_study:
            candidates = response.get("retrieval_candidates")
            if not isinstance(candidates, list):
                raise RuntimeError(
                    "API did not return retrieval_candidates. "
                    "Pull the latest backend code and restart FastAPI."
                )
            result_item["retrieval_candidates"] = candidates
            result_item["no_rerank_retrieved_pages"] = unique_pages(candidates)

            if relevant_pages:
                result_item["no_rerank_metrics"] = retrieval_study_metrics(
                    result_item["no_rerank_retrieved_pages"],
                    relevant_pages,
                    args.k,
                )

        results.append(result_item)

        if relevant_pages:
            metric_text = (
                f"P@{args.k}={metrics[f'precision_at_{args.k}']:.3f} "
                f"R@{args.k}={metrics[f'recall_at_{args.k}']:.3f} "
                f"RR={metrics['reciprocal_rank']:.3f}"
            )
        else:
            metric_text = "Unanswerable (excluded from retrieval averages)"

        print(
            f"[{index}/{len(cases)}] {metric_text} "
            f"rerank={response.get('reranking_latency_ms')} ms"
        )

    answerable_results = [
        item for item in results if item[f"precision_at_{args.k}"] is not None
    ]
    if not answerable_results:
        raise SystemExit("No answerable questions with relevant_pages labels were found.")

    summary = {
        f"precision_at_{args.k}": statistics.mean(
            item[f"precision_at_{args.k}"] for item in answerable_results
        ),
        f"recall_at_{args.k}": statistics.mean(
            item[f"recall_at_{args.k}"] for item in answerable_results
        ),
        "mrr": statistics.mean(
            item["reciprocal_rank"] for item in answerable_results
        ),
        "answerable_question_count": len(answerable_results),
        "unanswerable_question_count": len(results) - len(answerable_results),
    }

    for key in [
        "retrieval_latency_ms",
        "reranking_latency_ms",
        "generation_latency_ms",
        "end_to_end_latency_ms",
        "deduplicated_chunks_count",
        "candidate_chunks_count",
        "unique_candidate_chunks_count",
        "reranker_input_chunks_count",
    ]:
        values = [
            item[key] for item in results
            if isinstance(item.get(key), (int, float))
        ]
        if values:
            summary[f"average_{key}"] = statistics.mean(values)

    candidate_study = None
    if args.candidate_study:
        candidate_ks = [6, 8, 10, 12, 16]
        study_rows = []
        for candidate_k in candidate_ks:
            per_question = []
            for item in results:
                relevant_pages = item["relevant_pages"]
                if not relevant_pages:
                    continue
                ranked_pages = rank_candidate_pages(
                    item["retrieval_candidates"],
                    candidate_k=candidate_k,
                )
                per_question.append(
                    retrieval_study_metrics(
                        ranked_pages,
                        relevant_pages,
                        args.k,
                    )
                )

            study_rows.append({
                "candidate_k": candidate_k,
                "answerable_question_count": len(per_question),
                f"precision_at_{args.k}": statistics.mean(
                    row[f"precision_at_{args.k}"] for row in per_question
                ),
                f"recall_at_{args.k}": statistics.mean(
                    row[f"recall_at_{args.k}"] for row in per_question
                ),
                "mrr": statistics.mean(
                    row["reciprocal_rank"] for row in per_question
                ),
                "hit_at_1": statistics.mean(
                    row["hit_at_1"] for row in per_question
                ),
                f"ndcg_at_{args.k}": statistics.mean(
                    row[f"ndcg_at_{args.k}"] for row in per_question
                ),
            })

        no_rerank_rows = [
            item["no_rerank_metrics"]
            for item in results
            if item.get("no_rerank_metrics") is not None
        ]
        token_counts = [
            candidate["pair_token_count"]
            for item in results
            for candidate in item["retrieval_candidates"]
            if isinstance(candidate.get("pair_token_count"), int)
        ]
        token_counts_sorted = sorted(token_counts)

        def percentile(values, fraction):
            if not values:
                return None
            index = min(
                len(values) - 1,
                int((len(values) - 1) * fraction),
            )
            return values[index]

        candidate_study = {
            "note": (
                "Offline replay of the cached candidate pool and reranker scores. "
                "K values select prefixes of the original candidate order before "
                "sorting those selected candidates by their cached scores."
            ),
            "token_length_audit": {
                "pair_count": len(token_counts),
                "p50_pair_tokens": percentile(token_counts_sorted, 0.50),
                "p95_pair_tokens": percentile(token_counts_sorted, 0.95),
                "pairs_over_256_tokens": sum(
                    count > 256 for count in token_counts
                ),
                "pairs_over_384_tokens": sum(
                    count > 384 for count in token_counts
                ),
                "pairs_over_512_tokens": sum(
                    count > 512 for count in token_counts
                ),
            },
            "no_rerank_baseline": {
                "answerable_question_count": len(no_rerank_rows),
                f"precision_at_{args.k}": statistics.mean(
                    row[f"precision_at_{args.k}"] for row in no_rerank_rows
                ),
                f"recall_at_{args.k}": statistics.mean(
                    row[f"recall_at_{args.k}"] for row in no_rerank_rows
                ),
                "mrr": statistics.mean(
                    row["reciprocal_rank"] for row in no_rerank_rows
                ),
                "hit_at_1": statistics.mean(
                    row["hit_at_1"] for row in no_rerank_rows
                ),
                f"ndcg_at_{args.k}": statistics.mean(
                    row[f"ndcg_at_{args.k}"] for row in no_rerank_rows
                ),
            },
            "reranked_candidate_k_sweep": study_rows,
        }

    report = {
        "dataset": str(dataset_path),
        "question_count": len(results),
        "k": args.k,
        "summary": summary,
        "candidate_study": candidate_study,
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
    if candidate_study is not None:
        print("\nCandidate study:")
        print(json.dumps(candidate_study, indent=2))


if __name__ == "__main__":
    main()
