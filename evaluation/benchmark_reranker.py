import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np


MODEL_NAME = "BAAI/bge-reranker-v2-m3"


def load_cases(dataset_path: Path, candidate_k: int):
    report = json.loads(dataset_path.read_text(encoding="utf-8"))
    results = report.get("results")
    if not isinstance(results, list) or not results:
        raise SystemExit("Input must be a candidate-study report with a results list.")

    cases = []
    for item in results:
        candidates = item.get("retrieval_candidates")
        if not isinstance(candidates, list) or not candidates:
            raise SystemExit(
                "The input report is missing retrieval_candidates. "
                "Generate it with retrieval_eval.py --candidate-study first."
            )

        selected = candidates[:candidate_k]
        question = item.get("retrieval_query") or item.get("question")
        if not question:
            raise SystemExit("A question is missing from one evaluation result.")

        pairs = []
        pages = []
        for candidate in selected:
            text = candidate.get("text")
            page = candidate.get("page")
            if not isinstance(text, str) or page is None:
                raise SystemExit(
                    "Each retrieval candidate must contain text and page fields."
                )
            pairs.append((question, text))
            pages.append(int(page))

        cases.append({
            "id": item.get("id"),
            "question": question,
            "relevant_pages": sorted(set(item.get("relevant_pages", []))),
            "pairs": pairs,
            "pages": pages,
        })

    return cases


def unique_pages(pages):
    seen = set()
    result = []
    for page in pages:
        if page not in seen:
            seen.add(page)
            result.append(page)
    return result


def evaluate_rankings(cases, scores_by_case, k: int = 4):
    precision_values = []
    recall_values = []
    reciprocal_ranks = []

    for case, scores in zip(cases, scores_by_case):
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )
        retrieved_pages = unique_pages(
            [case["pages"][index] for index in ranked_indices[:k]]
        )
        relevant = set(case["relevant_pages"])
        if not relevant:
            continue

        hits = sum(page in relevant for page in retrieved_pages)
        precision_values.append(hits / k)
        recall_values.append(hits / len(relevant))

        reciprocal_rank = 0.0
        for rank, page in enumerate(retrieved_pages, start=1):
            if page in relevant:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)

    return {
        "answerable_question_count": len(precision_values),
        "precision_at_4": statistics.mean(precision_values) if precision_values else None,
        "recall_at_4": statistics.mean(recall_values) if recall_values else None,
        "mrr": statistics.mean(reciprocal_ranks) if reciprocal_ranks else None,
    }


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def benchmark_variant(name, cases, predict, warmup: int, repeats: int, candidate_k: int):
    warmup_pairs = cases[0]["pairs"]
    for _ in range(warmup):
        predict(warmup_pairs)

    latency_samples = []
    last_scores = []
    repeat_wall_times = []

    for repeat_index in range(repeats):
        repeat_start = time.perf_counter()
        repeat_scores = []
        for case in cases:
            start = time.perf_counter()
            scores = np.asarray(predict(case["pairs"])).reshape(-1)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if len(scores) != len(case["pairs"]):
                raise RuntimeError(
                    f"{name} returned {len(scores)} scores for "
                    f"{len(case['pairs'])} candidate pairs."
                )
            latency_samples.append(elapsed_ms)
            repeat_scores.append(scores.tolist())
        repeat_wall_times.append((time.perf_counter() - repeat_start) * 1000)
        last_scores = repeat_scores
        print(
            f"{name}: completed repeat {repeat_index + 1}/{repeats}"
        )

    total_candidates = sum(len(case["pairs"]) for case in cases)
    total_inference_seconds = sum(latency_samples) / 1000
    metrics = evaluate_rankings(cases, last_scores)

    return {
        "model_variant": name,
        "candidate_k": candidate_k,
        "question_count": len(cases),
        "total_candidate_pairs_per_repeat": total_candidates,
        "warmup_runs": warmup,
        "measured_repeats": repeats,
        "average_question_latency_ms": statistics.mean(latency_samples),
        "median_question_latency_ms": statistics.median(latency_samples),
        "p95_question_latency_ms": percentile(latency_samples, 0.95),
        "average_full_dataset_wall_time_ms": statistics.mean(repeat_wall_times),
        "candidate_pairs_per_second": (
            total_candidates * repeats / total_inference_seconds
            if total_inference_seconds else None
        ),
        **metrics,
    }


def export_and_quantize(model_name: str, model_dir: Path, force_export: bool):
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from onnxruntime.quantization import QuantType, quantize_dynamic

    fp32_dir = model_dir / "fp32"
    int8_dir = model_dir / "int8"
    fp32_model_path = fp32_dir / "model.onnx"
    int8_model_path = int8_dir / "model_quantized.onnx"

    if force_export and model_dir.exists():
        import shutil
        shutil.rmtree(model_dir)

    if not fp32_model_path.exists():
        fp32_dir.mkdir(parents=True, exist_ok=True)
        print(f"Exporting {model_name} to ONNX. This is a one-time setup step.")
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            model_name,
            export=True,
        )
        ort_model.save_pretrained(fp32_dir)
        del ort_model
        gc.collect()

    if not fp32_model_path.exists():
        available = [path.name for path in fp32_dir.iterdir()]
        raise RuntimeError(
            "ONNX export did not create model.onnx. "
            f"Files found in export directory: {available}"
        )

    if not int8_model_path.exists():
        int8_dir.mkdir(parents=True, exist_ok=True)
        print("Applying dynamic INT8 quantization to the ONNX model.")
        quantize_dynamic(
            model_input=str(fp32_model_path),
            model_output=str(int8_model_path),
            weight_type=QuantType.QInt8,
            per_channel=True,
        )

    if not int8_model_path.exists():
        raise RuntimeError("ONNX Runtime did not create the quantized model.")

    return fp32_model_path, int8_model_path


def make_ort_predictor(model_path: Path, tokenizer, threads: int, max_length: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_names = {item.name for item in session.get_inputs()}

    def predict(pairs):
        queries = [pair[0] for pair in pairs]
        passages = [pair[1] for pair in pairs]
        encoded = tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        feeds = {
            name: value
            for name, value in encoded.items()
            if name in input_names
        }
        if "token_type_ids" in input_names and "token_type_ids" not in feeds:
            feeds["token_type_ids"] = np.zeros_like(encoded["input_ids"])
        missing_inputs = input_names - set(feeds)
        if missing_inputs:
            raise RuntimeError(
                "The ONNX model requires unsupported inputs: "
                + ", ".join(sorted(missing_inputs))
            )
        outputs = session.run(None, feeds)
        return np.asarray(outputs[0]).reshape(-1)

    return session, predict


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark PyTorch FP32, ONNX FP32, and ONNX Runtime dynamic INT8 "
            "for the CourseMateAI cross-encoder reranker."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Candidate-study JSON created by retrieval_eval.py --candidate-study.",
    )
    parser.add_argument("--candidate-k", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--model-dir",
        default="evaluation/models/bge-reranker-v2-m3-onnx",
    )
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--output", default="evaluation/results/reranker_onnx_int8_benchmark.json")
    args = parser.parse_args()

    if args.candidate_k < 1 or args.warmup < 0 or args.repeats < 1:
        raise SystemExit("candidate-k/repeats must be positive and warmup cannot be negative.")

    dataset_path = Path(args.dataset)
    cases = load_cases(dataset_path, args.candidate_k)
    print(
        f"Loaded {len(cases)} questions and "
        f"{sum(len(case['pairs']) for case in cases)} candidate pairs per repeat."
    )

    try:
        import torch
        from sentence_transformers import CrossEncoder
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "Benchmark dependencies are missing. Install the project's "
            "requirements and retry."
        ) from error

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    model_dir = Path(args.model_dir)
    fp32_model_path, int8_model_path = export_and_quantize(
        MODEL_NAME,
        model_dir,
        args.force_export,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\nBenchmarking PyTorch FP32 on CPU.")
    cross_encoder = CrossEncoder(
        MODEL_NAME,
        max_length=args.max_length,
        device="cpu",
    )

    def predict_torch(pairs):
        return cross_encoder.predict(
            pairs,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    pytorch_result = benchmark_variant(
        "pytorch_fp32",
        cases,
        predict_torch,
        args.warmup,
        args.repeats,
        args.candidate_k,
    )
    del cross_encoder
    gc.collect()

    print("\nBenchmarking ONNX Runtime FP32 on CPU.")
    fp32_session, predict_onnx_fp32 = make_ort_predictor(
        fp32_model_path,
        tokenizer,
        args.threads,
        args.max_length,
    )
    onnx_fp32_result = benchmark_variant(
        "onnxruntime_fp32",
        cases,
        predict_onnx_fp32,
        args.warmup,
        args.repeats,
        args.candidate_k,
    )
    del fp32_session, predict_onnx_fp32
    gc.collect()

    print("\nBenchmarking ONNX Runtime dynamic INT8 on CPU.")
    int8_session, predict_onnx_int8 = make_ort_predictor(
        int8_model_path,
        tokenizer,
        args.threads,
        args.max_length,
    )
    onnx_int8_result = benchmark_variant(
        "onnxruntime_dynamic_int8",
        cases,
        predict_onnx_int8,
        args.warmup,
        args.repeats,
        args.candidate_k,
    )
    del int8_session, predict_onnx_int8
    gc.collect()

    baseline_latency = pytorch_result["average_question_latency_ms"]
    for result in [onnx_fp32_result, onnx_int8_result]:
        result["latency_reduction_vs_pytorch_percent"] = (
            (baseline_latency - result["average_question_latency_ms"])
            / baseline_latency * 100
            if baseline_latency else None
        )

    report = {
        "model": MODEL_NAME,
        "device": "CPU",
        "dataset": str(dataset_path),
        "candidate_k": args.candidate_k,
        "final_k": 4,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "methodology": (
            "Each backend receives the same query-passage pairs from the "
            "candidate-study report. Tokenization and model inference are "
            "included in each timed call. Model loading, ONNX export, and "
            "quantization are excluded. Each backend receives warm-up calls "
            "before the measured repeats."
        ),
        "results": [
            pytorch_result,
            onnx_fp32_result,
            onnx_int8_result,
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nBenchmark summary:")
    for result in report["results"]:
        print(
            f"{result['model_variant']}: "
            f"avg={result['average_question_latency_ms']:.2f} ms, "
            f"p50={result['median_question_latency_ms']:.2f} ms, "
            f"p95={result['p95_question_latency_ms']:.2f} ms, "
            f"P@4={result['precision_at_4']:.4f}, "
            f"R@4={result['recall_at_4']:.4f}, "
            f"MRR={result['mrr']:.4f}"
        )
    print(f"\nSaved benchmark report: {output_path}")


if __name__ == "__main__":
    main()
