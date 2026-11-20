# Retrieval evaluation

Use a manually reviewed, fixed test set for every before/after comparison. The historical SQL evaluation dataset (`evaluation/evaluation_dataset.json`) and its source PDF are local evaluation assets and are not currently included in the public repository. Supply your own dataset and matching uploaded/indexed document, or recreate the JSON file using the schema below, before running the example command.

The evaluator supports both:
- The expected `evaluation/evaluation_dataset.json` format (create this file locally; it is not tracked in the repository):
  ```json
  {
    "document": "coursemate_test_guide.pdf",
    "tests": [
      {
        "id": "Q01",
        "question": "What is artificial intelligence?",
        "relevant_pages": [1],
        "expected_answerable": true
      }
    ]
  }
  ```
- A simple list of cases with `filename`, `question`, and `relevant_pages`.

For the existing dataset, start FastAPI locally, ensure the named PDF is uploaded and indexed, then run from the repository root:

```powershell
python evaluation/retrieval_eval.py --dataset evaluation/evaluation_dataset.json --k 4 --output evaluation/results/baseline.json
```

The evaluator reports macro Precision@K, Recall@K, and MRR (mean reciprocal rank), plus average retrieval, reranking, and end-to-end latency. It scores unique source pages in the order returned by the API. Cases with no relevant page labels (unanswerable cases) are excluded from retrieval-ranking averages and counted separately.

Keep the same dataset, K, document version, and test conditions for every run. Save each report under a different filename. Do not compare results from different question sets as if they were a controlled before/after experiment.

The earlier project-report values (page precision 45.83%, page recall 100%, correctness 60.72%, faithfulness 91.1%, relevancy 84.71%) are a separate historical evaluation and are not interchangeable with these page-level Precision@K / Recall@K / MRR calculations.
