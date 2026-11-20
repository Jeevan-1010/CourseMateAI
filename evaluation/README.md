# Retrieval evaluation

Use a manually reviewed, fixed test set for every before/after comparison.

Create \`evaluation/retrieval_questions.json\`:

\`\`\`json
[
  {
    "filename": "SQL (notes) (1).pdf",
    "question": "What is a primary key in SQL?",
    "relevant_pages": [16, 18]
  },
  {
    "filename": "SQL (notes) (1).pdf",
    "question": "Explain the different types of SQL joins with examples.",
    "relevant_pages": [41, 48, 52]
  }
]
\`\`\`

The page labels above are illustrative starting points from earlier retrieval observations. Verify every relevant page against the complete PDF before treating them as ground truth. Add all pages that contain a complete, useful answer.

Start FastAPI locally, upload/index the test document, then run from the repository root:

\`\`\`powershell
python evaluation/retrieval_eval.py --dataset evaluation/retrieval_questions.json --k 4 --output evaluation/results/optimized.json
\`\`\`

The evaluator reports macro Precision@K, Recall@K, and MRR (mean reciprocal rank), plus average retrieval, reranking, and end-to-end latency. It scores unique source pages in the order returned by the API.

Keep the same dataset, K, document version, and test conditions for every run. Save each report under a different filename. Do not compare results from different question sets as if they were a controlled before/after experiment.

The earlier project-report values (page precision 45.83%, page recall 100%, correctness 60.72%, faithfulness 91.1%, relevancy 84.71%) are a separate historical evaluation and are not interchangeable with these page-level Precision@K / Recall@K / MRR calculations.
