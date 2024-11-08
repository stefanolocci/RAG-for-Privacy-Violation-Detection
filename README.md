# RAG for Privacy Violation Detection

Leveraging Retrieval-Augmented Generation (RAG) for detecting privacy risks and enhancing explainability in data analysis.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Contributing](#contributing)

## About

This project uses RAG for privacy explainable violation detection. The RAG offers two retriever options (BM25 or FAISS) along with a generator to detect privacy risks in texts.

## Features

- **Privacy Violation Detection**: Detects and flags potential privacy issues in text documents.
- **Explainability**: Provides insight into why a text is flagged as a privacy risk.
- **Retrievers**: Offers both BM25 and FAISS retrievers for sparse and dense options.
- **Metrics and Evaluation**: Tools to evaluate the performance.


## Structure

- `dataset/`: Contains the annotated datasets.
- `classifier_FAISS.py`: FAISS-based privacy violation classifier.
- `classifier_bm25.py`: BM25-based privacy violation classifier.
- `email_parser.py`: Parses email data.
- `paper_parser.py`: Parses academic papers.
- `paper_retrieval.py`: Retrieves relevant data from academic papers.
- `metrics.py`: Metrics and evaluation tools.
- `utils.py`: Utility functions.

## Contributing

Contributions are welcome! Please submit a pull request with your improvements or open an issue to discuss changes.

