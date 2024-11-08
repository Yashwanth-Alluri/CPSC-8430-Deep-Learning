# Extractive Question Answering on Spoken-SQuAD Using BERT

This repository contains code for building a BERT-based extractive question-answering model on ASR-transcribed spoken data using the Spoken-SQuAD dataset. The project includes three model configurations, each demonstrating the impact of progressive improvements:

## Models

- **Simple**: `bert-base-uncased`
- **Medium**: `bert-base-uncased` with learning rate decay and overlapping windows (`doc_stride=128`)
- **Strong**: `albert-large-v2` with automatic mixed precision (AMP) for optimized performance

All models are sourced from Hugging Face and fine-tuned to improve answer span accuracy and efficiency.
