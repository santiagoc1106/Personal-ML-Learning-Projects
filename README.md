# CineBot: Movie Chatbot

A personal project I built while learning machine learning. I used Claude to
help me work through the steps and write code.

## What it does
- `movieReviewModel.ipynb`: fine-tunes DistilBERT on the IMDB movie review
  dataset ([25,000 training / 25,000 test] reviews) for binary sentiment
  classification, and generates movie overviews with Qwen2.5-3B-Instruct.
- `movieSentimentProj.py`: a chat version. After an overview, the user can chat
  about the movie (up to 5 chats per session). The sentiment classifier reads
  each message, and Qwen adapts its tone to the result.

## Limitations
Overviews come from the model's own knowledge and are not fact-checked, so
plot details can be wrong.

## To run
1. Run `movieReviewModel.ipynb` to train and save the model to `./my-model-fixed`.
2. Run `python movieSentimentProj.py`.

Requires: transformers, datasets, evaluate, accelerate, torch.
