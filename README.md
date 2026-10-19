## This repository holds my personal ML project(s). I used Claude AI to help walk me through the steps of training several models. 

### * movieReviewModel.ipynb *
#### This notebook carries the framework for my sentiment analysis model. This uses the 25,000 IMDB movie review dataset on Hugging Face. It uses a DistilBERT pre-trained model which I then trained to identify sentiment on a binary basis. I also used a Qwen-2.5-3B language model to generate overviews on movies.


### * movieSentimentProj.py *
#### This Python file contains a more sophisticated version of the language model from movieReviewModel.ipynb. In this file, the user is able to interact with the model about a movie of their choice up to 5 times per session. The Qwen-2.5-3B model uses the feedback from the sentiment analysis model to adapt to user input.

