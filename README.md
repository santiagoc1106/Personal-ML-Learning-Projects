## This repository holds several ML projects of mine that I worked on independently. I used Claude AI to help walk me through the steps of training several models. 

### *train.ipynb*
#### This notebook carries the training for my diabetes prediction model which uses a Random Forest Regressor to predict the diabetes risk score of a patient.

### * movieReviewModel.ipynb *
#### This notebook carries the framework for my sentiment analysis model. This uses the 25,000 IMDB movie review dataset on Hugging Face. It uses a DistilBERT pre-trained model which I then trained to identify sentiment on a binary basis. I also used a Qwen-2.5-3B language model to generate overviews on movies.


### * movieSentimentProj.py *
#### This Python file contains a more sophisticated version of the language model from movieReviewModel.ipynb. In this file, the user is able to interact with the model about a movie of their choice up to 5 times per session. The Qwen-2.5-3B model uses the feedback from the sentiment analysis model to adapt to user input.

### * weather.py *
#### This weather app uses the python_weather library for weather data up to 3 days from the current day. It displays weather trends (temperature, humidity, precipitation) to the user from a city of their choice.
