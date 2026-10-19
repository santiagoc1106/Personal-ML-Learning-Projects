"""
CineBot: Movie Overview Chatbot
===============================
1. Fine-tunes DistilBERT on IMDB for binary sentiment classification, keeping the
   checkpoint with the lowest validation loss to avoid an overfit model.
2. Uses Qwen2.5-3B-Instruct to generate a short plot overview for a movie title.
3. After the overview, the user can chat with the model about the movie (up to 5
   chats per session). The sentiment classifier reads each message and adapts the
   chatbot's tone to the user's reaction.
4. Runs an emotion classifier on generated text when the request calls for it.

Note: overviews come from the model's own knowledge and are not fact-checked,
so plot details may be wrong.

I used Claude to walk me through the steps of this project and to help write the code.
See movieReviewModel.ipynb for how the models were trained.

Set TRAIN_FROM_SCRATCH = True in main to retrain, or leave it False to load the
saved model from ./my-model-fixed.
"""

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
import evaluate

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"       # only show actual errors, not info/warnings
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"      # suppress download progress bars
os.environ["TOKENIZERS_PARALLELISM"] = "false"        # silences a common tokenizer warning
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1" # suppress advisory-style warnings

import warnings
warnings.filterwarnings("ignore")  # catch any remaining Python-level warnings (e.g. deprecation notices)

from transformers.utils import logging
logging.set_verbosity_error()


# ============================================================
# SECTION 1: TRAINING THE SENTIMENT CLASSIFIER
# ============================================================

def train_sentiment_model(save_dir="./my-model-fixed"):
    """Fine-tunes DistilBERT on IMDB for binary sentiment classification.

    Uses load_best_model_at_end so the final model is whichever epoch had the
    LOWEST validation loss, not just the last epoch trained — this avoids
    keeping an overfit checkpoint even if its raw accuracy looked slightly higher.
    """

    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    print("Loading tokenizer and tokenizing dataset...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    print("Loading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        fp16=True,
        logging_steps=10,
        load_best_model_at_end=True,       # keep the best-generalizing checkpoint
        metric_for_best_model="eval_loss", # judge "best" by validation loss
        greater_is_better=False,           # lower loss is better
        report_to="none",                  # skip wandb/etc prompts
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Saving fixed model to {save_dir}...")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    return trainer.model, tokenizer


def load_sentiment_model(save_dir="./my-model-fixed"):
    """Loads a previously trained + saved sentiment model instead of retraining."""
    model = AutoModelForSequenceClassification.from_pretrained(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    return model, tokenizer


def check_sentiment(text, model, tokenizer):
    """Returns (predicted_label, probabilities) for a piece of text.

    Useful for sanity-checking model calibration — an overfit model tends to
    be falsely overconfident even on ambiguous text, while a well-fit model
    should show more balanced probabilities when the sentiment is genuinely mixed.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        probs = torch.softmax(outputs.logits, dim=-1)
    return prediction, probs


# ============================================================
# SECTION 2: MOVIE OVERVIEW + EMOTION PIPELINE
# ============================================================

def load_generator():
    """Loads the instruction-tuned text-generation model for plot overviews."""
    return pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-3B-Instruct",
        dtype=torch.bfloat16,
        device_map="auto",
    )


def load_emotion_classifier(): 
    """Loads the emotion classifier (returns scores across all emotion labels)."""
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
    )


def generate_movie_overview(movie_title, generator, emotion_classifier, prompt, history=None):
    """Generates a short plot overview for a movie and analyzes its emotional tone.

    Note: the overview is LLM-generated from the model's own knowledge, not pulled
    from a factual database — plot details (character names, settings, specific
    events) may be inaccurate, especially for smaller/older movies. Treat the
    overview as illustrative text for the emotion analysis, not a factual summary.
    """
    messages = (history or []) + [
        {"role": "user", "content": f"{prompt} about the movie {movie_title}."}
    ]

    classification = [
        {"role": "user", "content": f"Does this request call for descriptive, emotional, or narrative text? "
                                 f"Respond with ONLY the single word 'yes' or 'no' — no other text. "
                                 f"Request: {prompt}"}
    ]

    classResult = generator(classification, max_new_tokens = 200, do_sample = False)
    classText = classResult[0]["generated_text"][-1]["content"]


    result = generator(messages, max_new_tokens=200, do_sample=True, temperature=0.7)
    overview_text = result[0]["generated_text"][-1]["content"]

    print(f"(CineBot) {overview_text}")

    return overview_text


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # --- Sentiment model: train fresh, or load if already saved ---
    TRAIN_FROM_SCRATCH = False  # set True to retrain instead of loading a saved model

    if TRAIN_FROM_SCRATCH:
        sentiment_model, sentiment_tokenizer = train_sentiment_model()
    else:
        sentiment_model, sentiment_tokenizer = load_sentiment_model("./my-model-fixed")
    
    # --- Movie overview + emotion pipeline ---
    generator = load_generator()
    emotion_classifier = load_emotion_classifier()

    print("Movie Overview Chatbot")
    print("=======================")

    MAX_CHATS = 5 ## Hard limit on how many times the user can talk about the movie review
    CHAT_REVISIONS = 0


    movie_title = input("(CineBot) Enter a movie title: ")
    userPrompt = input(f"(CineBot) What would you like to know about {movie_title}? ")


    SYSTEM = {"role": "system", "content": (
        "You are CineBot, a friendly movie expert. Reply in 2-3 plain sentences "
        "with no headers, bullet points, or markdown. If a title could mean several "
        "movies, pick the most famous one and name it. If you're unsure about a "
        "detail, say so instead of guessing. Never claim personal experiences or favorites."
    )} # Sets a base for the model to understand its role and tone in the conversation

    history = [SYSTEM]
    turns = 0
    overview = generate_movie_overview(movie_title, generator, emotion_classifier, prompt=userPrompt, history=history)

    history += [
        {"role": "user", "content": f"{userPrompt} about the movie {movie_title}."},
        {"role": "assistant", "content": overview},
    ]

    while turns < MAX_CHATS:
        userChat = input(f"(User) ")

        if not userChat:
            break
        else:
            if (MAX_CHATS - turns == 2):
                print("Warning: only 2 more tokens left!")

            label, probs = check_sentiment(userChat, sentiment_model, sentiment_tokenizer)
            sentiment = "positive" if label == 1 else "negative"
            confidence = probs[0][label].item()

            DEBUG = False  # set True to see debug info about sentiment classification

            if confidence >= 0.75:
                tone = "positive" if label == 1 else "negative"
                hint = (f"(Private note: your friend seems to feel {tone} about this. "
                        f"Let that shape your tone, but never mention that you detected it.) ")
                ending = "acknowledging how they feel"
            else:
                hint = ""
                ending = "responding to what they said"

            reply = generate_movie_overview(
                movie_title, generator, emotion_classifier,
                prompt=f"You are a movie expert chatting with a friend. They said: '{userChat}'. "
                    f"{hint}Reply in 2-3 sentences, {ending}",
                history=history,
            )

            history += [
                {"role": "user", "content": userChat},
                {"role": "assistant", "content": reply},
            ]
            turns += 1

    print("Thank you for using this chatbot!")
        



