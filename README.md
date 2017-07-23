# whosup

Train classifiers with Whatsapp chat data and predict the author of new messages

# Requirements

`whosup` does not need to be installed, but has the following dependencies:

 - Python > 2.7
 - Python package `scikit-learn`
 - Python package `pandas`

# Usage

`whosup` is a Python script that can either train and save a model, or make predictions with a previously trained model. Train a model with the option `--train`:

    python whosup.py --train --model model.pkl --data training.txt

If `--data` is omitted, the trainer assumes training data is supplied from STDIN. The option `--model` is required in any case.

After training a model, use it for predictions with `--predict`:

    echo "completely new message! You'll never guess who wrote me." | python whosup.py --predict --model model.pkl

Alternatively, you can specify a file with a list of samples for which a class should be predicted with the option `--samples`.

Type `python whosup.py --help` for more details and advanced options.

# How to put together training data

The program works directly on the data dump currently provided by Whatsapp. On your phone, find a chat with at least two members, and have the contents emailed to you as a `*.txt` file:

https://faq.whatsapp.com/fil/android/23756533

This file can be used directly with the `--data` option for training. The chat should contain a sizeable number of messages, do not try anything below 1000 messages.
