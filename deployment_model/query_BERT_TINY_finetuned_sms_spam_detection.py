from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
import pandas as pd
import numpy as np

print('Starting model...')

checkmate_tokenizer = AutoTokenizer.from_pretrained(
    '../models/tokeniser/BERT_TINY_finetuned_sms_spam_detection')
checkmate_model = TFAutoModelForSequenceClassification.from_pretrained(
    '../models/classifier/BERT_TINY_finetuned_sms_spam_detection')

print('Model loaded!')


def classify(message):
    ''' Classify messages into trivial and non-trivial messages - 1 = non-trivial & 0 = trivial
    :param message: string
    :type message: message to classify
    :rtype: int
    :return: 1 for non-trivial, 0 for trivial'''

    tokens = checkmate_tokenizer(
        message, return_tensors='np', padding=True, truncation=True)

    output = checkmate_model(tokens).logits

    prediction_spam_test = np.argmax(output, axis=1)
    return prediction_spam_test[0]


text1 = """Watch Channel 514 for pre coronation arrangement"""
print(classify(text1))
