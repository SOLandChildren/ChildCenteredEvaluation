#!/bin/bash
# This file contains the code that can be used to load one of our fine-tuned models and use it to make predictions
# On a test file.

# Importing all libraries
import sys
import torch
import numpy as np
import tensorflow as tf
import random as python_random
import pandas as pd
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

### This script only works on CUDA devices ###
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'CUDA device found: {torch.cuda.get_device_name(0)}')

else:
    print('No GPU available, exiting...')
    sys.exit(1)

### Setting up the seed, 42 was used to obtain our fine-tuned models ###
seed = 42

np.random.seed(seed)
tf.random.set_seed(seed)
python_random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

### Set this variable to true when you also have gold labels ###
gold = False

### The model from the fine-tuned models folder to use ###
model_to_use = 'GroNLP/mdebertav3-subjectivity-english'

### Location of the test data ###
test_path = '../data/kid-friend-en/en/inputs/trial_documents.jsonl'

### Model settings, best to set to those that were used during fine-tuning ###
with open(test_path) as f:
    lines = f.read().splitlines()

line_dicts = [json.loads(line) for line in lines]
df_test = pd.DataFrame(line_dicts)
df_test.drop_duplicates(inplace=True,ignore_index=True)
# df_test = pd.read_csv(test_path, sep='\t')
max_length = max([len(text) for text in df_test["main_content"]])
batch_size = 2

### Loading model and tokenizer ###
model = AutoModelForSequenceClassification.from_pretrained(model_to_use)
tokenizer = AutoTokenizer.from_pretrained(model_to_use)

def tok_data(sentences):
    """
    Function that for each sentence in a list tokenizes the sentence to BERT specific format.
    Tokenizer and settings that are defined above are used
    :param sentences: list of sentences
    :return: Two lists containing the input ids and attention masks
    """
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                          sent,
                          add_special_tokens = True,
                          max_length = max_length,
                          padding='max_length',
                          truncation=True,
                          return_attention_mask = True,
                          return_tensors = 'pt'
                    )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

### Tokenizing the data ###
test_input_ids, test_attention_masks = tok_data(df_test['main_content'].to_list())

def create_dataloader(input_ids, attention_masks, bs):
    """
    Creates a TensorDataset for the input_ids, attention_masks and labels.
    The TensorDataset is placed into a DataLoader using a SEQUENTIAL sampler.
    """
    dataset = TensorDataset(input_ids, attention_masks)
    return DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=bs)

def generate_predictions():

    test_dataloader = create_dataloader(test_input_ids, test_attention_masks, batch_size)
    
    print('Predicting labels for {:,} test sentences...'.format(len(df_test['main_content'])))

    ### Generating Predictions ###
    model.eval()
    model.cuda()
    
    predictions = []
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
      
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
    
        with torch.no_grad():
            result = model(b_input_ids,
                           token_type_ids=None, 
                           attention_mask=b_input_mask,
                           return_dict=True)
    
        logits = result.logits
    
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
    
    flat_predictions = np.concatenate(predictions, axis=0)
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    return flat_predictions

# ### Create a classification report and confusion matrix if we have gold labels ###
# if gold:
#     encoder = LabelBinarizer()
#     Y_dev_bin = encoder.fit_transform(df_test['label'].to_list())
#     Y_dev_bin = [i[0] for i in Y_dev_bin]
#     print(classification_report(Y_dev_bin, flat_predictions))
#     print(confusion_matrix(flat_predictions, Y_dev_bin))