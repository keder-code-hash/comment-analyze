from json import load
from unittest import result
import numpy as np
import pandas as pd
import re
import tensorflow as tf
import transformers 
from tqdm.notebook import tqdm
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences 
import matplotlib.pyplot as plt

tokenizer=transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
max_length=128
bert_model=transformers.TFBertModel.from_pretrained("./static/model/bert-base-uncased/")
bert_model.trainable=True
 
def filter_comment(comment): 
    comment=re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", comment)
    return [comment]

def tokenize(data,tokenizer=tokenizer,max_length=max_length): 
    bert_outputs=[] 
    encoded_data=tokenizer.batch_encode_plus(
                    data,
                    add_special_tokens=True,
                    max_length=max_length,
                    truncation=True,
                    return_attention_mask=True,
                    return_token_type_ids=False,
                    pad_to_max_length=True,
                    return_tensors="tf",
                )
        
    bert_output=bert_model(**encoded_data)
    sequence_output = bert_output.last_hidden_state
    bert_outputs.append(sequence_output)
    return bert_outputs

 