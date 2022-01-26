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
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate','positive']
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
def predict_sense(comment_text,api_mode=False):
    if len(comment_text)!=0:
        bert_op=tokenize(filter_comment(comment_text))  
        classification_model=load_model("./static/model/my_custom_train_model.h5")
        result=classification_model.predict(bert_op) 
        predictions=list(result[0])
    else: 
        predictions=[0,0,0,0,0,0,0]
    if api_mode:
        json={}
        for i in range(len(label_cols)):
            json[label_cols[i]]=float(predictions[i])
        return json   
def prepare_data(comment_text,selected_sense):
        data={
            "comment_text":comment_text,
            }
        for label in label_cols:
            if label in selected_sense:
                data[label]=1
            else:
                data[label]=0
        return data


 