import numpy as np 
import re 
import transformers  
from keras.models import load_model   

from rest_framework.response import Response
from rest_framework.views import APIView,status

from django.http import HttpResponse

bert_model=transformers.TFBertModel.from_pretrained("./staticfiles/model/bert-base-uncased")
tokenizer=transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True) 
 

def filter_comment(comment): 
    comment=re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", comment)
    return [comment]

def tokenize(data,tokenizer=None,max_length=128): 
    bert_outputs=[] 
    bert_model.trainable=True
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

def get_results(result_arr):
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate','positive']
    predictions=list(result_arr[0])
    sense=label_cols[np.argmax(result_arr)]
    return (predictions,sense)
  


 
def getCommentAnalyse(request):
    # request=request.data
    # comment=request['comment']
    comment="Thank you for understanding. I think very highly of you and would not revert without discussion."
    # filtered_comment=filter_comment(comment)
    bert_op=tokenize([comment],tokenizer)
    classification_model=load_model("./staticfiles/model/my_custom_train_model.h5")
    result=classification_model.predict(bert_op) 
    final_result_arr,predicted_sense=get_results(result)
    return HttpResponse(predicted_sense,status.HTTP_200_OK)