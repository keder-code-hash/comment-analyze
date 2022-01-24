# streamlit configuration 
import streamlit as st
from prediction import *
 
st.title("Classification of Comment ")
st.markdown("**Objective**: Given a details comment.MOdel will predict the sense positive or toxic If it is toxic how much toxic it is,will be also predicted.")
st.markdown('The model will predict some sense like, Positive,Toxic,Severe Toxic,Obscene,Threat,Insult,Identity Hate')

st.markdown("**Please enter the Comment**")
comment = st.text_input('Enter Comment here',placeholder='Enter Comment here',max_chars=128,autocomplete='default') 


def predict_sense(comment_text):
    bert_op=tokenize(filter_comment(comment_text))  
    classification_model=load_model("./static/model/my_custom_train_model.h5")
    result=classification_model.predict(bert_op) 
    predictions=list(result[0])

    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate','positive']
    fig=plt.figure(figsize=(10,5)) 
    plt.bar(label_cols,predictions,color="red",width=0.2)
    plt.xlabel("Comment Sense")
    plt.ylabel("value of Sense")
    plt.title("Comment Sense Analysis") 
    st.pyplot(fig)


if st.button("Predict"):
    predict_sense(comment)