# streamlit configuration 
api_mode_on=True
if api_mode_on:
    import os 
    os.system('uvicorn main:app --reload ')
else:
    import streamlit as st
    from prediction import *
    from db import *
    show_graph=False
    st.title("Classification of Comment ")
    st.markdown("**Objective**: Given a details comment.MOdel will predict the sense positive or toxic If it is toxic how much toxic it is,will be also predicted.")
    st.markdown('The model will predict some sense like, Positive,Toxic,Severe Toxic,Obscene,Threat,Insult,Identity Hate')

    st.markdown("**Please enter the Comment**")
    comment = st.text_input('Enter Comment here',placeholder='Enter Comment here',max_chars=128,autocomplete='default') 


    def predict_sense(comment_text):
        label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate','positive']
        if len(comment_text)!=0:
            bert_op=tokenize(filter_comment(comment_text))  
            classification_model=load_model("./static/model/my_custom_train_model.h5")
            result=classification_model.predict(bert_op) 
            predictions=list(result[0])
        else: 
            predictions=[0,0,0,0,0,0,0] 
        fig=plt.figure(figsize=(10,5)) 
        plt.bar(label_cols,predictions,color="red",width=0.2)
        plt.xlabel("Comment Sense")
        plt.ylabel("value of Sense")
        plt.title("Comment Sense Analysis") 
        st.pyplot(fig) 

    show_graph=True
    if st.button("Predict") or show_graph:  
        predict_sense(comment)


    st.subheader("If the predcited sense is different plse put the correct one.")
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate','positive']
    selected_option=[]
    selected_val=st.multiselect(
        "Please select(you can select multiple sense.)",
        label_cols,
        selected_option
    )  

    if st.button("submit"):
        st.write("Your selected value :- ")
        for i in selected_val:
            st.write(i)
        update_data(prepare_data(comment,selected_val))
        st.markdown("`Thank you for your contribution.`😊😊") 

