import streamlit as st 
from prompt_template import combined_chain 
from prompt_template import parallel_chain

st.title("Email Assistant")

email_topic=st.text_input("Email Topic")
recipient=st.text_input("Recipient")
name=st.text_input("Name")

def generate_email():
    email=parallel_chain.invoke({
        "email_topic":email_topic,
        "recipient":recipient,
        "name":name,
        "question":email_topic,
    })
    st.markdown(email["combined_chain"]["final_email"])
    st.markdown(email["subject_line"]["subject_line"])
  

st.button("Generate Email", on_click=generate_email)
