import streamlit as st 
from prompt_template import parallel_chain

st.title("Email Assistant")

email_topic = st.text_input("Email Topic")
recipient = st.text_input("Recipient")
name = st.text_input("Name")

if st.button("Generate Email"):
    # Run the chain
    result = parallel_chain.invoke({
        "email_topic": email_topic,
        "recipient": recipient,
        "name": name,
        "question": email_topic
    })
    
    # Display Subject
    st.subheader("Subject")
    st.write(result["subject_line"]["subject_line"])
    
    # Display email
    st.subheader("Final Email")
    st.write(result["combined_chain"]["final_email"])
