import streamlit as st
from src import query_data

st.title("Harry Potter and the Chamber of Secrets")
st.write("Ask anything from Harry Potter and the Chamber of Secrets...")

question = st.text_input('Ask your question')

if len(question) == 0:
    pass
else:
    with st.spinner('Please wait...'):
        answer = query_data.get_answer(question)
    st.success(answer)