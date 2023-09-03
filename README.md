# Text_QnA_using_Bert_and_ChatGPT
Building Robust Large Text Q&amp;A Systems with BERT and ChatGPT 3.5 Turbo

To understand in more detail please go through my bolg on Median:
https://medium.com/@mhdsahilkhn/building-robust-large-text-q-a-systems-with-bert-and-chatgpt-3-5-turbo-with-code-6101a7a43d58

To tackle the challenge of retrieving relevant information from a large corpora of text, one effective approach is to leverage the power of BERT (Bidirectional Encoder Representations from Transformers) and ChatGPT 3.5 Turbo. The integration of BERT and ChatGPT 3.5 Turbo has offered a powerful and dynamic approach to address this challenge. By combining BERT's contextual understanding with ChatGPT 3.5 Turbo's conversational abilities, we can navigate a vast corpora of text, providing users with accurate, personalised, and engaging information retrieval experiences.

Leveraging BERT, a powerful language model, can help retrieve relevant information from large text corpora. BERT generates contextualised word embeddings that capture semantic meaning. Here we explore how BERT generates embeddings for text chunks like sentences and paragraphs, and the benefits of storing them on disk for efficient retrieval. BERT excels at understanding word context and meaning, learning to encode representations that consider both preceding and subsequent text. These contextualised embeddings capture nuanced relationships, enhancing information retrieval accuracy.

# For Example
For demonstration purposes, I've chosen the beloved "Harry Potter and the Chamber of Secrets" book's text as the foundation of our project. This iconic literary work not only provides a familiar and engaging context but also showcases how advanced AI techniques like BERT embeddings and ChatGPT can be applied to a diverse range of content.

To enhance the user experience and make our project accessible, I've employed Streamlit as the frontend framework. Streamlit is known for its simplicity and efficiency in creating interactive and visually appealing web applications.

![02](https://github.com/MohammadSahil/Text_QnA_using_Bert_and_ChatGPT/assets/49077018/45f1671d-2759-494b-9def-8d83570bbfca)

![03](https://github.com/MohammadSahil/Text_QnA_using_Bert_and_ChatGPT/assets/49077018/7826c7e4-b2c3-4c4a-8fe5-e82fd0dcaecf)

# To run the application

1. Install requirements.txt
2. Create Embeddings using create_embeddings.py
3. Create your OpenAI key and past in query_data.py
4. Then run the streamlit app using "streamlit run app_streamlit.py"
