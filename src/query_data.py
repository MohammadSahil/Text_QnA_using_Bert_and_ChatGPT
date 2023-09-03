import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import numpy as np
import openai
import os
import json

#function to create embeddings
def create_query_embedding(query):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    encoded_input = tokenizer(query, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        model_output = model(**encoded_input)
    query_embedding = torch.mean(model_output.last_hidden_state, dim=1).squeeze().numpy()
    return query_embedding

#function to get similar sentences and paragraphs using cosine similarity
def get_similar_sentence_paragraphs(query):
    query_embedding= create_query_embedding(query)

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # loading embeddings
    with open(r'C:\Users\User\Desktop\text qna\embeddings\sentence_embeddings.json', "r") as json_file:
        sentence_embeddings = json.load(json_file)
    with open(r'C:\Users\User\Desktop\text qna\embeddings\paragraph_embeddings.json', "r") as json_file:
        paragraph_embeddings = json.load(json_file)

    # Calculate cosine similarity between query and sentences
    scores = []
    for embedding in list(sentence_embeddings.values()):
        score = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        scores.append(score)
    # Rank documents based on similarity scores
    results = np.argsort(scores)[::-1]
    similar_sentences = ''
    # Print top N search results
    top_n = 10
    for i in range(top_n):
        result_idx = results[i]
        score = scores[result_idx]
        similar_sentences += list(sentence_embeddings.keys())[result_idx]
        similar_sentences += ' '

    # Calculate cosine similarity between query and paragraphs
    scores = []
    for embedding in list(paragraph_embeddings.values()):
        score = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        scores.append(score)
    # Rank documents based on similarity scores
    results = np.argsort(scores)[::-1]
    similar_paragraphs = ''
    # Print top N search results
    top_n = 5
    for i in range(top_n):
        result_idx = results[i]
        score = scores[result_idx]
        similar_paragraphs += list(paragraph_embeddings.keys())[result_idx]
        similar_paragraphs += ' '
        
    return similar_sentences, similar_paragraphs

#Provide your openai api key
openai.api_key = ""

#function to use chatgpt chat completion to get the final answer.
def ChatGPT(text):
    chatgpt = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=[{"role": "assistant", "content": text}])
    answer =  chatgpt.choices[0].message
    return answer.content

#final function to get the answer
def get_answer(query):
    similar_sentences, similar_paragraphs = get_similar_sentence_paragraphs(query)
    context = similar_sentences + similar_paragraphs
    answer = ChatGPT(f'{context} Based on this preceeding information answer the question only, {query}')
    return answer