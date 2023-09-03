import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import numpy as np
import openai
import os
import json

#function to create sentences
def make_sentences(text):
    list_of_sentences = []
    sentences = text.split('.')
    for sentence in sentences:
        list_of_sentences.append(sentence)
    return list_of_sentences

#function to create paragraphs
def make_paragraphs(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size-overlap)]
    return [' '.join(chunk) for chunk in chunks]

#function to create embeddings
def make_embeddings(text):
    #making chunks of text
    sentences = make_sentences(text)
    paragraphs = make_paragraphs(text)
    
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    #Making Sentence Embeddings
    sentence_embeddings = []
    for sentence in tqdm(sentences,desc="Creating Sentence Embeddings"):
      encoded_input = tokenizer(sentence, return_tensors='pt',max_length=512, truncation=True, padding='max_length')
      with torch.no_grad():
          model_output = model(**encoded_input)
      embedding = torch.mean(model_output.last_hidden_state, dim=1).squeeze().numpy()
      sentence_embeddings.append(embedding.tolist())
    sentence_embedding_dict = dict(zip(sentences, sentence_embeddings))

    #Making Paragraph Embeddings
    paragraph_embeddings = []
    for paragraph in tqdm(paragraphs,desc="Creating Paragraph Embeddings"):
      encoded_input = tokenizer(paragraph, return_tensors='pt',max_length=512, truncation=True, padding='max_length')
      with torch.no_grad():
          model_output = model(**encoded_input)
      embedding = torch.mean(model_output.last_hidden_state, dim=1).squeeze().numpy()
      paragraph_embeddings.append(embedding.tolist())
    paragraph_embedding_dict = dict(zip(paragraphs, sentence_embeddings))
    
    return sentence_embedding_dict, paragraph_embedding_dict


#Reading text
with open(r'data\Harry_Potter_and_the_Chamber_of_Secrets.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#Making Embeddings
try:
    sentence_embeddings, paragraphs_embeddings = make_embeddings(text)
except Exception as e:
    print(e)

#saving embeddings in disk
try:
    with open(r'embeddings\sentence_embeddings.json', "w") as json_file:
        json.dump(sentence_embeddings, json_file)
except Exception as e:
    print(e)

try:
    with open(r'embeddings\paragraph_embeddings.json', "w") as json_file:
        json.dump(paragraphs_embeddings, json_file)
except Exception as e:
    print(e)
