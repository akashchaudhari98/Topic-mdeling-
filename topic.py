import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
stop_words = set(stopwords.words('english')) 
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
import umap
nlp = spacy.load("en_core_web_sm")

class topic:
    def __init__(self,documents) -> None:
        self.docs = documents
        self.embedding_model = "D:/projects/ZInc/topic modelling/distilbert-base-nli-max-tokens/"
        #self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.bert_model = SentenceTransformer(model_name_or_path = self.embedding_model)
        
        #docs = self._preprocess(self.docs)
        self._umap()
        #print(x)

    def _preprocess(self,docs):
        clean_data = []
        lemma = WordNetLemmatizer()
        for doc in self.docs:
            #remove all non alphanumeric characters
            doc = str(doc)
            doc =   re.sub(r'\W+', ' ', doc)
            #remove all stop words
            word_tokens = word_tokenize(doc) 
            doc = " ".join([word for word in doc.split(" ") if word not in stop_words])
            #lemmatize words
            normalized = " ".join([lemma.lemmatize(word) for word in doc.split(" ")])
            doc = nlp(normalized)
            #remove prepositions
            doc = " ".join([str(token) for token in doc if token.pos_ != "ADP"])

            clean_data.append(doc)
        return clean_data

    def _USE_embedding(self,docs):
        
        cleaned_docs = self._preprocess(docs)
        embeddings = self.embed([cleaned_docs])

        return embeddings

    def _BERT_embedding(self):

        cleaned_docs = self._preprocess(docs)
        #tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
        #model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
        embeddings = self.bert_model.encode(cleaned_docs)
        print(embeddings)
        return embeddings
    
    def _umap(self):
        umap_args = {'n_neighbors': 15,
                         'n_components': 5,
                         'metric': 'cosine'}
        umap_model = umap.UMAP(**umap_args).fit(self._BERT_embedding())
        umap_emd = umap_model.embedding_
        print(umap_emd)

    def main(self):
        pass

if __name__ == "__main__":
    docs = [["his tutorial walks you through how to package a simple Python project"],["It will show you how to add the necessary files and structure to create the package, "],
        ["how to build the package"],["and how to upload it to the Python Package Index."]]
    topic = topic(docs)
    #print(topic._USE_embedding())