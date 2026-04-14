# -*- coding: utf-8 -*-

pip install sentence-transformers faiss-cpu

#sentence-transformers converts text into vector embeddings
#faiss-cpu efficient library for vector search - can find which vectors are most similar to a query vector

documents = [
    "Laminar flow reduces drag by keeping airflow smooth over the aircraft surface.",
    "Composite materials reduce aircraft weight and improve fuel efficiency.",
    "High aspect ratio wings increase aerodynamic efficiency.",
    "Fuel burn reduction lowers operational costs and environmental impact."
]

# each string is a document or a chunk of knowledge. In a real system, these could be PDFs, manuals or some website content.

#documents are converted into embeddings

from sentence_transformers import SentenceTransformer
import numpy as np

# we load a pretrained sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = model.encode(documents)
#converts each document into vector of numbers (size 384 for this model)

import faiss
#FAISS - Facebook AI Similarity Search (a library designed for fast vector similarity search)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(np.array(doc_embeddings))

def search(query, k=2):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [documents[i] for i in indices[0]]

results = search("How does laminar flow improve efficiency?")
print("file is running")
