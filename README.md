Simple Retrieval-Augmented Generation (RAG)

This project implements a basic Retrieval-Augmented Generation (RAG) pipeline using semantic embeddings and a language model.

Overview

The system:

Splits documents into chunks

Creates vector embeddings

Stores embeddings in a vector index

Retrieves relevant context for a query

Generates answers using an LLM

Architecture

User Query → Embedding Search → Context Retrieval → LLM → Answer

Tech Stack

Python

Sentence Transformers

FAISS

HuggingFace Transformers

Purpose

This project demonstrates foundational LLM pipeline development and retrieval-based augmentation techniques.
