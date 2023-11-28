#
#
# Identify duplicates in a single dataset

import sys
import pandas as pd

from tqdm import tqdm

from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import CohereEmbeddings
from langchain.embeddings import CacheBackedEmbeddings

# for caching
from langchain.storage import LocalFileStore

from config_rag import (
    EMBED_TYPE,
    EMBED_HF_MODEL_NAME,
    EMBED_COHERE_MODEL_NAME,
    VECTOR_STORE_NAME,
)


#
# Functions
#
def check_cols(file_df):
    cols1 = file_df.columns

    if "text" not in cols1:
        print()
        print("The file should contain a column called text!!!")
        print()
        sys.exit(-1)

    return


def create_cached_embedder():
    print("Initializing Embeddings model...")

    # Introduced to cache embeddings and make it faster
    fs = LocalFileStore("./vector-cache/")

    if EMBED_TYPE == "COHERE":
        print("Loading Cohere Embeddings Model...")
        embed_model = CohereEmbeddings(
            model=EMBED_COHERE_MODEL_NAME, cohere_api_key="PUT_HERE"
        )
    elif EMBED_TYPE == "LOCAL":
        print(f"Loading HF Embeddings Model: {EMBED_HF_MODEL_NAME}")

        model_kwargs = {"device": "cpu"}
        # changed to True for BAAI, to use cosine similarity
        encode_kwargs = {"normalize_embeddings": True}

        embed_model = HuggingFaceEmbeddings(
            model_name=EMBED_HF_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    # the cache for embeddings
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embed_model, fs, namespace=embed_model.model_name
    )

    return cached_embedder


def create_vector_store(store_type, document_splits, embedder):
    print(f"Indexing: using {store_type} as Vector Store...")

    if store_type == "CHROME":
        # modified to cache
        vectorstore = Chroma.from_documents(
            documents=document_splits, embedding=embedder
        )
    elif store_type == "FAISS":
        # modified to cache
        vectorstore = FAISS.from_documents(
            documents=document_splits, embedding=embedder
        )

    return vectorstore


file_name1 = ""

if len(sys.argv) >= 2:
    file_name1 = sys.argv[1]
else:
    print()
    print("Not enough parameters provided.")
    print("Usage python check_duplicates.py file_name1")
    print()
    sys.exit(-1)

# We're using Pandas to easily read file and manipulate lines
file1_df = pd.read_csv(file_name1)

# check it has the column named text
check_cols(file1_df)

# info
print()
print(f"file contains {len(file1_df)} sentences!!!")
print()

sentences1 = list(file1_df["text"].values)
# create a list of Documents Object.. it is the format
# to load the vector db
documents1 = [Document(page_content=text) for text in sentences1]

# create cached embedder
embedder = create_cached_embedder()
# index and create a vector store
vector_store = create_vector_store(VECTOR_STORE_NAME, documents1, embedder)

#
# Here we search for duplicates
#
print("Searching for duplicates...")
all_docs_list = []

# here we search
TOP_K = 5

n_duplicates = 0
i = 0

THR = 0.005

for candidate_text in tqdm(sentences1):
    i += 1
    # search nearest neighbour. If more than 1 there are duplicates
    results = vector_store.similarity_search_with_score(query=candidate_text, k=TOP_K)

    n_zeros = 0
    for tupla in results:
        txt = tupla[0].page_content
        distance = tupla[1]

        if distance <= THR:
            n_zeros += 1

    if n_zeros > 1:
        n_duplicates += 1
        print(f"{candidate_text} has duplicates ({n_zeros}). Index is: {i}..")


print()
print("Report:")
print(f"Found {n_duplicates} duplicates..")
print()
