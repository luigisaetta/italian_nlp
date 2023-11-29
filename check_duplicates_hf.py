#
# Checking duplicates using Embeddings Vector
#

# this code loads two hf datasets
# it checks if any of the text contained in d2 is already contained in ds1
# 1. loads all the sentences in ds1
# 2. compute embeddings for each sentence in ds1 and store in Vector DB
# 3. load every text in ds2, compute embeddings and search vector in ds1 closer
# if istance < threshold, it is considered a duplicate and logged

import sys
import pandas as pd

from tqdm import tqdm

from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import CohereEmbeddings
from langchain.embeddings import CacheBackedEmbeddings

from datasets import load_dataset, concatenate_datasets, ClassLabel, Features, Value

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


# peculiarities regarding single dataset
# encapsulated in the function
def read_texts_ds1(dataset_name):
    # load Brand24/mms dataset
    ds1 = load_dataset(dataset_name)["train"]
    ds1.set_format(type="pandas")

    df1 = ds1[:]

    # filtra solo il linguaggio italiano
    df1 = df1[df1["language"] == "it"]

    print()
    print(f"ds1 contains {len(df1)} sentences!!!")
    print()

    sentences = list(df1["text"].values)
    return sentences


def read_texts_ds2(dataset_name):
    SUBSET_NAME = "italian"

    # load cardiff ita dataset
    ds2 = load_dataset(dataset_name, SUBSET_NAME)["train"]
    ds2.set_format(type="pandas")

    df2 = ds2[:]

    print()
    print(f"ds2 contains {len(df2)} sentences!!!")
    print()

    sentences = list(df2["text"].values)

    return sentences


#
# Main
#

# load dataset1 and 2
DATASET_NAME = "Brand24/mms"

sentences1 = read_texts_ds1(DATASET_NAME)

# create a list of Documents Object.. it is the format
# to load the vector db
documents1 = [Document(page_content=text) for text in sentences1]

# create cached embedder
embedder = create_cached_embedder()
# index and create a vector store
vector_store = create_vector_store(VECTOR_STORE_NAME, documents1, embedder)

# loading dataset ds2
DATASET_NAME = "cardiffnlp/tweet_sentiment_multilingual"

sentences2 = read_texts_ds2(DATASET_NAME)

#
# Here we search for duplicates
#
print("Searching for duplicates...")
all_docs_list = []

# here we search
for candidate_doc in tqdm(sentences2):
    # returning only the closest candidate (k=1)
    # return_list is (doc_in_db_closest, distance)
    return_list = vector_store.similarity_search_with_score(candidate_doc, k=1)

    all_docs_list.append((candidate_doc, return_list[0]))

# display results

# this one should be calibrated
THR = 0.005

tot_duplicated = 0

ref_list = []
closest_list = []
distance_list = []

for i, (doc, return_doc1) in enumerate(all_docs_list):
    closest, distance1 = return_doc1
    SIZE = 60

    print()
    print(f"Record n. {i}")
    ref_list.append(doc)
    closest_list.append(closest.page_content)
    distance_list.append(distance1)

    if distance1 < THR:
        tot_duplicated += 1
        print(f"Candidate: {doc[:SIZE]}...")
        print(f"Closest  : {closest.page_content[:SIZE]}..., distance: {distance1:.2f}")
        print(f"Duplicated found, distance: {distance1:.2f}")

print()

dict_results = {"ref": ref_list, "closest": closest_list, "dist": distance_list}

df_duplicates = pd.DataFrame(dict_results)

df_duplicates.to_csv("duplicates.csv", index=None)

print("Report:")
print()
print(f"Found {tot_duplicated} strong duplicated.")
print()
