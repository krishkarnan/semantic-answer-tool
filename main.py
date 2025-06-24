# from fastapi import FastAPI, Query
# from pydantic import BaseModel
# from typing import List
# import pandas as pd
# from sentence_transformers import SentenceTransformer, util

# app = FastAPI(
#     title="Semantic QA Tool",
#     version="1.0.0",
#     description="API to answer Chinese abnormal event questions based on Excel data."
# )

# class AnswerResponse(BaseModel):
#     solution: str

# # Load model and Excel
# model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# df = pd.read_excel("cathey_new_data.xlsx", sheet_name=None)

# data_frames = []
# for name, sheet in df.items():
#     if {"異常事故主旨", "事故說明", "問題解決方式"}.issubset(sheet.columns):
#         data_frames.append(sheet[["異常事故主旨", "事故說明", "問題解決方式"]].dropna())
# df_all = pd.concat(data_frames, ignore_index=True)

# corpus_texts = df_all["事故說明"].tolist()
# corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True)

# @app.get("/semantic-answer", response_model=AnswerResponse)
# def get_solution(question: str = Query(..., description="User's Chinese question")):
#     query_embedding = model.encode(question, convert_to_tensor=True)
#     scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
#     best_idx = int(scores.argmax())
#     result = df_all.iloc[best_idx]
#     return {"solution": result["問題解決方式"]}


from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
from functools import lru_cache
from langchain.embeddings import WatsonxEmbeddings
from langchain.vectorstores import FAISS
import os

app = FastAPI()

class AnswerResponse(BaseModel):
    solution: str

@lru_cache()
def get_vector_db():
    project_id = "01c9dc60-0b88-4b26-ba1e-624820af527b"
    api_key = "JxsGnk03edo6N6XN0mCYkkb-Mf6ACOn608JCYR0eNCZe"
    watsonx_url = "https://us-south.ml.cloud.ibm.com"
    model_id = "ibm/slate-bert-base-uncased"

    df = pd.read_excel("cathey_new_data.xlsx", sheet_name=None)
    data_frames = []
    for name, sheet in df.items():
        if {"異常事故主旨", "事故說明", "問題解決方式"}.issubset(sheet.columns):
            data_frames.append(sheet[["異常事故主旨", "事故說明", "問題解決方式"]].dropna())
    df_all = pd.concat(data_frames, ignore_index=True)

    embedder = WatsonxEmbeddings(
        model_id=model_id,
        url=watsonx_url,
        api_key=api_key,
        project_id=project_id
    )

    texts = df_all["事故說明"].tolist()
    vectordb = FAISS.from_texts(texts, embedding=embedder)
    return vectordb, df_all

@app.get("/semantic-answer", response_model=AnswerResponse)
def get_solution(question: str):
    vectordb, df_all = get_vector_db()
    result = vectordb.similarity_search(question, k=1)[0]
    matched = df_all[df_all["事故說明"] == result.page_content].iloc[0]
    return {"solution": matched["問題解決方式"]}
