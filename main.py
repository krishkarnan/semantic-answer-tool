from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import pandas as pd
from sentence_transformers import SentenceTransformer, util

app = FastAPI(
    title="Semantic QA Tool",
    version="1.0.0",
    description="API to answer Chinese abnormal event questions based on Excel data."
)

class AnswerResponse(BaseModel):
    solution: str

# Load model and Excel
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
df = pd.read_excel("cathey_new_data.xlsx", sheet_name=None)

data_frames = []
for name, sheet in df.items():
    if {"異常事故主旨", "事故說明", "問題解決方式"}.issubset(sheet.columns):
        data_frames.append(sheet[["異常事故主旨", "事故說明", "問題解決方式"]].dropna())
df_all = pd.concat(data_frames, ignore_index=True)

corpus_texts = df_all["事故說明"].tolist()
corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True)

@app.get("/semantic-answer", response_model=AnswerResponse)
def get_solution(question: str = Query(..., description="User's Chinese question")):
    query_embedding = model.encode(question, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    best_idx = int(scores.argmax())
    result = df_all.iloc[best_idx]
    return {"solution": result["問題解決方式"]}
