import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load Excel data
def load_data(file_path):
    excel = pd.ExcelFile(file_path)
    data_frames = []
    for sheet in excel.sheet_names:
        df = excel.parse(sheet)
        if {"異常事故主旨", "事故說明", "問題解決方式"}.issubset(df.columns):
            df = df[["異常事故主旨", "事故說明", "問題解決方式"]].dropna()
            data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)

# Build embeddings
def build_embeddings(df, model):
    texts = df["事故說明"].tolist()
    embeddings = model.encode(texts, convert_to_tensor=True)
    return texts, embeddings

# Semantic search
def semantic_search(user_question, df, model, corpus_embeddings):
    query_embedding = model.encode(user_question, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    best_idx = int(scores.argmax())
    
    result = df.iloc[best_idx]
    return {
        #"問題": result["異常事故主旨"],
        #"回答": result["事故說明"],
        "解決方案": result["問題解決方式"]
    }

# Main
if __name__ == "__main__":
    # Load model and data
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # good for Chinese
    df = load_data("cathey_new_data.xlsx")
    corpus_texts, corpus_embeddings = build_embeddings(df, model)

    # User input
    question = "跨行轉帳變得很慢，是什麼問題？"
    answer = semantic_search(question, df, model, corpus_embeddings)

    #print("問題：", answer["問題"])
    #print("回答：", answer["回答"])
    print("解決方案：", answer["解決方案"])
