from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
from preprocessing import Preprocessor

class SemanticSearch:
    def __init__(self, path="data/careers.csv"):
        prep = Preprocessor(path)
        df = prep.clean_data()
        self.df = prep.feature_engineering()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(self.df["Career"].tolist(), convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query, top_k=3):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, top_k)
        return self.df.iloc[I[0]][["Field","Career"]]