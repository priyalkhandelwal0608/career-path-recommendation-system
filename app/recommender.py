import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import Preprocessor

class CollaborativeRecommender:
    def __init__(self, path="data/careers.csv"):
        prep = Preprocessor(path)
        df = prep.clean_data()
        self.df = prep.feature_engineering()
        self.feature_cols = [
            "GPA","Extracurricular_Activities","Internships","Projects",
            "Leadership_Positions","Field_Specific_Courses","Research_Experience",
            "Coding_Skills","Communication_Skills","Problem_Solving_Skills",
            "Teamwork_Skills","Analytical_Skills","Presentation_Skills",
            "Networking_Skills","Industry_Certifications"
        ]
        self.feature_matrix = self.df[self.feature_cols].values

    def recommend(self, idx, top_k=3):
        sims = cosine_similarity([self.feature_matrix[idx]], self.feature_matrix)[0]
        top_indices = sims.argsort()[-top_k-1:-1][::-1]
        return self.df.iloc[top_indices][["Field","Career"]]