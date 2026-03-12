import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class Preprocessor:
    def __init__(self, path="data/careers.csv"):
        self.df = pd.read_csv(path)

    def clean_data(self):
        # Handle missing values
        self.df.fillna(0, inplace=True)

        # Strip whitespace from column names
        self.df.columns = [c.strip() for c in self.df.columns]

        # Drop duplicates
        self.df.drop_duplicates(inplace=True)

        return self.df

    def feature_engineering(self):
        # Normalize numeric skill features
        skill_cols = [
            "GPA","Extracurricular_Activities","Internships","Projects",
            "Leadership_Positions","Field_Specific_Courses","Research_Experience",
            "Coding_Skills","Communication_Skills","Problem_Solving_Skills",
            "Teamwork_Skills","Analytical_Skills","Presentation_Skills",
            "Networking_Skills","Industry_Certifications"
        ]
        scaler = MinMaxScaler()
        self.df[skill_cols] = scaler.fit_transform(self.df[skill_cols])

        # Encode categorical fields
        le_field = LabelEncoder()
        le_career = LabelEncoder()
        self.df["Field_encoded"] = le_field.fit_transform(self.df["Field"])
        self.df["Career_encoded"] = le_career.fit_transform(self.df["Career"])

        return self.df