import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run_eda(path="data/careers.csv"):
    # Load dataset
    df = pd.read_csv(path)

    # Basic overview
    print("Dataset Info:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())

    # Distribution of GPA
    plt.figure(figsize=(6,4))
    sns.histplot(df["GPA"], bins=10, kde=True, color="skyblue")
    plt.title("GPA Distribution")
    plt.xlabel("GPA")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Career counts
    plt.figure(figsize=(6,4))
    sns.countplot(y=df["Career"], palette="viridis")
    plt.title("Career Frequency")
    plt.xlabel("Count")
    plt.ylabel("Career")
    plt.tight_layout()
    plt.show()

    # Correlation heatmap for skills
    skill_cols = [c for c in df.columns if "Skills" in c or "Activities" in c]
    plt.figure(figsize=(10,8))
    sns.heatmap(df[skill_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Skill Correlation Heatmap")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_eda("data/careers.csv")