# Career Path & Job Recommendation System  
**Tagline:** Hybrid ML system that recommends career paths using semantic search (BERT + FAISS) and collaborative filtering.

---

## 🚀 Overview / Elevator Pitch
This project helps students and professionals discover suitable career paths by analyzing their skills, GPA, and experiences. It combines **semantic search** (for career names) and **collaborative filtering** (for student profiles) into one hybrid recommender.  

**Why it matters:** Recruiters and career counselors can quickly match candidates to relevant career options, improving efficiency and personalization.  

**Technologies:** Python, Flask, scikit‑learn, BERT embeddings, FAISS, Bootstrap frontend, Docker deployment.

---


---

## ✨ Features
- Hybrid recommender system:
  - **Semantic Search:** Career name → BERT embeddings + FAISS similarity search.  
  - **Collaborative Filtering:** Row index → profile similarity based on skills/features.  
- Flask API with Bootstrap frontend.  
- Automatic detection of input type (career name vs. index).  
- EDA plots for recruiter‑friendly data insights.  
- Dockerized for easy deployment.

---

## 🛠 Tech Stack
- **Languages & Frameworks:** Python, Flask  
- **ML/NLP:** scikit‑learn, SentenceTransformers (BERT), FAISS  
- **Visualization:** Seaborn, Matplotlib  
- **Frontend:** HTML, CSS (Bootstrap + custom `style.css`)  
- **Deployment:** Docker, Render/Heroku

---





---

##  Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run EDA
python analysis/eda.py

# Start Flask API
python app/api.py

