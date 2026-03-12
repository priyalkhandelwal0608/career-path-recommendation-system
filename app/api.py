from flask import Flask, request, render_template
from recommender import CollaborativeRecommender
from semantic_search import SemanticSearch

app = Flask(__name__)
collab = CollaborativeRecommender()
semantic = SemanticSearch()

@app.route("/", methods=["GET","POST"])
def index():
    results = None
    query_value = ""   # default empty
    if request.method == "POST":
        query = request.form.get("query")
        query_value = query  # keep the typed value

        if query.isdigit():
            idx = int(query)
            results = collab.recommend(idx)
        else:
            results = semantic.search(query)

    return render_template("index.html", results=results, query_value=query_value)

if __name__ == "__main__":
    app.run(debug=True)