import os
from flask import Flask, render_template, request, jsonify
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()
from search_service import vector_patient_search, chat_from_query_using_rag

IRIS_HOST = os.getenv("IRIS_HOST", "iris4health")
IRIS_PORT = int(os.getenv("IRIS_PORT", "1972"))
IRIS_NS   = os.getenv("IRIS_NAMESPACE", "DEMO")
IRIS_USER = os.getenv("IRIS_USERNAME", "_SYSTEM")
IRIS_PWD  = os.getenv("IRIS_PASSWORD", "ISCDEMO")
SCHEMA = "SQL1"
TABLE  = "patient_info"
VECTOR_COL = "patient_vector"
EMB_MODEL = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")

app = Flask(__name__)
engine = create_engine(f"iris://{IRIS_USER}:{IRIS_PWD}@{IRIS_HOST}:{IRIS_PORT}/{IRIS_NS}", pool_pre_ping=True)
model = SentenceTransformer(EMB_MODEL)

DISPLAY_COLUMNS = [
    "Name", "City", "DOB", "Allergies", "FamilyHistory", "Medication",
    "PostalCode", "State", "Street"
]

@app.route("/", methods=["GET", "POST"])
def index():
    query_text = ""
    results = []
    error = None

    if request.method == "POST":
        query_text = request.form.get("query", "").strip()
        try:
            results = vector_patient_search(
                engine=engine,
                model=model,
                query_text=query_text,
                schema=SCHEMA,
                table=TABLE,
                vector_col=VECTOR_COL,
                display_columns=DISPLAY_COLUMNS,
                top_k=5,
            )
            if not results:
                error = "No results found."
        except Exception as e:
            error = f"Search failed: {e}"

    return render_template(
        "home.html",
        query_text=query_text,
        results=results,
        columns=DISPLAY_COLUMNS + ["score"],
        error=error
    )


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify({"reply": "Please ask a question."})
    try:
        out = chat_from_query_using_rag(
            engine=engine,
            model=model,
            user_question=user_msg,
            schema=SCHEMA,
            table=TABLE,
            vector_col=VECTOR_COL,
            display_columns=DISPLAY_COLUMNS,
            top_k=5,
            cutoff=True,
        )
        return jsonify({"reply": out["answer"]})
    except Exception as e:
        return jsonify({"reply": f"Error: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
