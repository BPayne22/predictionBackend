from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import predictionModelV3 as model_logic
import os

app = Flask(__name__)

# CORS setup — trusted frontend origins (local + production)
CORS(app, origins=[
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://rylandbangerter.github.io",
    "https://686e981c9930ce00086f44c9--merry-gnome-9ee3d2.netlify.app",
    "https://686e981c9930ce00086f44c9--merry-gnome-9ee3d2.netlify.app/mbgpt/"
], supports_credentials=True)

# Global CORS headers injection (Render-safe and future-proof)
@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    allowed_origins = [
        "https://686e981c9930ce00086f44c9--merry-gnome-9ee3d2.netlify.app",
        "https://686e981c9930ce00086f44c9--merry-gnome-9ee3d2.netlify.app/mbgpt/",
        "https://rylandbangerter.github.io",
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ]

    if origin in allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# Wake-up endpoint to ping backend
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "Backend is awake"}), 200

# Prediction endpoint
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        # Preflight CORS response
        response = make_response()
        origin = request.headers.get("Origin")
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

    # Actual prediction request
    data = request.get_json()
    player = data.get("player")
    stat = data.get("stat")
    opponent = data.get("opponent")

    if not all([player, stat, opponent]):
        return jsonify({"error": "Missing input fields"}), 400

    try:
        target_stats, prediction = model_logic.get_prediction(player, stat, opponent)
        return jsonify({target_stats[0]: f"{float(prediction):.3f}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Deployment entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
