from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import predictionModelV3 as model_logic
import os

app = Flask(__name__)

# Apply flask-cors first
CORS(app, supports_credentials=True)

# Inject headers manually after every response
@app.after_request
def apply_cors_headers(response):
    origin = request.headers.get("Origin")
    allowed = [
        "https://686e981c9930ce00086f44c9--merry-gnome-9ee3d2.netlify.app",
        "https://686e981c9930ce00086f44c9--merry-gnome-9ee3d2.netlify.app/mbgpt/",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://rylandbangerter.github.io"
    ]
    if origin in allowed:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# Health check
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "Backend is awake"}), 200

# Prediction route
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        response = make_response()
        origin = request.headers.get("Origin")
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return response

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

# Entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
