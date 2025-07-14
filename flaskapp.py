from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import predictionModelV3 as model_logic
import os

app = Flask(__name__)
CORS(app, origins=[
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://rylandbangerter.github.io",
    "https://686e981c9930ce00086f44c9--merry-gnome-9ee3d2.netlify.app",
    "https://686e981c9930ce00086f44c9--merry-gnome-9ee3d2.netlify.app/mbgpt/"
], supports_credentials=True)

@app.route("/predict", methods=["OPTIONS", "POST"])
def predict():
    if request.method == "OPTIONS":
        # Return a valid preflight response
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = request.headers.get("Origin")
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
