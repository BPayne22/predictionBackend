from flask import Flask, request, jsonify
import predictionModelV3 as model_logic
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=[
    "http://127.0.0.1:5500",           # Local dev
    "http://localhost:5500",           # Alt local dev
    "https://your-github-username.github.io"  # Your GitHub Pages domain
])


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    player = data.get("player")
    stat = data.get("stat")
    opponent = data.get("opponent")

    if not all([player, stat, opponent]):
        return jsonify({"error": "Missing input fields"}), 400

    try:
        target_stats, prediction = model_logic.get_prediction(player, stat, opponent)
        return jsonify({stat: float(val) for stat, val in zip(target_stats, prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)
