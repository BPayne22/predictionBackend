from flask import Flask, request, jsonify
import predictionModelV3 as model_logic  

app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    player_name = "freddie_freeman"
    selected_stat = "H"
    opponent = request.args.get("opponent", "DET")

    # Run prediction pipeline
    df = model_logic.fetch_and_clean_player_data(player_name, selected_stat)
    features, targets, target_stats = model_logic.prepare_features_and_targets(df, selected_stat)
    features = model_logic.add_user_input_opponent(features, opponent)

    model, _ = model_logic.train_model(features, targets)
    prediction = model.predict(features.iloc[[-1]])[0]

    return jsonify({stat: float(val) for stat, val in zip(target_stats, prediction)})

if __name__ == "__main__":
    app.run(debug=True)