import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold
import os
import re

# Retrieve the firebase key path from environment variables
firebase_key_path = os.environ.get("FIREBASE_KEY_PATH")
if not firebase_key_path:
    raise ValueError(
        "No Firebase key path found. Please set FIREBASE_KEY_PATH in your environment or .env file.")

# ===  Firebase Setup ===
cred = credentials.Certificate(firebase_key_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# === Lagged Features ===


def add_lagged_features(df, stats, lags=[1, 3, 10],
                        lag_weights={1: 1.2, 3: 1.1, 10: 1.3},
                        rolling_weight=1.3,
                        rolling10_weight=1.1): 
    df = df.copy()

    for stat in stats:
        if stat not in df.columns:
            print(f"Warning: '{stat}' not found in DataFrame. Skipping.")
            continue

        df[stat] = pd.to_numeric(df[stat], errors='coerce')

        for lag in lags:
            column_name = f'{stat}_lag{lag}'
            df[column_name] = df[stat].shift(lag)
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
            df[column_name] *= lag_weights.get(lag, 1)

        # Rolling average over last 3 games
        rolling3_name = f'{stat}_rolling3'
        df[rolling3_name] = df[stat].rolling(3).mean().shift(1)
        df[rolling3_name] = pd.to_numeric(df[rolling3_name], errors='coerce') * rolling_weight

        # Rolling average over last 10 games
        rolling10_name = f'{stat}_rolling10'
        df[rolling10_name] = df[stat].rolling(10).mean().shift(1)
        weight = rolling10_weight if rolling10_weight is not None else rolling_weight
        df[rolling10_name] = pd.to_numeric(df[rolling10_name], errors='coerce') * weight

    return df

# ===  Load and Clean Player Data ===
def fetch_and_clean_player_data(player_name, selected_stat):
    # Avoid pulling all 100,000+ games
    docs = (
        db.collection("gameStats")
          .where("player", "==", player_name)
          .stream()
    )

    cleaned_data = []

    for doc in docs:
        doc_id = doc.id
        if "postseason" in doc_id.lower():
            continue  # Skip postseason games

        raw = doc.to_dict()
        cleaned = {}

        for key, val in raw.items():
            val = str(val).strip()
            if re.match(r'^-?\d+\.?\d*%$', val):
                cleaned[key] = float(val.replace('%', '')) / 100
            elif re.match(r'^-?\d+\.?\d*$', val):
                cleaned[key] = float(val)
            else:
                cleaned[key] = val

        cleaned_data.append(cleaned)

    if not cleaned_data:
        raise ValueError(f"No data found for player: {player_name}")

    df = pd.DataFrame(cleaned_data).fillna(0)

    # Drop games with zero AB
    if "AB" in df.columns:
        df = df[df["AB"] != 0].reset_index(drop=True)

    # Normalize column names
    pos_col = [col for col in df.columns if "Pos" in col]
    if pos_col:
        df.rename(columns={pos_col[0]: "Pos"}, inplace=True)
        df["Pos"] = df["Pos"].str.strip().replace({'\r': '', '\n': ''}, regex=True)

    # Normalize Opponent column and enforce consistent casing
    if "Opp" in df.columns:
        df["Opp"] = df["Opp"].astype(str).str.strip().str.upper().replace(
            {'\r': '', '\n': '', ' ': ''}, regex=True)
        df["Opp"].fillna("UNKNOWN", inplace=True)

    # Confirm opponent distribution for debugging
    #print("\nOpponent frequency preview:")
    #print(df["Opp"].value_counts())
    #print("All opponents:", df["Opp"].unique())

    # Convert key stats to numeric
    numeric_cols = ['OBP', 'OPS', 'SLG', 'BA']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Add lagged and rolling features if stat exists
    if selected_stat in df.columns:
        df = add_lagged_features(df, stats=[selected_stat], lags=[1, 3, 10])
    else:
        print(f"Warning: selected stat '{selected_stat}' not found in data.")

    # Drop rows missing required lagged values
    df = df.dropna().reset_index(drop=True)

    print(f"Loaded {len(df)} games for {player_name}")
    return df

# ===  Define Features and Targets ===
def prepare_features_and_targets(df, selected_stat):
    target_stats = [selected_stat]

    # Columns to drop that are metadata or too noisy
    drop_cols = [
        'Date','Result','Team','DFS(DK)','DFS(FD)','WPA','cWPA','aLI','acLI',
        'RE24','Rk','Inngs','Pos','Gtm','Gcar','@/H','BOP','IBB','HBP','SH','SF','CS','player'
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Bin sparse stats to improve signal
    rare_stats_to_bin = ['HR', '3B', 'SB', 'BB']
    for stat in [s for s in rare_stats_to_bin if s != selected_stat]:
        if stat in df.columns:
            df[stat] = pd.to_numeric(df[stat], errors='coerce')
            df[f'{stat}_bin'] = pd.cut(
                df[stat],
                bins=[-1, 0, 1, 3, 5, 100],
                labels=['0', '1', '2-3', '4-5', '6+']
            )
            df.drop(columns=[stat], inplace=True)

    # Ensure clean numeric conversion
    df = df.apply(pd.to_numeric, errors='ignore')
    features = df.copy()

    # Detect and encode categorical features
    categorical_cols = [
        col for col in features.columns
        if features[col].dtype == 'object' and col not in target_stats
    ]
    categorical = pd.get_dummies(features[categorical_cols], drop_first=True)

    # Separate opponent flags to protect them from filtering
    opp_cols = [col for col in categorical.columns if col.startswith("Opp_")]
    other_cols = [col for col in categorical.columns if not col.startswith("Opp_")]

    selector = VarianceThreshold(threshold=0.001)
    filtered = selector.fit_transform(categorical[other_cols])
    filtered_df = pd.DataFrame(filtered, columns=[
        col for col, keep in zip(other_cols, selector.get_support()) if keep
    ])

    # Recombine all categorical flags safely
    categorical = pd.concat([filtered_df, categorical[opp_cols]], axis=1)

    # Grab numeric columns and combine
    numeric = features.drop(columns=categorical_cols)
    numeric = numeric.apply(pd.to_numeric, errors='coerce')
    features = pd.concat([categorical, numeric], axis=1).fillna(0)

    # Final target and input separation
    targets = df[target_stats].apply(pd.to_numeric, errors='coerce').fillna(0)
    features = features.drop(columns=target_stats, errors='ignore')

    return features, targets, target_stats


# === Select Opponent ===
def add_user_input_opponent(features, user_opponent):
    # Identify the last row (used for prediction)
    pred_idx = features.index[-1]

    # Normalize opponent formatting
    user_opponent = user_opponent.strip().upper()
    opp_col = f'Opp_{user_opponent}'

    # Reset all Opponent flags for prediction row
    for col in features.columns:
        if col.startswith('Opp_'):
            features.at[pred_idx, col] = False

    # If the selected opponent flag exists, set it to True
    if opp_col in features.columns:
        features.at[pred_idx, opp_col] = True
    else:
        print(f"Inserted missing opponent flag: {opp_col}")
        # Add the missing column with 0s, set last row to True
        features[opp_col] = 0
        features.at[pred_idx, opp_col] = True

    return features

# === Weighted Opponent Stat ===


def add_opponent_lagged_stats(df, stat, opponent):
    # Normalize opponent column for consistency
    if "Opp" not in df.columns:
        raise ValueError("Opponent column 'Opp' not found in DataFrame.")

    df["Opp"] = df["Opp"].str.strip().str.upper()
    opponent = opponent.strip().upper()

    # Create mask for games vs selected opponent
    mask = df["Opp"] == opponent
    opp_indices = df[mask].index

    # Validate that the target stat exists
    if stat not in df.columns:
        raise ValueError(f"Target stat '{stat}' not found in DataFrame.")

    # Create lagged stats for matchups against this opponent
    for lag in [1, 3, 5]:
        lag_col = f"{stat}_vs_{opponent}_lag{lag}"
        df[lag_col] = np.nan

        shifted_vals = df.loc[mask, stat].shift(lag)
        df.loc[opp_indices, lag_col] = shifted_vals.values

    return df


# ===  Train Model ===
def train_model(features, targets, target_stat):
    # Drop target column from feature matrix if it snuck in
    features = features.drop(columns=[target_stat], errors='ignore')

    # Emphasize recent games with linear weights
    num_rows = len(features)
    weights = np.linspace(1, 3, num=num_rows)
    weights = np.asarray(weights).flatten()

    # Force all features to float for XGBoost (bool → 0.0 or 1.0)
    features = features.apply(pd.to_numeric, errors='coerce').astype(float).fillna(0)

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.07,
        n_estimators=350,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.7
    )

    model.fit(features, targets[target_stat], sample_weight=weights)
    return model, weights

# ===  Walk-Forward Validation Follows more real world predictions ===


def walk_forward_validation(features, targets, weights, window=10, val_window=5):
    r2_scores = []

    for i in range(window, len(features) - val_window):
        X_train = features.iloc[:i]
        y_train = targets.iloc[:i]
        w_train = weights[:i]

        X_val = features.iloc[i:i + val_window]
        y_val = targets.iloc[i:i + val_window]

        model = MultiOutputRegressor(
            xgb.XGBRegressor(objective='reg:squarederror'))
        model.fit(X_train, y_train, sample_weight=w_train)

        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred, multioutput='uniform_average')
        r2_scores.append(r2)

    print("\nWalk-Forward R2 Scores (windowed):", r2_scores)
    print("Average R2:", np.mean(r2_scores))
# ===  Save Models ===


def save_models(model, target_stats):
    os.makedirs("xgb_json_models_v3", exist_ok=True)
    for i, est in enumerate(model.estimators_):
        est.get_booster().save_model(
            f"xgb_json_models_v3/{target_stats[i]}.json")

# ===  Predict Next Game ===


def predict_next_game(model, features, target_stats):
    next_game_input = features.drop(columns=target_stats, errors='ignore').iloc[[-1]]
    print("\nActive Features Used for Prediction:")
    for col, val in next_game_input.iloc[0].items():
        if val != 0:
            print(f"{col:<25} {val}")
        prediction = model.predict(next_game_input)[0]

    print("\nPredicted Next Game Stats:")
    print(f"{'Stat':<8} {'Prediction':>10}")
    print('-' * 20)
    for stat, val in zip(target_stats, prediction):
        print(f"{stat:<8} {val:>10.3f}")

# === Pulls data from HTML ===
def get_prediction(player_name, selected_stat, opponent):
    from dotenv import load_dotenv
    import os
    import firebase_admin
    from firebase_admin import credentials

    # Load environment variables
    load_dotenv()
    key_path = os.environ.get("FIREBASE_KEY_PATH")
    if not key_path:
        raise ValueError("FIREBASE_KEY_PATH environment variable is not set.")

    # Initialize Firebase
    if not firebase_admin._apps:
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)

    # Step 1: Load and clean data
    df = fetch_and_clean_player_data(player_name, selected_stat)

    # Step 2: Add opponent-specific lagged stats
    df = add_opponent_lagged_stats(df, selected_stat, opponent)

    # Step 3: Add lagged and rolling features
    df = add_lagged_features(df, stats=[selected_stat], lags=[1, 3, 10])

    # Step 4: Build final feature matrix
    features, targets, target_stats = prepare_features_and_targets(df, selected_stat)

    # Step 5: Insert opponent flag AFTER dummy encoding
    features = add_user_input_opponent(features, opponent)

    # Step 6: Coerce all data to float for model safety
    features = features.astype(float)

    # Step 7: Train and predict
    model, _ = train_model(features, targets, selected_stat)
    next_input = features.iloc[[-1]].copy()
    prediction = model.predict(next_input)[0]

    return target_stats, prediction


# ===  Main Execution ===
if __name__ == "__main__":
    # Inputs from frontend or config
    player_name = "shohei_ohtani"
    selected_stat = "TB"
    opponent = "SFG"

    # Step 1: Load and preprocess data
    df = fetch_and_clean_player_data(player_name, selected_stat)
    features, targets, target_stats = prepare_features_and_targets(df, selected_stat)

    # Step 2: Add matchup-specific context
    features = add_user_input_opponent(features, opponent)
    features = add_opponent_lagged_stats(features, selected_stat, opponent)

    # Step 3: Train model
    model, weights = train_model(features, targets, selected_stat)

    # Step 4: Save model artifacts (optional for deployment)
    save_models(model, target_stats)

    # Step 5: Audit data and input row
    print("Audit the last 5 Rows\n----------")
    print(df[['Date', 'TB', 'H', 'R', 'AB', 'BA', 'HR', selected_stat]].tail())
    print("\nLatest input used for prediction:\n", features.iloc[[-1]].T)

    # Step 6: Feature importance
    selected_model = model.estimators_[target_stats.index(selected_stat)]
    importances = selected_model.feature_importances_
    min_len = min(len(importances), len(features.columns))

    important_features = pd.Series(
        importances[:min_len], index=features.columns[:min_len]
    ).sort_values(ascending=False)

    # Step 7: Filter irrelevant opponent flags (only active one stays)
    active_opponent = [col for col in features.columns if col.startswith('Opp_') and features.iloc[-1][col]]
    opponent_cols = [col for col in features.columns if col.startswith('Opp_')]
    irrelevant_opponents = [col for col in opponent_cols if col not in active_opponent]

    important_filtered = important_features.drop(index=irrelevant_opponents).sort_values(ascending=False)

    # Step 8: Generate prediction
    predict_next_game(model, features, target_stats)

    # Step 9: Evaluation and snapshots
    print(f"\nFiltered Top Features for {selected_stat} Prediction:\n", important_filtered.head(10))
    print(f"\nAverage {selected_stat} in training data: {df[selected_stat].mean()}")
    print(f"{selected_stat} distribution:\n", df[selected_stat].value_counts().sort_index())
    print(f"\nActual {selected_stat} from last game: {df.iloc[-1][selected_stat]}")

    stat_cols = [c for c in features.columns if selected_stat in c]
    print(f"{selected_stat.upper()} Feature Snapshot Last 5 Entries")
    print(features[stat_cols].tail().to_string(index=False))

    print("\nFinal prediction input row:")
    print(features.iloc[-1][features.iloc[-1] != 0].sort_values(ascending=False))
