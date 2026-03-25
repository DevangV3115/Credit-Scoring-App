import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and columns
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "../ml/model.pkl")
columns_path = os.path.join(base_dir, "../ml/columns.pkl")

model = joblib.load(model_path)
expected_columns = joblib.load(columns_path)

@app.route("/")
def home():
    return "Credit Scoring API Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        # Construct DataFrame ensuring all expected columns are present
        input_data = {col: [data.get(col, None)] for col in expected_columns}
        df = pd.DataFrame(input_data)
        
        # Predict using the pipeline
        prediction = model.predict(df)
        
        # Provide both prediction and probability
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(df)[0][1])
            return jsonify({
                "prediction": int(prediction[0]),
                "probability": probability
            })
            
        return jsonify({
            "prediction": int(prediction[0])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)