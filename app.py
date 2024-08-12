# Importing required libraries
from flask import Flask, request, jsonify
import numpy as np
import pickle
import warnings
from feature import FeatureExtraction

warnings.filterwarnings('ignore')

# Load the pre-trained model
file = open("pickle/model1.pkl", "rb")
gbc = pickle.load(file)
file.close()

# Initialize Flask app
app = Flask(__name__)

# Define the phishing detection endpoint
@app.route("/check-phishing", methods=["POST"])
def check_phishing():
    data = request.get_json()

    if 'url' not in data:
        return jsonify({"error": "URL not provided"}), 400
    
    url = data["url"]

    # Feature extraction
    obj = FeatureExtraction(url)
    x = np.array(obj.getFeaturesList()).reshape(1, 30)

    # Model prediction
    y_pred = gbc.predict(x)[0]
    y_pro_phishing = gbc.predict_proba(x)[0, 0]
    y_pro_non_phishing = gbc.predict_proba(x)[0, 1]

    # Prepare response
    result = {
        "is_phishing": bool(y_pred == -1),
        "confidence": round(y_pro_non_phishing if y_pred == 1 else y_pro_phishing, 2)
    }

    return jsonify(result), 200

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
