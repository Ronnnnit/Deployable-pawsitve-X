from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the model and other artifacts
model = load_model('disease_prediction_model.keras')

# Load the scaler and label encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('target_encoder.pkl', 'rb') as f:
    target_encoder = pickle.load(f)

# Function to preprocess user input
def preprocess_input(user_input):
    input_df = pd.DataFrame([user_input])
    
    # Apply label encoding to categorical columns
    categorical_columns = input_df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        if column in label_encoders:
            le = label_encoders[column]
            try:
                input_df[column] = le.transform(input_df[column])
            except ValueError:
                input_df[column] = le.transform([le.classes_[0]])  # Handle unseen labels
    
    # Scale numeric columns
    numeric_columns = input_df.select_dtypes(include=[np.number]).columns
    input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])
    
    return input_df

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user input from the form
        user_input = {
            "Animal_Type": request.form['Animal_Type'],
            "Breed": request.form['Breed'],
            "Age": float(request.form['Age']),
            "Gender": request.form['Gender'],
            "Weight": float(request.form['Weight']),
            "Symptom_1": request.form['Symptom_1'],
            "Symptom_2": request.form['Symptom_2'],
            "Symptom_3": request.form['Symptom_3'],
            "Symptom_4": request.form['Symptom_4'],
            "Duration": request.form['Duration'],
            "Appetite_Loss": request.form['Appetite_Loss'],
            "Vomiting": request.form['Vomiting'],
            "Diarrhea": request.form['Diarrhea'],
            "Coughing": request.form['Coughing'],
            "Labored_Breathing": request.form['Labored_Breathing'],
            "Lameness": request.form['Lameness'],
            "Skin_Lesions": request.form['Skin_Lesions'],
            "Nasal_Discharge": request.form['Nasal_Discharge'],
            "Eye_Discharge": request.form['Eye_Discharge'],
            "Body_Temperature": float(request.form['Body_Temperature']),
            "Heart_Rate": int(request.form['Heart_Rate']),
        }

        # Preprocess user input
        input_df = preprocess_input(user_input)
        
        # Predict disease using the trained model
        prediction = model.predict(input_df)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_disease = target_encoder.inverse_transform(predicted_class)
        
        # Return JSON response
        return jsonify({"disease": predicted_disease[0]})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
