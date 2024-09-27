from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
# Load the heart failure prediction model
model_path = 'lgs_model.pkl'
with open('lgs_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict-heart-failure', methods=['POST'])
def predict_heart_failure():
    if request.method == 'POST':
        # Get form data
        Age = int(request.form['Age'])
        Sex = int(request.form['Sex'])
        ChestPainType = request.form['ChestPainType']
        Cholesterol = int(request.form['Cholesterol'])
        FastingBS = int(request.form['FastingBS'])
        MaxHR = int(request.form['MaxHR'])
        ExerciseAngina = int(request.form['ExerciseAngina'])
        Oldpeak = float(request.form['Oldpeak'])
        ST_Slope = request.form['ST_Slope']
        
        # Convert categorical variables to numeric (if needed)
        # Assuming ChestPainType and ST_Slope need to be encoded
        ChestPainType_dict = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}  # Example encoding
        ST_Slope_dict = {'Up': 0, 'Flat': 1, 'Down': 2}  # Example encoding
        
        ChestPainType_encoded = ChestPainType_dict.get(ChestPainType, 0)  # Default to 0 if not found
        ST_Slope_encoded = ST_Slope_dict.get(ST_Slope, 0)
        
        # Create input array for the model
        input_data = np.array([[Age, Sex, ChestPainType_encoded, Cholesterol, FastingBS, MaxHR, ExerciseAngina, Oldpeak, ST_Slope_encoded]])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Interpret the prediction result (Assume 1 = Heart Disease, 0 = No Heart Disease)
        if prediction[0] == 1:
            result = 'You are at risk of heart failure. Please consult a doctor.'
        else:
            result = 'You are not at risk of heart failure.'
        
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=8088)
