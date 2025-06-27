from flask import Flask, render_template, request
import numpy as np
import pickle
import sklearn
print(sklearn.__version__)

app = Flask(__name__)
model = pickle.load(open(r'C:\Users\Pavan\OneDrive\Documents\Desktop\Stroke-Prediction-Project\Stroke.pkl', 'rb'))
@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            gender = request.form['gender']
            gender_Male = 1 if gender == 'gender_Male' else 0

            age = float(request.form['age'])
            hypertension = int(request.form['hypertension'])
            heart_disease = int(request.form['heart_disease'])
            ever_married = int(request.form['ever_married'])
            Residence_type = int(request.form['Residence_type'])
            avg_glucose_level = float(request.form['avg_glucose_level'])
            bmi = float(request.form['bmi'])

            work_type = request.form['work_type']
            work_type_Never_worked = 1 if work_type == 'Never_worked' else 0
            work_type_Private = 1 if work_type == 'Private' else 0
            work_type_Self_employed = 1 if work_type == 'Self_employed' else 0
            work_type_children = 1 if work_type == 'children' else 0
            work_type_Govt_job = 1 if work_type == 'Govt_job' else 0

            smoking_status = request.form['smoking_status']
            smoking_status_formerly_smoked = 1 if smoking_status == 'formerly_smoked' else 0
            smoking_status_never_smoked = 1 if smoking_status == 'never_smoked' else 0
            smoking_status_Smokes = 1 if smoking_status == 'Smokes' else 0
            # Remove Unknown status as it's the reference category

            values = np.array([[gender_Male, age, hypertension, heart_disease, ever_married,
                                Residence_type, avg_glucose_level, bmi,
                                work_type_Never_worked, work_type_Private, work_type_Self_employed, work_type_children,
                                smoking_status_formerly_smoked, smoking_status_never_smoked, smoking_status_Smokes]])
            prediction = model.predict(values)

            return render_template('result.html', prediction=prediction[0])
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)

