import os
import time
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from pytorch_tabnet.tab_model import TabNetClassifier

# Load the TabNet model from Module 5
model_path = 'F:\Kuliah Nadia Semseter 7\Praktikum Semster 7\modul6\S7-M6-ML-main\model_m5_tabular_.zip'  
class_list = {
    'Pemasukkan kurang dari 50k': 0,
    'Pemasukkan lebih dari 50k': 1
}

# Create a Flask web application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('awal.html')

@app.route('/index')
def next():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            loaded_model = TabNetClassifier()
            loaded_model.load_model(model_path)
        except FileNotFoundError:
            return render_template('error.html', message="Model file not found.")

        # Mengambil data input dari form
        age = float(request.form['age'])
        edu = float(request.form['education'])
        ocu = float(request.form['occupation'])
        hours = float(request.form['hours_per_week'])
        country = float(request.form['native_country'])
        start = time.time()

        # Lakukan prediksi 
        probabilities = loaded_model.predict(np.array([[age, edu, ocu, hours, country]]))
        runtimes = round(time.time() - start, 4)
        result = probabilities.tolist()[0]
        prediction_label = list(class_list.keys())[result]

        return render_template('/predict.html', prediction=prediction_label, runtime=runtimes)

if __name__ == '_main_':
    # Jalankan aplikasinya
    app.run(debug=True)