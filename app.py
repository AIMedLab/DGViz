"""main entrance of the system
"""

# Author: Rui Li <li.8950@osu.edu>
# License: MIT
# Copyright: Rui Li
# Date: 9/10/19

from flask import Flask, render_template, request, jsonify, json
import os

# database
import config
from common.database import Database
from models.admission import Admission
from models.patient import Patient
# from models.DG_RNN import DGRNN
from models.code import inference

DGRNN = inference.DGRNN()

# DG-RNN model

app = Flask(__name__)
app.config.from_object(config)  #load configuration

@app.before_first_request
def initialize_database():
    Database.initialize()


@app.route('/find_all_admissions')
def find_all_admissions():
    result = Admission.find_all()
    return result

@app.route('/find_all_patient_vis')
def find_all_patient_vis():
    result = Patient.find_all_vis()
    return result

@app.route('/find_all_patient')
def find_all_patient():
    result = Patient.find_all_patient()
    return result

# model api
@app.route('/get_test_data')
def model_test():
    return DGRNN.get_test_data()

@app.route('/get_pred_data', methods=['POST'])
def get_pred_data():
    data_str = request.form.get('data')
    data_dic = json.loads(data_str)
    return DGRNN.predict(data_dic)

@app.route('/')
def index():
    return render_template('index.html')




if __name__ == '__main__':
    app.run(port=80)
