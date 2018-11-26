from flask import Flask, render_template, request, jsonify
from numpy import NAN
from classifier import HeartDiseaseClassifier as hdc

app = Flask(__name__)

clf = hdc()

@app.route('/')
def index():
    return render_template('index.html')

attributes = ['age', 'sex', 'chestPain', 'bloodPressure', 'serumCholesterol',
              'bloodSugar', 'restingECG', 'maxHeartRate',
              'exerciseInduceAngina', 'stDepression', 'peakExerciseST',
              'vesselsColored', 'thal']

@app.route('/check', methods=['POST'])
def check():
    global clf
    features = []
    for attr in attributes:
        features.append(request.form.get(attr, default=NAN, type=float))
    output_i64, = clf.predict(features)
    output = int(output_i64)
    features_denanified = [x if x is not NAN else None for x in features]
    features_pretty = {x: y for (x, y) in zip(attributes, features_denanified)}
    return jsonify({"features": features_pretty, "output": output})
