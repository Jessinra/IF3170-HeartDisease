from flask import Flask, render_template, request, jsonify
from numpy import NAN
from classifier import HeartDiseaseClassifier as hdc

app = Flask(__name__)

clf = hdc()

def execute_classify(features):
    features_nanified = [x if x is not None else NAN for x in features]
    output_i64, = clf.predict(features_nanified)
    output = int(output_i64)
    if output == 0:
        label = 'Absent'
    else:
        label = 'Present ({})'.format(output)
    return output, label

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
    import time
    time.sleep(2)
    features = []
    for attr in attributes:
        features.append(request.form.get(attr, default=None, type=float))
    output, label = execute_classify(features)
    features_pretty = {x: y for (x, y) in zip(attributes, features)}
    if request.accept_mimetypes.accept_html:
        return render_template('result.html', result=label)
    else:
        return jsonify({"features": features_pretty, "output": output, "label": label})
