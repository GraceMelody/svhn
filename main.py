from flask import Flask, render_template, request, jsonify, redirect
from werkzeug import secure_filename
import os
from werkzeug.wrappers import response
import json

app = Flask(__name__)
os.makedirs(os.path.join(app.instance_path, 'Data_Training'), exist_ok=True)
os.makedirs(os.path.join(app.instance_path, 'Img_Predict'), exist_ok=True)

@app.route("/")
def home():
    return render_template('UI_UAS.html')

@app.route('/predict')
def predict():
    return render_template('prediksi.html')

@app.route('/predict/preprocessing', methods=['POST'])
def uploadImage():
    file = request.files['test']
     # filename = secure_filename(file.filename)
    file.save(os.path.join(app.instance_path, 'Img_Predict', secure_filename(file.filename)))
    return '',204

@app.route('/latih/training', methods=['POST'])
def uploadData():
     file = request.files['Latih']
     # filename = secure_filename(file.filename)
     file.save(os.path.join(app.instance_path, 'Data_Training', secure_filename(file.filename)))
     return '',204
    
if __name__ == "__main__":
    app.run(debug=True)