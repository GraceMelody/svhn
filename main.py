from flask import Flask, render_template, request, jsonify, redirect
from werkzeug import secure_filename
from werkzeug.wrappers import response
import json

app = Flask(__name__)
@app.route("/")
def home():
    return render_template('UI_UAS.html')

@app.route('/predict')
def predict():
    return render_template('prediksi.html')

@app.route('/predict/preprocessing', methods=['POST'])
def upload():
    file = request.files['test']
    filename = secure_filename(file.filename)
    file.save(filename)
    return '',204
    
if __name__ == "__main__":
    app.run(debug=True)