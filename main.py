from flask import Flask, render_template, request, jsonify, redirect
from werkzeug import secure_filename
import os
import logging
from werkzeug.wrappers import response
import json
from CNN import load_mat
from CNN import custom_train
# from CNN import cnn

app = Flask(__name__)
os.makedirs(os.path.join(app.instance_path, 'Data_Training'), exist_ok=True)
os.makedirs(os.path.join(app.instance_path, 'Img_Predict'), exist_ok=True)

@app.route("/", methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        iterasi = request.form['Iterasi']
        akurasi = request.form['Akurasi']
        strength = request.form['Strength']
        file = request.files['Latih']
        # filename = secure_filename(file.filename)
        load_mat.main(file,strength)
        print('done')
        path_svhn = 'CNN/SVHN_grey.h5'
        custom_train.main(path_svhn, int(iterasi),float(akurasi))
        # file.save(os.path.join(app.instance_path, 'Data_Training', secure_filename(file.filename)))
        print("Iterasi :"+iterasi)
        path_train = 'train_eval/svhnet/checkpoint'
        return jsonify({'train':path_train})
    else :
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

@app.route('/latih/training', methods=['POST', 'GET'])
def uploadData():
    if request.method == 'POST':
            iterasi = request.form['Iterasi']
            akurasi = request.form['Akurasi']
            strength = request.form['Strength']
            file = request.files['Latih']
            # filename = secure_filename(file.filename)
            load_mat.main(file,strength)
            print('done')
            path_svhn = 'CNN/SVHN_grey.h5'
            custom_train.main(path_svhn, int(iterasi),float(akurasi))
            # file.save(os.path.join(app.instance_path, 'Data_Training', secure_filename(file.filename)))
            print("Iterasi :"+iterasi)
            path_train = 'train_eval/svhnet/checkpoint'
            return jsonify(train=path_train)

if __name__ == "__main__":
    app.run(debug=True)