from flask import Flask, render_template, request, jsonify, redirect
from werkzeug import secure_filename
import os
import logging
from werkzeug.wrappers import response
import json
from CNN import load_mat
from CNN import custom_train
import os
from CNN import cnn
from CNN import convert_mat
import h5py
from scipy.io import loadmat

app = Flask(__name__)
os.makedirs(os.path.join(app.instance_path, 'Data_Training'), exist_ok=True)
os.makedirs(os.path.join(app.instance_path, 'Img_Predict'), exist_ok=True)
os.makedirs(os.path.join(app.instance_path, 'model'), exist_ok=True)

@app.route("/", methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        #get every form input
        iterasi = request.form['Iterasi']
        akurasi = request.form['Akurasi']
        strength = request.form['Strength']
        nama = request.form['SaveAs']
        path_train_mat = request.files['Latih']
        path_test_mat = request.files['Test']

        #load mat file
        load_mat.main(path_train_mat,path_test_mat,float(strength))
        print('done')
        path_svhn = 'SVHN_grey.h5'

        #define unique model folder
        counter = len(os.listdir('instance/model/'))
        counter += 1
        model_name = str(counter)+nama
        #train the model using existing h5 file
        custom_train.main(path_svhn, int(iterasi),float(akurasi),model_name)
        path_train = model_name
        return jsonify({'train':path_train})
    else :
        return render_template('UI_UAS.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if(request.method == 'POST'):
        model = str(request.form['model'])
        image = request.files['img']
        path = os.path.join('instance/Img_Predict/', image.filename)
        image.save(path)
        convert_mat.img2mat(path)
        mat = loadmat('predict.mat')
        newMat = load_mat.preprocess(mat,float(1))
        f = h5py.File('SVHN_grey.h5', 'w')
        f.create_dataset('X_train', data=newMat["X"])
        f.close()
        if(model == 'default'):
            data = cnn.main('CNN/test/svhnet')
        else:
            data = cnn.main('instance/model/'+model+'/svhnet')
        accuracy = str(max(data['probabilities'])*100)+'%'
        detected = str(data['classes'])
        return jsonify({'class':detected,'accuracy':accuracy})
    else:
        model = os.listdir('instance/model/')
        return render_template('prediksi.html',model=model)

@app.route('/help')
def help():
    return render_template('Help.html')

if __name__ == "__main__":
    app.run(debug=True)