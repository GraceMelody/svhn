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

app = Flask(__name__)
os.makedirs(os.path.join(app.instance_path, 'Data_Training'), exist_ok=True)
os.makedirs(os.path.join(app.instance_path, 'Img_Predict'), exist_ok=True)

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
        load_mat.main(path_train_mat,path_test_mat,int(strength))
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
        if(model == 'default'):
            print('CNN/test/svhnet')
            # data = cnn.main(image,'CNN/test/svhnet')
        else:
            print('instance/model/'+model+'/svhnet')
            # data = cnn.main(image,'instance/model/'+model+'/svhnet')
        return jsonify(test='test')
        # return jsonify(detect=data[0],accuracy=data[1])
    else:
        model = os.listdir('instance/model/')
        return render_template('prediksi.html',model=model)

if __name__ == "__main__":
    app.run(debug=True)