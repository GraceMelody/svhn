from flask import Flask, render_template
app = Flask(__name__)
@app.route("/")
def home():
    return render_template('UI_UAS.html')

@app.route('/predict')
def predict():
    return render_template('prediksi.html')

@app.route('/salvador')
def salvador():
    return 'hello, meme'
if __name__ == "__main__":
    app.run(debug=True)