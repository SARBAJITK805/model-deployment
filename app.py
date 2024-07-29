from flask import Flask, render_template,request,jsonify
from utils import make_predictions
import pickle

cv=pickle.load(open("models/cv.pkl","rb"))
clf=pickle.load(open("models/clf.pkl","rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    if request.method=="POST":
        email=request.form.get("email-content")
    predictions= make_predictions(email)

    return render_template("index.html",predictions=predictions,email=email)

@app.route("/api/predict",methods=["POST"])
def api_predict():
    data=request.get_json(force=True)
    email=data["content"]
    predictions= make_predictions(email)
    return jsonify({prediction:prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
