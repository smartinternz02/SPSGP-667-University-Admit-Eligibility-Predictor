import pickle
from flask import Flask , request, render_template
app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def admin():
    gre=request.form["gre"]
    tofl=request.form["tofl"]
    rating=request.form["rating"]
    sop=request.form["sop"]
    lor=request.form["lor"]
    cgpa=request.form["cgpa"]
    research=request.form["research"]
    if (research=="Yes"):
        research=1
    else:
        research=0
    xx=model.predict([[eval(gre),eval(tofl),eval(rating),eval(sop),eval(lor),eval(cgpa),research]])
    if (xx==True):
        return render_template("chance.html")
    return render_template("nochance.html")
if __name__ == '__main__':
    app.run(debug = False, port=5000)
