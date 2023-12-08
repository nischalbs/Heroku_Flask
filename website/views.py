from flask import Blueprint,render_template,request,redirect,url_for,flash
from flask_login import login_required,current_user
import numpy as np
import pickle

views = Blueprint('views',__name__)

@views.route('/', methods=['GET','POST'])
@login_required
def home():
    if request.method == 'POST':
        data1 = request.form['a']
        data2 = request.form['b']
        data3 = request.form['c']
        data4 = request.form['d']
        data5 = request.form['e']
        data6 = request.form['f']
        data7 = request.form['g']
        data8 = request.form['h']
        data9 = request.form['i']
        data10 = request.form['j']
        data11 = request.form['k']
        data12 = request.form['l']
        arr = np.array([[data1, data2, data3, data4,data5,data6,data7,data8,data9,data10,data11,data12]])
        model = pickle.load(open('website\churn.pkl', 'rb'))
        try:
            pred = model.predict(arr)
            return render_template("after.html",user=current_user, data=pred)
        except ValueError:
            flash('Fill all Fields Before Submitting.', category='error')
    return render_template("home.html", user=current_user)
           
            
    



