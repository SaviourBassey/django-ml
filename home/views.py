from django.shortcuts import render
import pandas as pd
import pickle
from django.views import View

# Create your views here.

class HomeView(View):
    def get(self, request, *args, **kwargs): 
        return render(request, "home/index.html")

    def post(self, request, *args, **kwargs):
        #Get user data from form
        gender = int(request.POST.get("gender"))
        age = int(request.POST.get("age"))
        hours_spent = int(request.POST.get("hours_spent"))
        tech_used = int(request.POST.get("tech_used"))

        #Open and load the trained model
        pickle_in = open("home/LnearRegressionModel.pickle", "rb")
        linear = pickle.load(pickle_in)

        #make predictions
        prediction = linear.predict(pd.DataFrame([[gender,age,hours_spent,tech_used]], columns=["Gender","Age","hours_spent","TECHNOLOGY"]))
        prediction_point = prediction[0]

        #Making classification
        if prediction_point >= 1.38:
            predict = "You are not Stressed"
        else:
            predict = "You are Stressed"
        
        #Passing the necessary information to the html file
        context = {
            "prediction_point":prediction_point,
            "predict":predict
        }
        return render(request, "home/result.html", context)