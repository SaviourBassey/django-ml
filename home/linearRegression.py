#Importing all the required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

#read data from the csv file
data = pd.read_csv("home/nnnnnsave.csv")

#target variable
predict = "Technostress"

#Model
x = data.drop(columns="Technostress")
y = data["Technostress"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

best_model = 0

#training model
# for i in range(10000):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i)

#     linear = LinearRegression()

#     linear.fit(x_train, y_train)
#     accuracy = linear.score(x_test, y_test)
#     #print(accuracy)

#     if accuracy > best_model:
#         best_model = accuracy
#         #Save the model
#         with open("home/LnearRegressionModel.pickle", "wb") as f:
#             pickle.dump(linear, f)

#Open and load the trained model
pickle_in = open("home/LnearRegressionModel.pickle", "rb")
linear = pickle.load(pickle_in)

#printing the accuracy, coefficient, and intercept of the model
print("accuracy: \n", best_model)
print()
print("coefficient: \n", linear.coef_)
print()
print("Intercept: \n", linear.intercept_)

#make predictions
prediction = linear.predict(pd.DataFrame([[0,1,1,1]], columns=["Gender","Age","hours_spent","TECHNOLOGY"]))
#print(prediction)