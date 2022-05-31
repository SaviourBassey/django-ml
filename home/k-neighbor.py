#Importing all the required libraries
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import pickle

#read data from the csv file
data = pd.read_csv("nnnnnsave.csv")

#target variable
predict = ""

#Model
x = np.array(data.drop(predict, axis=1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

best_model = 0

#training model
# for i in range(10000):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i)

#     model = KNeighborsClassifier(n_neighbors=9)

#     model.fit(x_train, y_train)
#     accuracy = model.score(x_test, y_test)
#     print(accuracy)

#     if accuracy > best_model:
#         best = accuracy
#         #Save the model
#         with open("home/KNeighborsModel.pickle", "wb") as f:
#             pickle.dump(model, f)

#Open and load the trained model
pickle_in = open("home/KNeighborsModel.pickle", "rb")
model = pickle.load(pickle_in)

#make predictions
predictions = model.predict(pd.DataFrame([[0,1,1,1]], columns=["Gender","Age","hours_spent","TECHNOLOGY"]))
print(predictions)