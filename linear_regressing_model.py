import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

"""
x_train : section of x
y_train : section of y

test_size=0.1 : we are splitting up 10% of our data into test samples
(its the information which model does not know about so it will be good for testing after training the model)

x_test & y_test: test sample data from x & y
"""

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
'''
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression() # created a model
    linear.fit(x_train, y_train) # trains the data
    acc = linear.score(x_test, y_test) # accuracy of the model how good is it able to predict
    print(acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
'''
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Co: ", linear.coef_)
print("Intercept: ", linear.intercept_)

predictions = linear.predict(x_test)

for p in range(len(predictions)):
    print(predictions[p], x_test[p], y_test[p])

p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()