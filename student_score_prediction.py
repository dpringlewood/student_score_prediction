"""
Beginners level Machine Learning (ML) project to build my skills.
The aim here is to predict a students final score (G3) based on a number
of factors. This will be a regression based ML project using sklearn.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Import the data and inspect it
maths = pd.read_csv(r'./student-mat.csv', sep=';')
print(maths.head())
print(maths.info())

"""
Important to note here that the data is generally very well structured.
There are no missing values and it is very complete. However sklearn
can only accept numerical inputs. In this case I will only use some of the 
numerical columns we have. I will leave the pre-processing of labeled 
data for a more in-depth project.
"""

maths = maths.select_dtypes('int64')
maths = maths[['studytime', 'failures', 'absences', 'G1', 'G2', 'G3']]
print(maths.info())

# set our prediction of a students final score (G3)
predict = 'G3'

# split-up X & y and make sure that they are np array's
# sklearn needs numpy array's as inputs
X = np.array(maths.drop(predict, axis=1))
y = np.array(maths[predict])

# split-up our current X & y variables into training
# and testing data.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.8, random_state=42
)

# Now we need to initiate our model and train it

linear = LinearRegression()
# Cross-Valadation testing

cv_scores = cross_val_score(linear, X, y, cv=5)
print('The CV R^2 are: ', cv_scores)
print('The mean R^2 from CV is: ', np.mean(cv_scores))
linear.fit(X_train, y_train)
print("The R^2 is: ", linear.score(X_test, y_test))


