"""
Beginners level Machine Learning (ML) project to build my skills.
The aim here is to predict a students final score (G3) based on a number
of factors. This will be a regression based ML project using sklearn.

The idea in this is to use a lasso regression analysis. Lasso allows
the model to normalise the factors, and shrink the factors that are not
important to 0.

This is important in a professional sense as it allows us to identify
what effects the predicted value.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# Import the data and inspect it
maths = pd.read_csv(r'./student-mat.csv', sep=';')
print(maths.head())
print(maths.info())

"""
Cleaned the data to only have integer values. I have
included all integer values in the hope that the Lasso
algorithm will filter out the ones that are not needed.
"""
maths = maths.select_dtypes('int64')
print(maths.info())

# set our prediction to G3 and split the data

predict = 'G3'

X = np.array(maths.drop(predict, axis=1))
y = np.array(maths[predict])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initialize and train the model
lasso = Lasso(alpha=0.01, normalize=True)
lasso.fit(X_train, y_train)

# Find the lasso coefficients
lasso_coef = lasso.fit(X_train, y_train).coef_
print(lasso_coef)

# Lets chart these coefficients so that we can use them
# for explanation purposes.
maths_cols = maths.columns
maths_cols = maths_cols[:-1]
plt.bar(range(len(maths_cols)), lasso_coef)
plt.xticks(range(len(maths_cols)), maths_cols.values, rotation=60)
plt.show()

"""
Predictably the results that students get in previous assessments, 
and how many failures they have had goes a long way in explaining what 
they will achieve in their final grade. What is interesting when adjusting 
the alpha in the lasso regression is some of the other factors that show. 
The quality of family relationships is a factor in how well a student will
preform.
"""
