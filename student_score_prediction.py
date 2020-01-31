"""
Beginners level Machine Learning (ML) project to build my skills.
The aim here is to predict a students final score (G3) based on a number
of factors. This will be a regression based ML project using sklearn.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import the data and inspect it
maths = pd.read_csv(r'./student-mat.csv', sep=';')
print(maths.head())
print(maths.info())

"""
Important to note here that the data is generally very well structured.
There are no missing values and it is very complete. However sklearn
can only accept numerical inputs. In this case I will only use what numerical
columns we have. I will leave the pre-processing of labeled data for a more
in-depth project.
"""

maths = maths.select_dtypes(include='int64')

# set our prediction of a students final score (G3)
predict = 'G3'

# split-up X & y and make sure that they are np array's
# sklearn needs numpy array's as inputs
X = np.array(maths.drop(predict, axis=1))
y = np.array(maths[predict])

# split-up our current X & y variables into training
# and testing data.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Now we need to initiate our model and train it

linear = LinearRegression()
linear.fit(X_train, y_train)
print(linear.score(X_test, y_test))
