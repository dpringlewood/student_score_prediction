"""
In the last example I used a select amount of predictive variables.
Here I will increase the amount of predictive variables and hence
the dimensionality in the data.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE


# Import the data and inspect it
maths = pd.read_csv(r'./student-mat.csv', sep=';')
print(maths.head())
print(maths.info())

"""
As before the data is in good shape. However we can see that some variables
here are down as numeric when they are actually categorical. E.G. 'Medu' 
denotes the mothers education level from 1-4 rather than a quantity or numerical
feature.

Even some of the numbers that you would naturally think should be numerical actually
seem to be categorical when you inspect the description of the data, such as studytime.
I will see if changing this to categorical makes my model more accurate.
"""

print(maths.select_dtypes('int64').columns.values)

cols_to_change = {'Medu':'category', 'Fedu':'category', 'traveltime':'category',
                  'studytime':'category', 'famrel':'category','freetime':'category',
                  'goout':'category', 'Dalc':'category', 'Walc':'category',
                  'health':'category',}

maths = maths.astype(cols_to_change)
maths = pd.get_dummies(maths, drop_first=True)
print(maths.shape)

"""
We can see now that the dataframe has 70 predictive variables after having converted our
object and categorical columns to dummy variables.

Lets preform a lasso regression analysis to filter out the best predictive variables.
We can tune the parameters using cross-validation to find the best alpha values for our
lasso regression.
"""
# Initiate the LassoCV and split the data
alphas = np.linspace(0.01, 4, 100)
lasso_cv = LassoCV(alphas=alphas, random_state=42)
X = maths.drop('G3', axis=1)
y = np.array(maths.G3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the data to the model
# See how well the model preforms
lasso_cv.fit(X_train, y_train)
y_pred = lasso_cv.predict(X_test)
rmse = (MSE(y_test, y_pred))**(1/2)
print(lasso_cv.alpha_)
print(lasso_cv.score(X_test, y_test))
print(rmse)

"""
One benefit of Lasso regression is that it shrinks coefficients that are not relevant
for our model. We can use this to show us exactly what is contributing to our output
variable.
"""
coeff = {}

for i in range(len(X.columns.values)):
    coeff[X.columns.values[i]] = lasso_cv.coef_[i]

coeff_df = pd.DataFrame.from_dict(coeff, orient='index', columns=['Coefficients'])
coeff_df = coeff_df[coeff_df['Coefficients'] != 0].reset_index()

# set-up our plot.
sns.set_style('dark')
sns.barplot(x='index', y='Coefficients', data=coeff_df)
plt.xticks(rotation=45)
plt.title('Model Coefficients')
plt.tight_layout()
plt.show()




