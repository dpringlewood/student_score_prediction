"""
In the last example I used a select amount of predictive variables.
Here I will increase the amount of predictive variables and hence
the dimensionality in the data.
"""

import pandas as pd
from sklearn.linear_model import Lasso


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

Lets preform a lasso regression analysis to filter out the best predictive variables
"""