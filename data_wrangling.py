##------------- All Libraries in use should be read while going through the code to have an understanding of what is happening.-------------##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#------Import Dataset-------#

dataset = pd.read_csv('Data.csv')

#print(dataset)

# Matrix of Features

X = dataset.iloc[:,:-1].values  # extracting as 2-D list the independent variables : = all rows, :-1 = all cols but last.

#print(X)

Y = dataset.iloc[:, 3].values   # extracting as list the dependent variables : = all rows, 3 = 3rd col only.

#print(Y)

# Taking care of  missing values
import scipy
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) # Check Imputer documentation for the variables used, basically it is to replace missig values with mean of the col.
imputer = imputer.fit(X[:, 1:3]) 
X[:, 1:3] = imputer.transform(X[:, 1:3])                               # replacing the NaN with imputer values.

#print(X)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])              # Encoding data to numerical values but this does not take care of the ordinality problem. 
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()                 # Converting categorical variables to binary multicolumns.

#print(X)

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#print(Y)

#Splitting Data in Traning and Test sets
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#print(X_train, X_test, Y_train, Y_test)

#Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()                # To scale numerical values to compareable levels(Normalization) in accordance with Euclidean distance b/w points to speed up ML algo's where ED applies.
X_train = sc_X.fit_transform(X_train)  
X_test = sc_X.transform(X_test)        # We dont have to fit again as it has already been done in train set.

print(X_train)
print(X_test)
""" NOTE: Dummy variables like binary values of country(Encoding categorical data) may or may not be NORMALIZED depending on the senario.
	Standardization = [(x-mean(x))/sd(x)]
	Normalization = [(x-min(x))/(max(x) - min(x))]"""


