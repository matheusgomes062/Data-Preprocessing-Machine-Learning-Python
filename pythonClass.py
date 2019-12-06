# -*- coding: utf-8 -*-
# Data preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Data.csv', encoding='latin-1')   # vai ler o csv
X = dataset.iloc[:, :-1].values     # vai pegar os valores da coluna 0 até a penultima coluna
y = dataset.iloc[:, 3].values       # vai pegar os valores da coluna 3

# Taking care of missing data
from sklearn.impute import SimpleImputer                        # completa os missing values 
imputer = SimpleImputer(missing_values=np.nan, strategy="mean") # vai completar eles utilizando a média
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])                    # vai colocar os valores transformados na própria tabela

# Enconding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder   # labelEncoder> categoriza os valores em números
labelencoder_x = LabelEncoder()
X[:, 0] = labelencoder_x.fit_transform(X[:, 0])                 # coloca os novos valores na tabela
oneHotEncoder = OneHotEncoder(categorical_features = [0])       # permite que ao inves de números sejam uma sequencia de binários (?)
X = oneHotEncoder.fit_transform(X).toarray();
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Spliting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Future Scalling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)