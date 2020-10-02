import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('dataset.csv')

X = df.drop(columns=['Costume'])
y = df['Costume']

model = DecisionTreeClassifier()
model.fit(X, y)
predictions = model.predict( [ [1, 70] ] )
print(predictions)
