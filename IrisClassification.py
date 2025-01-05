# Iris Flower Classification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data_path = 'Iris.csv'
df = pd.read_csv(data_path)

# Display the first few rows
print("Dataset Preview:")
print(df.head())

# Preprocessing
df = df.rename(columns={"Species": "species"})
X = df.iloc[:, 1:5]  # Selecting feature columns (SepalLengthCm, SepalWidthCm, etc.)
y = df['species']

# Encode target labels
y = y.astype('category').cat.codes

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Setosa', 'Versicolor', 'Virginica']))
