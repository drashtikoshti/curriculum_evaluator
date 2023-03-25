import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest

df = pd.read_csv('S:/University Of Windsor/Term 2/ADT/Project/Submission/curriculum_evaluator/Dataset.csv')

# Removing the missing values rows
df = df.dropna()

# print(df.describe())

# Training the model
# anomaly_inputs = ['CLO1', 'CLO2', 'CLO3', 'CLO4', 'CLO5', 'CLO6', 'CLO7', 'CLO8', 'CLO9']
anomaly_inputs = ['CLO1']
model_IF = IsolationForest(contamination=float(0.1),random_state=20)
model_IF.fit(df[anomaly_inputs])
df['anomaly_scores'] = model_IF.decision_function(df[anomaly_inputs])
df['anomaly'] = model_IF.predict(df[anomaly_inputs])

print(df.loc[:, ['CLO1','anomaly_scores','anomaly'] ])
# print(df.loc[:, ['CLO1', 'CLO2', 'CLO3', 'CLO4', 'CLO5', 'CLO6', 'CLO7', 'CLO8', 'CLO9','anomaly_scores','anomaly'] ])