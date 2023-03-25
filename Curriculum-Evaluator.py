import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

df = pd.read_csv('S:/University Of Windsor/Term 2/ADT/Project/Submission/curriculum_evaluator/Dataset.csv')

# Removing the missing values rows
df = df.dropna()

# print(df.describe())

# Training the model
prefix = 'CLO'
anomaly_string = ''
anomaly_inputs = []
model_IF = IsolationForest(contamination=float(0.1),random_state=20)
# For CL01, CLO2, CLO3, CLO4, CLO5, CLO6, CLO7, CLO8 and CLO9
for i in range(1, 10):
    anomaly_string = f"{prefix}{i}"
    print(anomaly_string)
    anomaly_inputs.append(anomaly_string)
    model_IF.fit(df[anomaly_inputs])
    df['anomaly_scores'] = model_IF.decision_function(df[anomaly_inputs])
    df['anomaly'] = model_IF.predict(df[anomaly_inputs])
    subset_df = df.loc[:, [anomaly_string,'anomaly_scores','anomaly'] ]
    subset_df.to_csv(anomaly_string + '_output.csv', index=False)

    # Creating FacetGrid object and mapping it to a scatter plot
    g = sns.FacetGrid(df, hue='anomaly', height=5)
    # g.map(sns.scatterplot, anomaly_string, 'anomaly_scores', s=100)
    g.map(sns.scatterplot, 'Year', anomaly_string, s=100)
    plt.title(anomaly_string)

    # Displaying the plot
    plt.show()

    anomaly_inputs.clear()