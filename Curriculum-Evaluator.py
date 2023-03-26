# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 20:10:26 2023

@author:
Soham Goswami
Shalin Shah
Harsh Prajapati
Drashti Koshti
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import IsolationForest

startYear = 2000
endYear = 2022
minMarks = 0
maxMarks = 100
clColumns = ['CLO1', 'CLO2', 'CLO3', 'CLO4', 'CLO5', 'CLO6', 'CLO7', 'CLO8', 'CLO9']

df = pd.read_csv('S:/University Of Windsor/Term 2/ADT/Project/Submission/curriculum_evaluator/Dataset.csv')

# Removing the missing values rows
df = df.dropna()

# -----------------------------------------------------------------------------------------------
# Creating all CLO in one line graph
# -----------------------------------------------------------------------------------------------
# filter the required columns and rows
# df = df[['Year', 'CLO1', 'CLO2', 'CLO3', 'CLO4', 'CLO5', 'CLO6', 'CLO7', 'CLO8', 'CLO9']]
# df = df[df['Year'].between(startYear, endYear)]
# df = df[(df[clColumns] >= 1).all(1) & (df[clColumns] <= 100).all(1)]

# # reshape the data to long format
# df = pd.melt(df, id_vars='Year', var_name='CLO', value_name='Value')

# # plot the line chart
# fig = px.line(df, x='Year', y='Value', color='CLO')

# # set the axis ranges and title
# fig.update_layout(
#     title='CLO Performance',
#     xaxis=dict(range=[startYear, endYear]),
#     yaxis=dict(range=[minMarks, maxMarks])
# )

# fig.show()
# -----------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------
# Training the model
# -----------------------------------------------------------------------------------------------
prefix = 'CLO'
anomaly_string = ''
anomaly_inputs = []

# Keeping contamination = 0.1 for expecting 10% data to be contaminated
# and random_state = 20 as a parameter which can provide same values again
model_IF = IsolationForest(contamination=float(0.1),random_state=20)

# For CL01, CLO2, CLO3, CLO4, CLO5, CLO6, CLO7, CLO8 and CLO9
for i in range(1, 10):
    anomaly_string = f"{prefix}{i}"
    print(anomaly_string)
    anomaly_inputs.append(anomaly_string)
    model_IF.fit(df[anomaly_inputs])
    df['anomaly_scores'] = model_IF.decision_function(df[anomaly_inputs])
    df['anomaly'] = model_IF.predict(df[anomaly_inputs])

    # Storing each anomaly to a separate file as 'COL*_output.csv'
    subset_df = df.loc[:, [anomaly_string,'anomaly_scores','anomaly'] ]
    subset_df.to_csv(anomaly_string + '_output.csv', index=False)

    # # Creating FacetGrid object and mapping it to a scatter plot
    # g = sns.FacetGrid(df, hue='anomaly', height=5)
    # # g.map(sns.scatterplot, anomaly_string, 'anomaly_scores', s=100)
    # g.map(sns.scatterplot, 'Year', anomaly_string, s=100)
    # plt.title(anomaly_string)
    # # Displaying the plot
    # plt.show()


    # Creating a line graph of X=Year vs Y=Scores achieved in each CLO
    # fig = px.line(df.reset_index(), x='Year', y=anomaly_string, title=anomaly_string)

    # Creating a scatter graph of X=Year vs Y=Scores achieved in each CLO
    fig = px.scatter(df.reset_index(), x='Year', y=anomaly_string, color='anomaly', title=anomaly_string)

    # Modifying the Plotly Express to make the RangeSlider for 
    # the years visible and view the Y-label as Scores
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_yaxes(title_text='Scores')
    fig.show()

    anomaly_inputs.clear()