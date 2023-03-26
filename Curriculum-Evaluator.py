# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 20:10:26 2023

@author:
Soham Goswami
Shalin Shah
Harsh Prajapati
Drashti Koshti
"""

import tkinter as tk
from tkinter import RIGHT, Y, Scrollbar, filedialog
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

def openfile():
    """
       Desc: Function for uploading the user CSV file
       param: None
    """
    root.filename: object = tk.filedialog.askopenfilename(initialdir="/",
                                                          title="Select a CSV File",
                                                          filetypes=(("csv files", "*.csv"),
                                                                     ("all files", "*.*")))
    return root.filename

def getfiledata():
    """
        Desc: Read Selected CSV File
        param: None
    """
    df = pd.read_csv(root.filename)
    # Removing the missing values rows
    df = df.dropna()
    return df

def viewallCLO():
    """
        Desc: View All CLO data accumulated by the selected CSV as line graph
        param: None
    """
    text_box.delete("1.0", tk.END)
    text_box.insert(tk.END, "\nView All CLO data values\n\n\n")
    # -----------------------------------------------------------------------------------------------
    # Creating all CLO in one line graph
    # -----------------------------------------------------------------------------------------------
    df = getfiledata()
    text_box.insert(tk.END, "\n\nSummary of the data from selected CSV\n\n\n")
    text_box.insert(tk.END, df.describe())
    # filter the required columns and rows
    df = df[['Year', 'CLO1', 'CLO2', 'CLO3', 'CLO4', 'CLO5', 'CLO6', 'CLO7', 'CLO8', 'CLO9']]
    df = df[df['Year'].between(startYear, endYear)]
    df = df[(df[clColumns] >= 1).all(1) & (df[clColumns] <= 100).all(1)]

    # reshape the data to long format
    df = pd.melt(df, id_vars='Year', var_name='CLO', value_name='Value')

    # plot the line chart
    fig = px.line(df, x='Year', y='Value', color='CLO')

    # set the axis ranges and title
    fig.update_layout(
        title='CLO Performance',
        xaxis=dict(range=[startYear, endYear]),
        yaxis=dict(range=[minMarks, maxMarks])
    )

    fig.show()
    # -----------------------------------------------------------------------------------------------

def viewCLOanomaly(clo_num):
    text_box.delete("1.0", tk.END)
    # -----------------------------------------------------------------------------------------------
    # Training the model
    # -----------------------------------------------------------------------------------------------
    prefix = 'CLO'
    anomaly_string = f"{prefix}{clo_num}"
    anomaly_inputs = []
    df = getfiledata()
    text_box.insert(tk.END, "\nView " + anomaly_string +" Anomaly Representation\n\n\n")

    # contamination = 0.1 for expecting 10% data to be contaminated
    # random_state = 20 as a parameter which can provide same values again
    model_IF = IsolationForest(contamination=float(0.1),random_state=20)

    print(anomaly_string)
    anomaly_inputs.append(anomaly_string)
    model_IF.fit(df[anomaly_inputs])
    df['anomaly_scores'] = model_IF.decision_function(df[anomaly_inputs])
    df['anomaly'] = model_IF.predict(df[anomaly_inputs])
    text_box.insert(tk.END, df[['Year', anomaly_string, 'anomaly_scores', 'anomaly']])

    # Storing each anomaly to a separate file as 'COL*_output.csv'
    subset_df = df.loc[:, [anomaly_string,'anomaly_scores','anomaly'] ]
    subset_df.to_csv(anomaly_string + '_output.csv', index=False)

    # Creating a scatter graph of X=Year vs Y=Scores achieved in each CLO
    fig = px.scatter(df.reset_index(), x='Year', y=anomaly_string, color='anomaly', title=anomaly_string)

    # Modifying the Plotly Express to make the RangeSlider for 
    # the years visible and view the Y-label as Scores
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_yaxes(title_text='Scores')
    fig.show()


    # Creating FacetGrid object and mapping it to a scatter plot
    g = sns.FacetGrid(df, hue='anomaly', height=5)
    g.map(sns.scatterplot, 'Year', anomaly_string, s=100)
    plt.title(anomaly_string)
    # Displaying the plot
    plt.show()


# GUI for the program
root = tk.Tk()
frame = tk.Frame(root, bg='blue')
frame.pack()

btn_open_file = tk.Button(frame,  # Button for calling upload function
                            text='Select File',
                            bg='dark grey',
                            fg='blue',
                            padx=30,
                            pady=10,
                            command=openfile)
btn_open_file.pack(side=tk.LEFT)

btn_view_all = tk.Button(frame,  # Button for calling viewallCLO function
                         text="View All CLO",
                         bg="dark grey",
                         fg='blue',
                         padx=40,
                         pady=10,
                         command=viewallCLO)
btn_view_all.pack(side=tk.LEFT)

btn_view_clo1 = tk.Button(frame,  # Button for calling sentiment function
                         text="View CLO1 Anomaly",
                         bg="dark grey",
                         fg='blue',
                         padx=10,
                         pady=10,
                         command=lambda: viewCLOanomaly(1))
btn_view_clo1.pack(side=tk.LEFT)

btn_view_clo2 = tk.Button(frame,  # Button for calling sentiment function
                         text="View CLO2 Anomaly",
                         bg="dark grey",
                         fg='blue',
                         padx=10,
                         pady=10,
                         command=lambda: viewCLOanomaly(2))
btn_view_clo2.pack(side=tk.LEFT)

btn_view_clo3 = tk.Button(frame,  # Button for calling sentiment function
                         text="View CLO3 Anomaly",
                         bg="dark grey",
                         fg='blue',
                         padx=10,
                         pady=10,
                         command=lambda: viewCLOanomaly(3))
btn_view_clo3.pack(side=tk.LEFT)

btn_view_clo4 = tk.Button(frame,  # Button for calling sentiment function
                         text="View CLO4 Anomaly",
                         bg="dark grey",
                         fg='blue',
                         padx=10,
                         pady=10,
                         command=lambda: viewCLOanomaly(4))
btn_view_clo4.pack(side=tk.LEFT)

btn_view_clo5 = tk.Button(frame,  # Button for calling sentiment function
                         text="View CLO5 Anomaly",
                         bg="dark grey",
                         fg='blue',
                         padx=10,
                         pady=10,
                         command=lambda: viewCLOanomaly(5))
btn_view_clo5.pack(side=tk.LEFT)

btn_view_clo6 = tk.Button(frame,  # Button for calling sentiment function
                         text="View CLO6 Anomaly",
                         bg="dark grey",
                         fg='blue',
                         padx=10,
                         pady=10,
                         command=lambda: viewCLOanomaly(6))
btn_view_clo6.pack(side=tk.LEFT)

btn_view_clo7 = tk.Button(frame,  # Button for calling sentiment function
                         text="View CLO7 Anomaly",
                         bg="dark grey",
                         fg='blue',
                         padx=10,
                         pady=10,
                         command=lambda: viewCLOanomaly(7))
btn_view_clo7.pack(side=tk.LEFT)

btn_view_clo8 = tk.Button(frame,  # Button for calling sentiment function
                         text="View CLO8 Anomaly",
                         bg="dark grey",
                         fg='blue',
                         padx=10,
                         pady=10,
                         command=lambda: viewCLOanomaly(8))
btn_view_clo8.pack(side=tk.LEFT)

btn_view_clo9 = tk.Button(frame,  # Button for calling sentiment function
                         text="View CLO9 Anomaly",
                         bg="dark grey",
                         fg='blue',
                         padx=10,
                         pady=10,
                         command=lambda: viewCLOanomaly(9))
btn_view_clo9.pack(side=tk.LEFT)

scrollbar: Scrollbar = tk.Scrollbar(root)  # Creating a scrollbar
scrollbar.pack(side=RIGHT, fill=Y)
text_box = tk.Text(root, width=150, height=120, bg='lightblue', yscrollcommand=scrollbar.set)
text_box.pack()
scrollbar.config(command=text_box.yview)

# Set the window size and position
# root.geometry("800x600+0+0")
root.attributes('-fullscreen', True)

root.mainloop()