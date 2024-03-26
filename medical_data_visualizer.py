import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Import data
data = pd.read_csv('medical_examination.csv')
df = pd.DataFrame(data)
#print(df.info())

# Add 'overweight' column
""" Add an overweight column to the data. To determine if a person is overweight, first calculate their BMI by dividing their weight in kilograms by the square of their height in meters. If that value is > 25 then the person is overweight. Use the value 0 for NOT overweight and the value 1 for overweight. """
#bmi = df.weight / (df.height / 100) ** 2 
#print(bmi)

df['overweight'] = [(1 if x > 25 else 0) for x in (df['weight'] / ((df['height'] / 100) ** 2))]

#print(df.overweight)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = df.cholesterol.apply(lambda x: 0 if x <= 1 else 1)
df['gluc'] = df.gluc.apply(lambda x: 0 if x <= 1 else 1)
#print(df.gluc)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_subdata = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=df_subdata)
    #print(df_cat)


    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index()
    df_cat = df_cat.rename(columns={0: 'total'})
    #print(df_group)

    # Draw the catplot with 'sns.catplot()'
    representation = sns.catplot(data=df_cat, x='variable', y='total', col='cardio', hue='value', kind='bar')
    #plt.show()

 
    # Get the figure for the output
    fig = representation.fig


    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig
    

# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                (df['height'] >= df['height'].quantile(0.025)) &
                (df['height'] <= df['height'].quantile(0.975)) &
                (df['weight'] >= df['weight'].quantile(0.025)) &
                (df['weight'] <= df['weight'].quantile(0.975))
                ]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(16, 9))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, square=True, linewidths=0.5, annot=True, fmt="0.1f")


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
