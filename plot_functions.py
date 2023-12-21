import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def barplot_groups(df, file_name = None, title = None, yaxis_label = 'Frequencies', barmode = 'relative'):
    
    fig = px.bar(df, title=title, barmode=barmode)

    # Update layout
    fig.update_layout(
        title=f"{title}",
        xaxis=dict(title='Categories', tickangle=45),  # Rotate x-axis labels for better readability
        yaxis=dict(title=yaxis_label),
        height=500,  # Adjust the height of the plot as needed
        margin=dict(l=80, r=80, t=80, b=80),  # Adjust margins for better appearance
        showlegend=False,
    )
    if barmode=='group':
        fig.update_layout(
            showlegend=True
        )

    fig.show()
    fig.write_html("Figures/Plotly_"+file_name+".html")

def barplot_solo(df, file_name = None, title = None, color = None, yaxis_label = 'Frequencies', barmode = 'relative'):
    
    fig = px.bar(df, title=title, barmode=barmode, color_discrete_sequence=[color])

    # Update layout
    fig.update_layout(
        title=f"{title}",
        xaxis=dict(title='Categories', tickangle=45),  # Rotate x-axis labels for better readability
        yaxis=dict(title=yaxis_label),
        height=500,  # Adjust the height of the plot as needed
        margin=dict(l=80, r=80, t=80, b=80),  # Adjust margins for better appearance
        showlegend=False,
    )
    if barmode=='group':
        fig.update_layout(
            showlegend=True
        )

    fig.show()
    fig.write_html("Figures/Plotly_"+file_name+".html")