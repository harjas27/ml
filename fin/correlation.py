import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd

style.use('ggplot')


def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
    df['AAPL'].plot()
    plt.show()

visualize_data()    