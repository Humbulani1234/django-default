
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------Base Class----------------------------------------------------------

class Base():

    def __init__(self, custom_rcParams):

        self.custom_rcParams = plt.rcParams.update(custom_rcParams)

    def plotting(self, title_, xlabel_, ylabel_):
        
        """ function used for plotting """

        self.axs.set_title(title_)
        self.axs.set_xlabel(xlabel_)
        self.axs.set_ylabel(ylabel_)
        self.axs.spines["top"].set_visible(False)  
        self.axs.spines["right"].set_visible(False) 

    def plot_legend(self, key_color: dict):

        key_color = {"Not-Absorbed":"#003A5D", "Absorbed":"#A19958"}
        labels = list(key_color.keys())
        handles = [plt.Rectangle((5,5),10,10, color=key_color[label]) for label in labels]
        
        self.axs.legend(handles, labels, fontsize=7, bbox_to_anchor=(1.13,1.17), loc="upper left", title="legend",shadow=True)
    
