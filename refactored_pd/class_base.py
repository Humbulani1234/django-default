
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

class Base(object):

    def __init__(self, custom_rcParams):
        self.custom_rcParams = plt.rcParams.update(custom_rcParams)

    def __str__(self):
        pattern = re.compile(r'^_')
        method_names = []
        for name, func in Base.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)
        return f"This is class: {self.__class__.__name__}, and it provides functionalities for others"

    def _plotting(self, title_, xlabel_, ylabel_):        
        """ function used for plotting """
        self.axs.set_title(title_)
        self.axs.set_xlabel(xlabel_)
        self.axs.set_ylabel(ylabel_)
        self.axs.spines["top"].set_visible(False)  
        self.axs.spines["right"].set_visible(False) 

    def _plot_legend(self, key_color: dict):
        labels = list(key_color.keys())
        handles = [plt.Rectangle((5, 5), 10, 10, color=key_color[label], label=label) for label in labels]
        self.axs.legend(handles=handles, fontsize=7, bbox_to_anchor=(0.9, 1.05), loc="upper left",
                        title="Legend", shadow=True)

    def _customization_bar(self):
        for i in range(0,3):
            if i%2:
                bars[i].set_color('red')
                p = str(bars[i].get_height())
                yval = bars[i].get_height()
                self.axs.text(bars[i].get_x(), yval + 3, p,fontsize=9)            
            else:
                bars.set_height('blue')
                p = str(bars[i].get_height())
                yval = bars[i].get_height()
                self.axs.text(bars[i].get_x(), yval + 3, p,fontsize=9, ha='center', color='white')
        self.axs.text(2, 16, 'Custom Text', fontsize=12, color='green')
        self.axs.annotate('Annotation', xy=(3, 13), xytext=(3.5, 16),
                           arrowprops=dict(arrowstyle='->', color='red'), fontsize=12)
    
    def _customization_pie(self, target:pd.Series):
        sizes = np.array(target.values)
        def absolute_value(val):
            a  = int(np.round(val/100.*sizes.sum(), 0))
            b = str(a) + "%"
            return b
        a = target.plot.pie(y=target.index, shadow=False, counterclock=True, rotatelabels=False, ylabel="",
                            autopct=absolute_value,colors=["#003A5D","#A19958","#808080"],
                            wedgeprops={"edgecolor":"white","linewidth":2},
                            labels=None,textprops={"size":7.5},radius=0.8)
        