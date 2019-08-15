# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 10:18:29 2019

@author: jplineb
"""

import matplotlib.pyplot as plt

for e in [5,7,9]:
    print(e)
    fig = plt.figure()
    ax = plt.gca()
    df_plot = df_mean[df_mean.epoch==e]
    ax.scatter(df_plot.lr, df_plot.AUROC, c=np.log(df_plot.weight_decay), cmap='viridis')
    ax.set_xscale('log')
    plt.show()