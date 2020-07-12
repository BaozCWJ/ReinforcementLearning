
import numpy as np

import matplotlib.pyplot as plt


def plot_his(his,his_name,smooth):
    x=np.arange(len(his))
    if smooth:
        k=100
        y=np.zeros(len(his)-k)
        for ii in range(len(his)-k):
            for jj in range(k):
                y[ii] += his[ii+jj]
            y[ii]=y[ii]/k
        plt.plot(x[:len(his)-k],y)
    else:
        plt.plot(x,his)
    plt.ylabel(his_name)
    plt.xlabel("episode")
    plt.title(his_name+"-episode")
    plt.savefig(his_name+".png")
    plt.show()


his = np.loadtxt("data")
plot_his(his,"reward",True)
#plot_his(his,'reward',True)
