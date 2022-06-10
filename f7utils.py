import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


def plotExperiment(exp: pd.DataFrame, size: Tuple[int, int] = (20, 8)) -> None:
    """
    Function to plot important aspects about each experiment
    """
    figure, axis = plt.subplots(2, 4, figsize = size)

    # X coordinate plot
    axis[0, 0].plot(exp.x_pos, -1 * exp.index)
    axis[0, 0].set_title("X coordinate")
    axis[0, 1].plot(exp.y_pos)
    axis[0, 1].set_title("Y coordinate")
    axis[1, 0].plot(exp.x_vel)
    axis[1, 0].set_title("X velocity")
    axis[1, 1].plot(exp.y_vel)
    axis[1, 1].set_title("Y velocity")
    axis[0, 2].plot(exp.angle)
    axis[0, 2].set_title("Angle")
    axis[1, 2].plot(exp.ang_vel)
    axis[1, 2].set_title("Angular velocity")
    axis[0, 3].plot(exp.main_booster)
    axis[0, 3].set_title("Main booster (Y acceleration)")
    axis[1, 3].plot(exp.lat_booster)
    axis[1, 3].set_title("Lateral booster (X acceleration)")

    plt.show()

    return

def cleanZeroes(exp: pd.DataFrame, var:str = "y_pos") -> pd.DataFrame:
    """
    Function to clean unintended zeros (noise) like in the y_pos variable
    New value -> Mean of the two contiguous observations
    """
    explen = exp.shape[0]
    idxs = exp[exp[var]==0].index.tolist()

    for i in idxs:
        if (i != 0) and (i != explen - 1):
            exp[i,1] = np.mean(exp[var][i-1],exp[var][i+1])   

    return exp