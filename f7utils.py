import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
try: 
    from termcolor import colored
    COLORFUL = True
except ImportError as ie:
    COLORFUL = False

if COLORFUL:
    WARNINGSTR = colored("WARNING", "red", attrs=["reverse", "blink"])
    WARNINGLEVE = colored("WARNING", "magenta", attrs=["reverse", "blink"])
else:
    WARNINGSTR = "WARNING"
    WARNINGLEVE = WARNINGSTR

def plotExperiment(exp: pd.DataFrame, size: Tuple[int, int] = (20, 8)) -> None:
    """
    Function to plot important aspects about each experiment
    """
    #Get landing time steps #TODO: Arreglar esta parte del codigo
    try:
        landing_pos_obs_list = exp.loc[((exp["leg_1"] == 1) & (exp["leg_2"] == 1))].index
        landing_pos_obs = landing_pos_obs_list[0]

        # Check bounces
        afterlanding = np.array(range(landing_pos_obs, exp.shape[0]))

        if not sum(landing_pos_obs_list != afterlanding):
            print("La nave no rebota")

        #TODO: No es buena práctica, sustituir por len() != len()
        else:
            raise ValueError

    

    except ValueError as e:
        print(f"{WARNINGLEVE}: La nave rebota varias veces")

        #TODO: Empezar a mirar la lista desde abajo para ver cuándo es la última vez
        # que rebota
        landing_pos_obs_list = exp.loc[((exp["leg_1"] == 1) & (exp["leg_2"] == 1))].index
        landing_pos_obs = landing_pos_obs_list[0]
        
    except IndexError as e:
        alert = colored("Landing position lines will not be plotted!", "red")

        print(f"{WARNINGSTR} There has been an exception when trying to \
find the landing position ({e})\n\n{alert}\n")
        landing_pos_obs = 0

    figure, axis = plt.subplots(2, 4, figsize = size)

    # X coordinate plot
    axis[0, 0].plot(exp.x_pos, -1 * exp.index)
    axis[0, 0].set_title("X trayectory")
    axis[0, 1].plot(exp.y_pos)
    axis[0, 1].set_title("Y coordinate")
    axis[1, 0].plot(exp.x_vel)
    axis[1, 0].axhline(y=0, linestyle = "--", color = "#333333")
    axis[1, 0].set_title("X velocity")
    axis[1, 1].plot(exp.y_vel)
    axis[1, 1].axhline(y=0, linestyle = "--", color = "#333333")
    axis[1, 1].set_title("Y velocity")
    axis[0, 2].plot(exp.angle)
    axis[0, 2].set_title("Angle")
    axis[1, 2].plot(exp.ang_vel)
    axis[1, 2].axhline(y=0, linestyle = "--", color = "#333333")
    axis[1, 2].set_title("Angular velocity")
    axis[0, 3].plot(exp.main_booster)
    axis[0, 3].axhline(y=0, linestyle = "--", color = "#333333")
    axis[0, 3].set_title("Main booster (Y acceleration)")
    axis[1, 3].plot(exp.lat_booster)
    axis[1, 3].axhline(y=0, linestyle = "--", color = "#333333")
    axis[1, 3].set_title("Lateral booster (X acceleration)")

    if landing_pos_obs:
        axis[0, 0].axhline(y=-1 * landing_pos_obs, linestyle = "--", color = "#ff9999")
        axis[0, 1].axvline(x=landing_pos_obs, linestyle = "--", color = "#ff9999")
        axis[1, 0].axvline(x=landing_pos_obs, linestyle = "--", color = "#ff9999")
        axis[1, 1].axvline(x=landing_pos_obs, linestyle = "--", color = "#ff9999")
        axis[0, 2].axvline(x=landing_pos_obs, linestyle = "--", color = "#ff9999")
        axis[1, 2].axvline(x=landing_pos_obs, linestyle = "--", color = "#ff9999")
        axis[0, 3].axvline(x=landing_pos_obs, linestyle = "--", color = "#ff9999")
        axis[1, 3].axvline(x=landing_pos_obs, linestyle = "--", color = "#ff9999")

    plt.show()

    return

def expLanded(exp: pd.DataFrame) -> int:
    """
    Function that returns a 1 if the experiment landed
    and 0 if it didn't
    """
    landed_obs = exp.loc[((exp["leg_1"] == 1) & (exp["leg_2"] == 1))]
    landedtimes = landed_obs.shape[0]

    if landedtimes:
        return 1
    else:
        return 0

def smoothY_pos(exper: pd.DataFrame, var:str = "y_pos") -> pd.DataFrame:
    """
    Function to clean unintended zeros (noise) like in the y_pos variable
    New value -> Mean of the two contiguous observations
    """
    explen = exper.shape[0]
    zeroIdxs = exper[exper[var]==0].index.tolist()

    for i in zeroIdxs:
        if (i != 0) and (i != explen - 1):
            exper[var][i] = (exper[var][i-1] + exper[var][i+1])/2

    return exper

# def smoothY_pos(exper: pd.DataFrame) -> pd.DataFrame:
#     """
#     Function to clean unintended zeros (noise) like in the y_pos variable
#     New value -> Mean of the two contiguous observations
#     """
#     explen = exper.shape[0]
#     zeroIdxs = exper[exper["y_pos"]==0].index.tolist()

#     for i in zeroIdxs:
#         if (i != 0):
#             exper["y_pos"][i] = exper["y_pos"][i-1] + 1.2 * exper["y_vel"][i-1]

#     return exper
