import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Union, List, NewType
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

NDArray = NewType("NDArray", np.ndarray)

def plotExperiment(exp: pd.DataFrame, size: Tuple[int, int] = (20, 8)) -> None:
    """
    Function to plot important aspects about each experiment
    """
    #Get landing time steps #TODO: Arreglar esta parte del codigo
#     try:
#         # landing_pos_obs_list = exp.loc[((exp["leg_1"] == 1) & (exp["leg_2"] == 1))].index
#         landing_pos_obs_list = getPossibleLandings(exp).index
#         landing_pos_obs = landing_pos_obs_list[0]

        
#     except IndexError as e:
#         alert = colored("Landing position lines will not be plotted!", "red")

#         print(f"{WARNINGSTR} There has been an exception when trying to \
# find the landing position ({e})\n\n{alert}\n")
#         landing_pos_obs = 0

    landed = expLanded(exp)
    
    if landed:
        landingPos, bounces = getBounces(exp)
    else:
        alert = colored("Landing position lines will not be plotted!", "red")

        print(f"{WARNINGSTR} There has been an exception when trying to \
find the landing position\n\n{alert}\n")
        landingPos = 0  
        bounces = []

    # Plots -------------------------------------------------
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

    if landingPos:
        axis[0, 0].axhline(y=-1 * landingPos, linestyle = "--", color = "#ff9999")
        axis[0, 1].axvline(x=landingPos, linestyle = "--", color = "#ff9999")
        axis[1, 0].axvline(x=landingPos, linestyle = "--", color = "#ff9999")
        axis[1, 1].axvline(x=landingPos, linestyle = "--", color = "#ff9999")
        axis[0, 2].axvline(x=landingPos, linestyle = "--", color = "#ff9999")
        axis[1, 2].axvline(x=landingPos, linestyle = "--", color = "#ff9999")
        axis[0, 3].axvline(x=landingPos, linestyle = "--", color = "#ff9999")
        axis[1, 3].axvline(x=landingPos, linestyle = "--", color = "#ff9999")
    
    if len(bounces) != 0:
        for bounce in bounces:
            axis[0, 0].axhline(y=-1 * bounce, linestyle = "--", color = "#ffcccc")
            axis[0, 1].axvline(x=bounce, linestyle = "--", color = "#ffcccc")
            axis[1, 0].axvline(x=bounce, linestyle = "--", color = "#ffcccc")
            axis[1, 1].axvline(x=bounce, linestyle = "--", color = "#ffcccc")
            axis[0, 2].axvline(x=bounce, linestyle = "--", color = "#ffcccc")
            axis[1, 2].axvline(x=bounce, linestyle = "--", color = "#ffcccc")
            axis[0, 3].axvline(x=bounce, linestyle = "--", color = "#ffcccc")
            axis[1, 3].axvline(x=bounce, linestyle = "--", color = "#ffcccc")

    plt.show()

    return

def expLanded(exp: pd.DataFrame) -> int:
    """
    Function that returns a 1 if the experiment landed
    and 0 if it didn't
    """
    # landed_obs = exp.loc[((exp["leg_1"] == 1) & (exp["leg_2"] == 1))]
    landed_obs = getPossibleLandings(exp)
    landedtimes = landed_obs.shape[0]

    if landedtimes:
        return 1
    else:
        return 0

def getPossibleLandings(exp: pd.DataFrame, y_pos_sensitivity: float = 0.005\
        , y_vel_sensitivity: float = 0.05) -> pd.DataFrame:
    """
    Function that returns the DataFrame of all possible observations that may be landings.

    We have to use constrains of minimum error because we have to remember that we are working 
    with continous
    """
    return exp.loc[((exp["leg_1"] == 1) & (exp["leg_2"] == 1)) |\
            ((abs(exp["y_pos"]) < y_pos_sensitivity) & (abs(exp["y_vel"]) < y_vel_sensitivity))]

def expBounced(exp: pd.DataFrame) -> Union[ int, bool ]:
    """
    Function that returns wether an experiment bounced or not
    """

    landing_obs_list = getPossibleLandings(exp).index
    landing_pos_obs = landing_obs_list[0]

    # Hacemos una lista de indices desde que cae al suelo por primera vez hasta 
    # El final de observaciones y vemos si los tamaños son iguales.
    # Si lo son, entonces no habrá rebotado 
    afterlanding = np.array(range(landing_pos_obs, exp.shape[0]))

    if len(landing_obs_list) == len(afterlanding):
        return 0

    else:
        return 1

def getBounces(exp:pd.DataFrame) -> Tuple[int, List[int]]:
    """
    Function that gives us where have experiments bounced
    It returns the timesteps where bounces have been detected
    """
    # If it didn't bounce, return the empty list
    landing_pos_obs_list = getPossibleLandings(exp)

    if landing_pos_obs_list.shape[0] == 0:
        return (exp.shape[0],[])

    else:
        # Create empty array vector
        bounces = []
        # Variable to keep track of contact with the floor 
        touching = True

        # TODO: Implement bounces
        landing_idx = landing_pos_obs_list.index
        landing_idx0: int = landing_idx[0]
        afterlanding = np.array(range(landing_idx0, exp.shape[0]))

        bounces.append(landing_idx0) 

        for i in afterlanding:

            # If the current index is NOT in the list
            if i not in landing_idx:
                if touching:
                    touching = False
            
            # If the current index is in the list.
            else:
                if not touching:
                    touching = True
                    bounces.append(i)
        
        landingPos = bounces.pop()
                    
        return (landingPos, bounces)

def smoothY_pos(exper: pd.DataFrame, var:str = "y_pos", y_pos_sensitivity: float = 0.005) -> pd.DataFrame:
    """
    Function to approximately clean unintended zeros (noise) like in the y_pos variable
    New value -> Mean of the two contiguous observations
    """
    explen = exper.shape[0]
    zeroIdxs = exper[abs(exper[var])< y_pos_sensitivity].index.tolist()

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

def angularVelFOStats(exp: pd.DataFrame) -> Tuple[float, float]:
    """
    Function to get the Firt Order Statistics (mean and variance) from thof the angular velocity
    """

    angVector = exp.ang_vel
    angMean = np.mean(angVector)
    angVar = np.var(angVector)

    return (angMean, angVar)

def meanFinalAngular(exp: pd.DataFrame, nObs: int = 15) -> float:
    """
    Function to get the mean of the last nObs from the angular velocity variable
    """
    
    angVector = exp.ang_vel
    finalVector = angVector[-nObs:]
    
    return np.mean(finalVector)

def meanFinalY_pos(exp: pd.DataFrame, nObs: int = 15) -> float:
    """
    Funtion to get the mean of the last nObs from the Y position
    """

    y_PosVector = exp.y_pos
    y_PosFinal = y_PosVector[-nObs:]

    return np.mean(y_PosFinal)

def meanFinalY_vel(exp: pd.DataFrame, nObs: int = 15) -> float:
    """
    Funtion to get the mean of the last nObs from the Y position
    """

    y_VelVector = exp.y_vel
    y_VelFinal = y_VelVector[-nObs:]

    return np.mean(y_VelFinal)

def meanFinalAngVel(exp: pd.DataFrame, nObs: int = 15) -> float:
    """
    Funtion to get the mean of the last nObs from the Y position
    """

    y_PosVector = exp.y_pos
    y_PosFinal = y_PosVector[-nObs:]

    return np.mean(y_PosFinal)

def boosterUsed(exp: pd.DataFrame) -> Tuple[float, float]:
    """
    Function to get the usage of the booster (mean and variance)
    """

    mainBooster = exp.main_booster
    boostMean = np.mean(mainBooster)
    boostVar = np.var(mainBooster)

    return (boostMean, boostVar)
