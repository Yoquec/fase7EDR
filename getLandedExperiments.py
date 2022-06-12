'''
Created on Sat Jun 11 08:21:59 PM CEST 2022

@file: getLandedExperiments.py

@author: Yoquec

@description:
    Archivo de python que comprueba para todos los experimentos de un directorio si
    han aterrizado o no.
'''
from f7utils import expLanded
import pandas as pd
import sys
from typing import List
try: 
    from termcolor import colored
    COLORFUL = True
except ImportError as ie:
    COLORFUL = False
import os

if COLORFUL:
    FINISHED = colored("FINISHED", "green", attrs=["reverse", "blink"])
else:
    FINISHED = "FINISHED"

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        OUTFILE = sys.argv[1]
    else:
        OUTFILE = "landedExperimentsTest.out"

    # Get data
    traincsvs = os.listdir("data/test/")
    nfiles = len(traincsvs)
    contains: List[None | str] = [None]*nfiles

    # Go throught each of the files and check if the experiment landed
        # Keep track of the number of landed files
    landed = 0
    for i in range(nfiles):
        expfile = traincsvs[i]
        exp: pd.DataFrame = pd.read_csv(f"data/test/{expfile}")
        if expLanded(exp):
            contains[i] = expfile
            landed += 1
        del exp

    if COLORFUL:
        print(f"{FINISHED}: There are {colored(str(landed),'white', attrs=['reverse', 'blink'])} experiments that landed")
    else:
        print(f"{FINISHED}: There are {str(landed)} experiments that landed")

    with open(OUTFILE, "w+") as outfile:
        for experiment in contains:
            if experiment:
                outfile.write(f"{experiment}\n")

    print("\nFinished generating file.")

