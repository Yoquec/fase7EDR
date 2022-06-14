'''
Created on Sat Jun 11 09:15:45 PM CEST 2022

@file: fileAggregator.py

@author: Yoquec (github)

@description:
    Archivo de python que analizará todos los experimentos buscando
    las estadísticas sobre ellos discutidas en el notebook fase7FE.ipynb

@License:
    MIT License

    Copyright (c) 2022 Alvaro Viejo

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
'''
import argparse
import os
from f7utils import expLanded
import pandas as pd
import numpy as np
from typing import Tuple, NewType

try: 
    from termcolor import colored
    COLORFUL = True
except ImportError as ie:
    print(f"""The package `termcolor` is an optional dependency, \
which seems to not be installed in your system.
It's purpose is to provide colorful and more informative prints in the terminal

To install it, stop the script with CTRL-C and type:
\t>>> python3 -m pip install termcolor
Or, get all the optional dependencies at once running
\t>>> python3 -m pip install -r requirements.txt
""")
    COLORFUL = False

#------------------CONSTANTS------------------
NDArray = NewType("NDArray", np.ndarray)
DEFOUTFILE = "aggregateData.csv"
DATASETS = ["train", "test"]

#------------------FUNCTIONS------------------
def getColoredText(string: str, color:str, highlight: bool = False , hascolor:bool = COLORFUL) -> str:
    """
    Function to generate colorful terminal text
    """
    if hascolor:
        if highlight:
            text = colored(string, color, attrs=["reverse", "blink"])
        else:
            text = colored(string, color)
    else:
        text = string

    return text

def confirmArgs(string:str) -> bool:
    """
    Function to make sure that the arguments have been introduced correctly
    """
    if string.lower() == "n":
        print(f"\n{getColoredText('WARNING', 'magenta', highlight=True)}: Specify your \
own arguments in the command-line. \nType {getColoredText('python3 fileAggregator.py -h', 'green')} \
for more help")
        return False
    elif string.lower() == "y" or string.lower() == "":
        return True
    else:
        return False

def getArgumentParser() -> argparse.ArgumentParser:
    """
    Function that prepares an argument parser with argument options
    (To be used indide getArguments)
    """
    parser = argparse.ArgumentParser(
            description="""
    ARCHIVO de python que analizará todos los experimentos buscando
    las estadísticas sobre ellos discutidas en el notebook fase7FE.ipynb
    """
            )
    # Adding optional arguments
    parser.add_argument("-d", "--dataset", type=bool, help = "Decide wether to \
use the training or testing set (Choose 0 for train and 1 for test)")
    parser.add_argument("-o", "--output_file", help = "File where the output\
 will dumped to")

    return parser

def getArguments() -> Tuple[int, str]:
    """
    Function that reads and sets program variable when reading 
    command-line arguments
    """
    parser = getArgumentParser()
    args = parser.parse_args()

    if args.dataset: dataset = args.dataset
    else: dataset = 0

    if args.output_file: outfile = args.output_file
    else: outfile = DEFOUTFILE

    return (dataset, outfile)

def checkDir() -> None:
    contents = os.listdir()
    if "data" not in contents:
        raise EnvironmentError("""The data of the projet should be stored in a `data` folder as such
            fileAggregator.py
                    └── data/
                       ├── train/
                       └── test/
""")
    
    return

#------------------------------------------------------------#
# BEGINNING OF SCRIPT
#------------------------------------------------------------#
if __name__ == "__main__":
    # Check the folders
    checkDir()

    # Get arguments
    dataset, outfile = getArguments()
    DATAFOLDER = f"data/{DATASETS[dataset]}"
    print(f"\n[{getColoredText('INFO', 'green')}]:The selected dataset is \
{getColoredText(DATASETS[dataset], 'white', highlight=True)} and the output \
file will be {getColoredText(outfile, 'blue')}")

    # Check the args are correct
    if not confirmArgs(input("\nContinue? [Y/n]: ")):
        exit()

    # Open the xlsx 
    print(f"\n[{getColoredText('INFO', 'green')}]: Opening file \
{getColoredText('data/experiments_summary_{}.xlsx'.format(DATASETS[dataset]), 'blue')}")
    summary = pd.read_excel(f"data/experiments_summary_{DATASETS[dataset]}.xlsx", 0)
    
    # Prepare np arrays to add to the dataframe
    lands = np.zeros(shape=summary.shape[0])

    # Analize each of the experiments
    print(f"\n\t[{getColoredText('ANALYZING', 'cyan')}]: Analyzing csvs")
    expn = 0
    for expfile in summary.filename:
        exp: pd.DataFrame = pd.read_csv(f"{DATAFOLDER}/{expfile}")
        # Analisys magic: {{{

            # Experiment Landing
        explands = expLanded(exp)
        if explands: 
            lands[expn] = explands
            

        #}}}
        del exp
        expn += 1

    print(f"\n\t[{getColoredText('ANALYZING', 'cyan')}]: {getColoredText('Finished', 'magenta')}")
    
    # Add the new columns to the dataframe
    print(f"\n[{getColoredText('INFO', 'green')}]: Adding variables to the dataframe")
    summary["lands"] = lands

    # Write the file
    print(f"\n[{getColoredText('INFO', 'green')}]: Saving the file to a csv format to {getColoredText('data/{}'.format(outfile), 'blue')}")
    summary.to_csv(f"data/{outfile}")
    print(f"\n[{getColoredText('FINISHED', 'magenta')}]: {getColoredText('OK', 'green')}")

