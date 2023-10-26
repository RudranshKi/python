import csv
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
filename = "14_3n.csv"
corner = [[] for _ in range(7)]

pcorner = [[] for _ in range(7)]

rows_skip = [0,1,2,3,4,5,7,9,10,11,12,13,14,15,16]

corner[0], corner[1], corner[2], corner[3], corner[4], corner[5],corner[6] = np.loadtxt(filename, delimiter=",", usecols=(2,8, 9, 10, 11, 12, 13), skiprows=13, unpack=True, dtype=float)
minimum,maximum = np.loadtxt(filename, delimiter=",", usecols=(6,7), skiprows=13, unpack=True, dtype=float)
pcorner[0], pcorner[1], pcorner[2], pcorner[3], pcorner[4], pcorner[5],pcorner[6] = np.loadtxt(filename, delimiter=",", usecols=(2,8, 9, 10, 11, 12, 13), max_rows=9, unpack=True, dtype=str)

for i in range(len(pcorner)) :
    pcorner[i][0] = pcorner[i][5]
    pcorner[i][1] = pcorner[i][8]

for i in range(len(pcorner)):
    pcorner[i] = pcorner[i][:2]

def convertNanoSecond(seconds):
    return seconds * 1e9

for i in range(7):
    corner[i] = [convertNanoSecond(value) for value in corner[i]]
    

for i in range(7):
    minimum[i] = convertNanoSecond(minimum[i])
for i in range(7):
    maximum[i] = convertNanoSecond(maximum[i])


read1to0  = [corner[0][0], corner[1][0], corner[2][0], corner[3][0], corner[4][0], corner[5][0],corner[6][0],minimum[0],maximum[0]]
read0to1  = [corner[0][1], corner[1][1], corner[2][1], corner[3][1], corner[4][1], corner[5][1],corner[6][1],minimum[1],maximum[1]]
precharge = [corner[0][2], corner[1][2], corner[2][2], corner[3][2], corner[4][2], corner[5][2],corner[6][2],minimum[2],maximum[2]]
write1to0 = [corner[0][3], corner[1][3], corner[2][3], corner[3][3], corner[4][3], corner[5][3],corner[6][3],minimum[3],maximum[3]]
write0to1 = [corner[0][4], corner[1][4], corner[2][4], corner[3][4], corner[4][4], corner[5][4],corner[6][4],minimum[4],maximum[4]]


col_names = ["Delay Output"] + ["mos: " + pcorner[i][0] + ", \ntemp :"+pcorner[i][1]+" °C" for i in range(7)] + ["Minimum"] + ["Maximum"]

data = [
    ["read 0 from 1"]  + [f"{value:.3f} nS" for value in read1to0],
    ["read 1 from 0"]  + [f"{value:.3f} nS" for value in read0to1],
    ["Precharge"]      + [f"{value:.3f} nS" for value in precharge],
    ["Write 0 from 1"] + [f"{value:.3f} nS" for value in write1to0],
    ["Write 1 from 0"] + [f"{value:.3f} nS" for value in write0to1],
    ]


print(tabulate(data, headers=col_names, tablefmt="fancy_grid", showindex="always"))

table = tabulate(data, headers=col_names, tablefmt="fancy_grid", showindex="always")

plt.figure(figsize=(6, 5))
plt.axis('off')  # Turn off axis

# Render the table as text and add it to the plot
plt.text(0.1, 0.1, table, fontsize=12, family='monospace')

# Save the plot as an image (e.g., PNG)
plt.savefig("table.png", bbox_inches='tight', pad_inches=0.1, dpi=300)

