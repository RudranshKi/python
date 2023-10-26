import csv
import numpy as np
import matplotlib.pyplot as plt
from math import *
import math
from sympy import Point, Line
# Load data from the CSV file
q_qbX, q_qbY, qb_qX, qb_qY = np.loadtxt("SRAM_SNM.csv", delimiter=",", skiprows=1, unpack=True)


line_length = 1    # Length of the line to draw from the points

# To find point of intersection of the line segments 

def find_intersection(line1, line2):
    line1 = Line(Point(line1[0]), Point(line1[1]))
    line2 = Line(Point(line2[0]), Point(line2[1]))

    intersection = line1.intersection(line2)

    if intersection:
        return (intersection[0].x, intersection[0].y)
    else:
        return None

#########################################################################################################################

# To calculate noise margin from 2nd lobe

line1= []
line3=[]
line2 = []
line4 = []
line_angle1 = 270
line_angle2=180

# Calculate and store only points with -1 slope
slope_threshold = 0.25    # How close we want the slope to be to -1 slope
slope_pairs = [] 
def negSlope2(pointX_1,pointX_2,pointY_1,pointY_2) :
    if (pointX_2 > 0.734) and (pointY_2 < 0.752) :
        derivativeX = pointX_1-pointX_2
        derivativeY = pointY_1-pointY_2
        slope =derivativeX/derivativeY
        if abs(slope+1) <= slope_threshold :
            slope_pairs.append([pointX_2,pointY_2])
           
for i in range(len(q_qbX)-1) :
    negSlope2(q_qbX[i],q_qbX[i+1],q_qbY[i],q_qbY[i+1])
    

# Calculate the ending point for q_qb 
x_end1 = slope_pairs[0][0] + line_length * (cos(radians(line_angle1)))
y_end1 = slope_pairs[0][1] + line_length * (sin(radians(line_angle1)))

# Storing the plot points for vertical line for 2nd lobe
line3.append(slope_pairs[0])
line3.append([x_end1,y_end1])

# Calculate the ending point for q_qb 
x_end2 = slope_pairs[0][0] + line_length * (cos(radians(line_angle2)))
y_end2 = slope_pairs[0][1] + line_length * (sin(radians(line_angle2)))

# Storing the plot points for horizontal line for q_qb
line1.append(slope_pairs[0])
line1.append([x_end2,y_end2])

# Changing angle to plot next pair of lines
line_angle1= 90
line_angle2=0

for i in range(len(qb_qX)-1) :
    negSlope2(qb_qX[i],qb_qX[i+1],qb_qY[i],qb_qY[i+1])
    
m=30          # to calculate the points from the second lobe only (used to limit the calculated parameters)
pairs = []    # To store the points and difference x and y axis

# Iterate through all the points till we find the closest to square 
for i in range(int(m)) :
    slope_pairs[1] = [qb_qX[20+i],qb_qY[20+i]]
   
    # Calculate and plot end point for first line
    x_end = slope_pairs[1][0] + line_length * (cos(radians(line_angle1)))
    y_end = slope_pairs[1][1] + line_length * (sin(radians(line_angle1)))

    line2.append(slope_pairs[1])
    line2.append([x_end,y_end])

    x_end = slope_pairs[1][0] + line_length * (cos(radians(line_angle2)))
    y_end = slope_pairs[1][1] + line_length * (sin(radians(line_angle2)))

    line4.append(slope_pairs[1])
    line4.append([x_end,y_end])

    intersection1 = find_intersection(line1,line2)
    intersection1 = Point(*intersection1)
    distance1 = intersection1.distance(slope_pairs[0])
    distance1 = distance1.evalf() 


    intersection2 = find_intersection(line3,line4)
    intersection2 = Point(*intersection2)
    distance2 = intersection2.distance(slope_pairs[0])
    distance2 = distance2.evalf() 

    pairs.append([line2,line4,abs(distance1-distance2),distance2])
    
    line2 = []
    line4 = []

# Ascending Sorting according to least difference between horizontal and vertical side of the square
pairs.sort(key=lambda x: x[2])

######################################################################################################################

# To calculate noise margin from 1st lobe

# Calculate and store only points with -1 slope
slope_threshold = 0.2    # How close we want the slope to be to -1 slope
slope_pairs = [] 

def negSlope1(pointX_1,pointX_2,pointY_1,pointY_2) :
    if (pointX_2 < 0.734) and (pointY_2 > 0.752) :
        derivativeX = pointX_1-pointX_2
        derivativeY = pointY_1-pointY_2
        slope =derivativeX/derivativeY
        if abs(slope+1) <= slope_threshold :
            slope_pairs.append([pointX_2,pointY_2])
########################################################################################
line1 = []   # Emptying the previous stored line segment points
line3 = []   # Emptying the previous stored line segment points
line2 = []   # Emptying the previous stored line segment points
line4 = []   # Emptying the previous stored line segment points
#line_length = 1
line_angle1 = 270
line_angle2=180

for i in range(len(qb_qX)-1) :
    negSlope1(qb_qX[i],qb_qX[i+1],qb_qY[i],qb_qY[i+1])


# Calculate the ending point for qb_qX 
x_end3 = slope_pairs[0][0] + line_length * (cos(radians(line_angle1)))
y_end3 = slope_pairs[0][1] + line_length * (sin(radians(line_angle1)))

# Storing the plot points for vertical line for 1st lobe
line3.append(slope_pairs[0])
line3.append([x_end3,y_end3])



# Calculate the ending point for qb_qX 
x_end4 = slope_pairs[0][0] + line_length * (cos(radians(line_angle2)))
y_end4 = slope_pairs[0][1] + line_length * (sin(radians(line_angle2)))

# Storing the plot points for horizontal line for 1st lobe
line1.append(slope_pairs[0])
line1.append([x_end4,y_end4])

# Overwriting the angles for next set of line segments
line_angle1= 90
line_angle2=0

n = 0  # For iterating through points (Used to find the number of points in the 1st lobe)
for i in range(len(qb_qX)-1) :
    negSlope1(q_qbX[i],q_qbX[i+1],q_qbY[i],q_qbY[i+1])
    n = n+1
    
pairs1 = []  # To store the points and difference x and y axis
# Iterate through all the points till we find the closest to square 
for i in range(int(n/2)) :
    slope_pairs[1] = [q_qbX[n-i],q_qbY[n-i]]
   
    # Calculate and plot end point for first line
    x_end = slope_pairs[1][0] + line_length * (cos(radians(line_angle1)))
    y_end = slope_pairs[1][1] + line_length * (sin(radians(line_angle1)))

    line2.append(slope_pairs[1])
    line2.append([x_end,y_end])

    x_end = slope_pairs[1][0] + line_length * (cos(radians(line_angle2)))
    y_end = slope_pairs[1][1] + line_length * (sin(radians(line_angle2)))

    line4.append(slope_pairs[1])
    line4.append([x_end,y_end])

    intersection1 = find_intersection(line1,line2)
    intersection1 = Point(*intersection1)
    distance1 = intersection1.distance(slope_pairs[0])
    distance1 = distance1.evalf() 


    intersection2 = find_intersection(line3,line4)
    intersection2 = Point(*intersection2)
    distance2 = intersection2.distance(slope_pairs[0])
    distance2 = distance2.evalf() 

    pairs1.append([line2,line4,abs(distance1-distance2),distance2])
    
    if (i==int(n/2)) :
        break
    line2 = []
    line4 = []

# Sorting according to least difference between horizontal and vertical side of the square
pairs1.sort(key=lambda x: x[2])

print(pairs1[0])
# Calculating the distance between the horizontal lines of the squares
#distance_1lobe = math.dist([pairs1[0][1][0][0],pairs1[0][1][0][1]],[pairs1[0][1][1][0],pairs1[0][1][1][1]])
#distance_lobe = math.dist([pairs[0][1][0][0],pairs[0][1][0][1]],[pairs[0][1][1][0],pairs[0][1][1][1]])
#distance_lobe = math.dist(pairs[0][1][0][0],pairs[0][1][0][1])

print(pairs1[0][3])
print(pairs[0][3])

distance_1lobe = pairs1[0][3]
distance_lobe = pairs[0][3]

# Comparing and plotting the square with the lowest side length
if (distance_lobe < distance_1lobe) :
    # For plotting points 1st half
    x_axis_ver=pairs1[0][0][0][0]
    y_axis_ver=pairs1[0][0][0][1]
    x_axis_ver_end=pairs1[0][0][1][0]
    y_axis_ver_end=pairs1[0][0][1][1]

    x_axis_hor=pairs1[0][1][0][0]
    y_axis_hor=pairs1[0][1][0][1]
    x_axis_hor_end=pairs1[0][1][1][0]
    y_axis_hor_end=pairs1[0][1][1][1]

    # PLotting Square    
    plt.plot([x_axis_ver,x_axis_ver_end], [y_axis_ver,y_axis_ver_end], marker="o")
    plt.plot([x_axis_hor,x_axis_hor_end], [y_axis_hor,y_axis_hor_end], marker="o")
    plt.plot([slope_pairs[0][0], x_end4], [slope_pairs[0][1], y_end4], marker="o")    
    plt.plot([slope_pairs[0][0], x_end3], [slope_pairs[0][1], y_end3], marker="o")
    label_x = (x_axis_hor + x_axis_hor_end) / 4
    label_y = (y_axis_hor + y_axis_hor_end) / 2
    distance_lobe = "{:.2f}".format(distance_lobe)
    # Add the distance label above the line segment
    plt.text(label_x, label_y, f'dx={distance_lobe}', ha='center', va='bottom')
else :
    x_axis_ver=pairs[0][0][0][0]
    y_axis_ver=pairs[0][0][0][1]
    x_axis_ver_end=pairs[0][0][1][0]
    y_axis_ver_end=pairs[0][0][1][1]

    x_axis_hor=pairs[0][1][0][0]
    y_axis_hor=pairs[0][1][0][1]
    x_axis_hor_end=pairs[0][1][1][0]
    y_axis_hor_end=pairs[0][1][1][1]

    # PLotting Square    
    plt.plot([x_axis_ver,x_axis_ver_end], [y_axis_ver,y_axis_ver_end], marker="o")
    plt.plot([x_axis_hor,x_axis_hor_end], [y_axis_hor,y_axis_hor_end], marker="o")
    plt.plot([slope_pairs[0][0], x_end4], [slope_pairs[0][1], y_end4], marker="o")    
    plt.plot([slope_pairs[0][0], x_end3], [slope_pairs[0][1], y_end3], marker="o")
    label_x = (x_axis_hor + x_axis_hor_end) / 4
    label_y = (y_axis_hor + y_axis_hor_end) / 2
    distance_1lobe = "{:.2f}".format(distance_1lobe)
    # Add the distance label above the line segment
    plt.text(label_x, label_y, f'dx={distance_1lobe}', ha='center', va='bottom')
    
    
    
# Plotting the butterfly output for DC sweep output for both q vs qb and qb vs q

#plt.scatter(q_qbX, q_qbY, marker="o")
#plt.scatter(qb_qX, qb_qY, marker="o")
plt.plot(q_qbX, q_qbY, qb_qX, qb_qY)


plt.xlabel("in V ------------------------->")
plt.gca().set_aspect('equal')   # Sets the graph's aspect ratio to 1:1
plt.title("Butterfly graph for finding Static Noise Margin for 6-T SRAM")
plt.show()