# -*- coding: utf-8 -*-
"""
Get the coordinate of mouse and convert the screen metric 
to the window size metric according to PC's size, and the
obtained data is then fed to the main ipynb file as input
data.

@author: Riemann Lee
"""
import os, time
import numpy as np
import pyautogui as pag


Trajectory = []
delta_t = 0.4


try:
    while True:
        print("Press Ctrl-C to end")
        screenWidth, screenHeight = pag.size()  # get our screen size
        x, y = pag.position()  # Get the coordinates of moving mouse
        Trajectory.append([x,y])
        # Print the coordinate of mouse on the console
        print("Screen size: (%s %s),  Position : (%s, %s)\n" % (screenWidth, screenHeight, x, y))  
 
        time.sleep(delta_t)  # print the point of the screen every half second
        os.system('cls')  # clear the screen
except KeyboardInterrupt:
    print('end')
    
"""
Now we need to  further process the obtained data, namely,
calculation from mouse tragectory to the input data. I have:
    Screen Size: (1920, 1080)
    Upper Left Corner: [502, 193]
    Bottom Left Corner: [502, 917]
    Bottom Right Corner: [1467, 917]
    Upper Right Corner: [1467, 193]
    
The parameters list here is dependent on personal computer screen's
size, and I use these parameters to change test dataset.
"""
Width_Max = 800
Height_Max = 600

Screen_Size =  np.array([1920, 1080])
Upper_Left_Corner = np.array([502, 193])
Bottom_Right_Corner = np.array([1467, 917])
Window_Size_Screen = Bottom_Right_Corner - Upper_Left_Corner

Window_Size = np.array([Width_Max, Height_Max])
Scaling_Factor = Window_Size / Window_Size_Screen

# change the list to numpy array for the convenience of computation
Trajectory_Sequence = np.asarray(Trajectory)
Input_Sequence = (Trajectory_Sequence - Upper_Left_Corner) * Scaling_Factor
Output_Sequence = Input_Sequence[:-10]
# delete the last 10 element for sake of manual noise
# and this Trajectory data should not be given out
# give velocity (as transition?) and distance to landmarks (as observation?)
np.save('./data/PF_data/trajectory/example_trajectory.npy', Output_Sequence)

# Now we have got the robot's trajectory (type np.ndarray), now use
# this to derive velocity (with angle) and distance information
angle_and_velocity = []
distance = []

Centers = np.load('./data/PF_data/centers/example_centers.npy')
Radii = np.load('./data/PF_data/radii/example_radii.npy')

for i in range(len(Output_Sequence)-1):
    angle = np.arctan2(Output_Sequence[i+1, 1] - Output_Sequence[i, 1], Output_Sequence[i+1, 0] - Output_Sequence[i, 0])
    velocity = np.linalg.norm(Output_Sequence[i+1] - Output_Sequence[i]) / delta_t
    angle_and_velocity.append([angle, velocity])
    dis = np.linalg.norm(Output_Sequence[i] - Centers, axis=1, keepdims=True) - Radii
    distance.append(dis)
np.save('./data/PF_data/velocity/example_velocity.npy', np.asarray(angle_and_velocity))
np.save('./data/PF_data/distance/example_distance.npy', np.asarray(distance))