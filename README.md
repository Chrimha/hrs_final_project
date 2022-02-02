# hrs_final_project

## Instructions to run project Blind Road Guider

Software:

- Change the IP of the robot in line 41 of the main.py file and system variables

- Start the robot using the command 
```
roslaunch nao_bringup nao_full_py.launch
```

- Activate tactile interactions using
```
roslaunch nao_apps tactile.launch
```
- Start RViz using command 
```
rosrun rviz rviz
```
- Run package using command 
```
rosrun hrs_final_project main.py
```

Hardware / Room:

- Position ArUco marker with the ID 5 in front of the robot, preferably in the center of the room.
- Place obstacles on the field with any ArUco marker but ID no. 5
- Place traffic light on the desired goal<br/>  

Note: Original ArUco Markers were used