# lanoising

[![License](https://img.shields.io/badge/License-BSD%203--Clause-gree.svg)](https://opensource.org/licenses/BSD-3-Clause)

Noising the point cloud.

The package is tested in Ubuntu 16.04, ROS kinetic 1.12.14, Python 3.6.

Requirements:

numpy 1.17.2

scikit-learn 0.23.1

tensorflow 1.14.0

keras 2.2.4

Anaconda3 recommended. With adaconda3, only tensorflow and keras need to be installed.

To make ROS and Anaconda3 compatible, in a new terminal:

`gedit ~/.bashrc`

add: `source /opt/ros/kinetic/setup.bash`

delete: `export PATH="/home/tyang/anaconda3/bin:$PATH"`

`source ~/.bashrc`

before launch the package:

`export PATH="/home/tyang/anaconda3/bin:$PATH"`

example:

download the lanoising package and decompress in ./src of your catkin workspace (e.g. catkin_ws).

in a new terminal:

```
cd ./catkin_ws

catkin_make
```

download the models and put all the files in ./catkin_ws/src/lanoising/models:

https://drive.google.com/file/d/1CoVrr3dVQ5DY4WpF7xCM9z6Vx7PYKW1w/view?usp=sharing

in the terminal:

`roscore`

in a new terminal:

`rviz`

play the reference rosbag (point clouds recorded by velodyne LiDAR under clear weather conditions):

`rosbag play -l --clock 2019-02-19-17-13-37.bag`

in rviz, change Fixed frame to "velodyne".

add the topic "/velodyne_points" in rviz to show the reference data.

set the visibility in lanoising.py.

in a new terminal:

```
cd ./catkin_ws

source devel/setup.bash

export PATH="/home/tyang/anaconda3/bin:$PATH"

roslaunch lanoising lanoising.launch
```

add the topic "/filtered_points" in rviz to show the noising point cloud.

## Citation
If you publish work based on, or using, this code, we would appreciate citations to the following:

    @artical{yt20iros,
        author       = {Tao Yang, You Li, Yassine Ruichek, and Zhi Yan}},
        title        = {LaNoising: A Data-driven Approach for 903nm ToF LiDAR Performance Modeling under Fog},
        booktitle    = {In Proceedings of the 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
        month        = {October},
        year         = {2020},
        }

