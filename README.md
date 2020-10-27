# lanoising

Noising the point cloud.

The package is tested in Ubuntu 16.04, ROS kinetic 1.12.14, Python 3.6.

Requirements:

numpy 1.17.2

scikit-learn 0.23.1

tensorflow 1.14.0

keras 2.2.4


Anaconda3 recommended. With adaconda3, only tensorflow and keras need to be installed.

To make ROS and Anaconda3 compatible:

`gedit ~/.bashrc`

add: `source /opt/ros/kinetic/setup.bash`

delete: `export PATH="/home/tyang/anaconda3/bin:$PATH"`

`source ~/.bashrc`

before launch the package:

`export PATH="/home/tyang/anaconda3/bin:$PATH"`


example:

download the models and put all the files in ./models:

https://drive.google.com/file/d/1CoVrr3dVQ5DY4WpF7xCM9z6Vx7PYKW1w/view?usp=sharing


download the lanoising package and decompress in your catkin workspace:

`catkin_make`


`roscore`

`rviz rviz`

play the reference bag, in which human model is placed at 15m from sensor:

`rosbag play -l --clock 2019-02-19-17-13-37.bag`

in rviz, change Fixed frame to velodyne

add topic /velodyne_points to show the reference data

set the visibility in lanoising.py


in catkin workspace:

`source devel/setup.bash`

`export PATH="/home/tyang/anaconda3/bin:$PATH"`

`roslaunch lanoising lanoising.launch`

add topic /filtered_points to show the noising point cloud


## Citation
If you publish work based on, or using, this dataset, we would appreciate citations to the following:

    @artical{taoy2020,
        author       = {Tao Yang, You Li, Yassine Ruichek, and Zhi Yan}},
        title        = {LaNoising: A Data-driven Approach for 903nm ToF LiDAR Performance Modeling under Fog},
        conference      = {IROS},
        year         = 2020,
        }

