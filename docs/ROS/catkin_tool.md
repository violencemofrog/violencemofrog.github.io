# catkin_tool

catkin_tool是一个Python包，用于代替传统的ROS包构建工具

# catkin_tool工作流程

``` shell
mkdir  -p ~/catkin_ws/src   #创建工作空间和源码空间
cd ~/catkin_ws
catkin init                 #初始化工作空间
cd ./src
catkin create pkg pkg_name --catkin-deps dep_name1 dep_name2                   #在源码空间下创建ROS包
catkin build                #生成工作空间下所有ROS包
source ~/catkin_ws/devel/setup.*sh #将该工作空间载入环境变量
catkin clean                #清除所有生成文件（包括build和devel文件夹的文件）
```

