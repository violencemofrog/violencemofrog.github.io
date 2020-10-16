> 参考ROS官网新手教程

# ROS包相关指令

* `rospkg list`：列出系统中安装的所有ROS包
* `rospack find pkg`：输出pkg包的位置
* `roscd pkg`：移动到pkg包的文件夹下
* `rosls` pkg：显示pkg包的文件
* `rosed pkg file`：编辑pkg包下的file文件
* `rosdep install pkg`：安装pkg包的所需依赖
* `catkin_create_pkg pkg dependa dependb dependc ...`：创建pkg包（后面指出该包的依赖），在工作空间下的src文件夹下执行该命令
* `rospack depends(n) pkg`：查看pkg包的n阶依赖（不指定则默认所有依赖）
* `roscore`：启动ROS master
* `rosrun pkg_name node_name`：运行pkg包中的node节点（一个包可以包含多个ROS节点）
* `roslaunch pkg file.launch`：启动ROS master和多个node（pkg包内的.launch文件可以用于启动该包内的多个node）

# ROS通信概念

* 节点（node）：就是通过ROS包中的代码启动的一个程序
* 主题（topic）：节点间通信的一种方式，一些节点可以向主题发布消息，另一些节点可以订阅这些消息，从而完成节点间的通信
* 消息（message）：节点向主题发布的数据的数据类型称为消息
* 服务（service）：节点间通信的一种方式
* 主节点（master）：登记注册节点、服务和话题的名称，并维护一个参数服务器
* 服务和主题的区别：

| topic                  | service              |
| ---------------------- | -------------------- |
| 多对多                 | 多对一               |
| 适用于连续，高频的数据 | 适用于偶尔调用的功能 |

# Node相关命令

* `rosnode list`：列出当前运行的ROS节点
* `rosnode info /node_running`：显示某个运行的节点的信息
* `rosnode kill /node_running`：结束某个运行的节点
* `rosnode ping  /node_running`：显示某个运行节点的通信信息

# Topic相关命令

* `rostopic list -v`：显示当前发布和订阅的主题
* `rostopic echo topic_name`：显示topic发布的数据
* `rostopic type topic_name`：查看主题发布的消息的类型
* `rostopic pub topic_name msg_type args`：使某一主题发布指定数据的消息

# Message相关命令

* `rosmsg list`：列出系统上当前的所有消息
* `rosmsg show msg_name`：显示某一消息的结构

# 服务相关命令

* `rosservice list`：显示当前所有服务
* `rosservice type srv_name`：显示服务类型
* `rosservice call srv_name`：调用某个服务
* `rossrv show srv_type`：显示某服务类型的参数和返回值

# 参数服务器相关命令

* `rosparam set param_name arg`：设置某个参数的值
* `rosparam get param_name`：获取某个参数的值（param_name为/时，获取所有参数的值）

# RQT显示ROS图信息

* `rosrun rqt_graph rqt_graph`：创建ROS系统动态结构图（可显示node和topic的关系结构）
* `rosrun rqt_console rqt_console`：显示节点输出
* `rosrun rqt_logger_level rqt_logger_level`：调整日志级别