# 用户和组

## 用户

Linux系统是一个多用户系统。同一时间可以有多个用户登录这一系统，互不干扰的执行程序。用户分为超级用户，普通用户和伪用户。

## 组

用户组是具有相同特征用户的逻辑集合。将用户分组是Linux 系统中对用户进行管理及控制访问权限的一种手段，通过定义用户组，在很大程度上简化了管理工作。

## 相关配置文件

*  `/etc/passwd`：系统用户配置文件，是用户管理中最重要的一个文件。这个文件记录了Linux系统中每个用户的一些基本属性，并且对所有用户可读。
* `/etc/shadow`：包含用户的密码信息，只对`root`用户可读。
* `/etc/group`：包含用户组的配置。
* `/etc/login.defs`：用来定义创建一个用户时的默认设置。
* `/etc/skel`：定义了新建用户在主目录下默认的配置文件，如`.bashrc`等。



# 权限

Linux的文件权限分为读，写和可执行。

利用`ls -l`指令可以列出文件夹下所有文件的权限信息，对于每行前十个字符，其含义如下：

* 第一个：-（普通文件），d（目录文件），l（符号链接），c（字符设备文件），b（块设备文件）。
* 剩下的九个三个三个为一组，分别为：所有者权限，组权限（所有者所在的组中的其他用户），其他用户权限。同一组中，三个字符按顺序分别代表读（r），写（w）和可执行（x），若为-，则代表无该权限。

# 文件权限管理

## `chmod`--改变文件权限

只有文件所有者和`root`用户才可以更改对应文件的权限。

### 八进制法

首先，每个文件对所有者，用户组和其他用户的权限都可以用三个字母表示（rwx），这三个字母有8种不同组合，从而产生了八进制表示法来表示权限：

| 八进制 | 二进制 | 文件权限 |
| ------ | ------ | -------- |
| 0      | 000    | ---      |
| 1      | 001    | --x      |
| 2      | 010    | -w-      |
| 3      | 011    | -wx      |
| 4      | 100    | r--      |
| 5      | 101    | r-x      |
| 6      | 110    | rw-      |
| 7      | 111    | rwx      |

于是，我们可以通过三个数字来指定所有者，用户组和其他用户的权限，例如：

```shell
chmod 777 ./file
```

### 字母表示法

字母表示法是用“权限改变对象+如何改变+改变哪种权限”的方式来改变文件权限的。其中，权限改变对象的表示方法如下（不指定对象默认为a）：

| 符号 | 含义                         |
| ---- | ---------------------------- |
| u    | 文件所有者                   |
| g    | 文件所有者所在的组           |
| o    | 其他用户（除所有者和所有组） |
| a    | 所有用户，即u+g+o            |

改变分为三种：+（添加这种权限），-（删除权限），=（只指定这一种权限）。

例子：

* `chmod u+x ./file`：为所有者增加文件执行权。
* `chmod o-rw`：删除其他用户的读写权。

## `chown`--改变文件所有者（和所属组）

```shell
chown [user]：[group] file
```

该命令必须有管理员权限才可以执行。

可以分别修改拥有者和所属组。

# 以另外一个成员的身份执行程序

* `su`：以其他用户的身份重新进入shell（会加载新成员的配置文件）。

    `su -l user`：以user的身份进入shell，需要其密码，如果user忽略（-l也可忽略），则默认进入root用户。

    `su -c 'command'`：以超级用户的身份执行命令，这里该命令被传入新的shell环境（不同于sudo）。

* `sudo`：该命令可以使普通用户以root用户的身份执行命令（这里该命令没有传入新的shell环境），需要管理员密码。



