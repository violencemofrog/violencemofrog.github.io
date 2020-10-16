# 数据库关系模型

* 在关系数据库中，表中的每一行称为记录，每一列称为字段。

* 对于关系数据表，任意两条记录不能重复，要能够通过某个字段唯一区分出不同的记录，这个字段被称为主键，插入相同主键的两条记录是不被允许的。

* 关系数据库可以有两个或更多的字段都设置为主键，这种主键被称为联合主键。对于联合主键，允许一列有重复，只要不是所有主键列都重复即可。

* 在一张表中，通过某个字段，把数据和另一张表关联起来，这个字段成为外键。通过外键可实现“一对多”关系。

* 通过中间表可实现“多对多”关系。

* 通过对数据库表创建索引，可以提高查询速度。

* 通过创建唯一索引，可以保证某一列的值具有唯一性。

# SQL语法

SQL命令不区分大小写（列名也对大小写不敏感，但是表名对大小写敏感），SQL语句的结尾通常为`;`

## `SELECT`（查询语句）

`SELECT`语句从表中选择数据，并将数据放在结果集中

### 一般的查询

`SELECT column_name FROM table_name;`

可以同时选择多个列：`SELECT column1，column2， ... FROM table_name;`，注意每一列间用`,`分隔开

也可以选择表的全部列：`SELECT * FROM table_name;`

显示列时也可指定别名：`SELECT column1 alias1,column2 alias2 FROM table_name;`

### 返回唯一值

`SELECT DISTINCT column_name FROM table_name;`，其结果集中将不会出现重复的记录（即每行都不同，但是可能某一列有相同值）

### 条件查询

```sql
SELECT column_name FROM table_name
WHERE column_name operator value
```

下面是一个具体实例（从websites表中查询country列中数值为'CN'的记录）：

```sql
SELECT * FROM Websites WHERE country='CN';
```

| 运算符 | 描述     |
| ------ | -------- |
| =      | 相等     |
| >      | 大于     |
| <      | 小于     |
| \>=    | 大于等于 |
| <=     | 小于等于 |
| <>     | 不等     |
|        |          |

也有一些特殊的条件判断：

* 空值判断：`SELECT * FROM table WHERE comm IS NULL;`

* 范围判断：`SELECT * FROM table WHERE comm BETWEEN value1 AND value2`（下限在前，上限在后）

* 匹配多个值：`SELECT * FROM table WHERE comm IN (value1,value2,.....)`

* 模糊查询：`SELECT * FROM table WHERE comm LIKE pattern`

    | 通配符  | 描述                   |
    | ------- | ---------------------- |
    | %       | 代替0或多个字符        |
    | _       | 代替一个字符           |
    | [char]  | 字符列中的任一单个字符 |
    | [^char] | 不在字符列中的任一字符 |
    |         |                        |

可以在`WHERE`子句中结合逻辑连接词：`AND`  `OR`  `NOT`和括号

### 排序

使用`ORDER BY`对查询的结果可以进行排序

```
SELECT * FROM table_name
ORDER BY column_name ASC/DESC;
```

对查询结果根据某一列的数据进行排序，默认参数`ACS`升序，若为`DESC`，则是降序

可以同时添加多个列，如果前一列数据有相同的值，则依次按后面列的元素排序

`DESC`和`ASC`仅对它前面的那一列生效，如`ORDER BY <列1> DESC,<列2> ASC`

### 返回规定数目的记录

```mysql
SELECT column_name(s)
FROM table_name
LIMIT number
```

不同的SQL服务器的命令是不同的，每一个SQL服务器都有自己的语法拓展

## `INSERT`（插入数据）

```sql
INSERT INTO table_name
VALUES (value1,value2,value3,...);
```
这种方式，要插入的值必须按顺序依次写入

```sql
INSERT INTO table_name (column1,column2,column3,...)
VALUES (value1,value2,value3,...);
```
第二种方式，可以不按顺序，但是值和列名要匹配

## `UPDATE`（更新数据）

```sql
UPDATE table_name
SET column1=value1,column2=value2......
WHERE ...
```

更新`WHERE`子句匹配的记录的数据，如果没有该子句，则更新所有记录

## `DELETE` （删除记录）

`DELETE FROM table_name WHERE ...;`

删除`WHERE`子句匹配的记录，如没有该子句，则删除所有记录

##  `JOIN`（连接）

### `INNER JOIN`

![](./images/innerj.png)

```sql
SELECT column_name(s)
FROM table1
INNER JOIN table2
ON table1.column_name=table2.column_name;
```

当`ON`的条件成立时，返回所选取的列中对应的记录。table1为主表。`INNER JOIN`	等价于`JOIN`

### `LEFT JOIN`和`RIGHT JOIN`

![](./images/leftj.png)

![](./images/rightj.png)

```sql
SELECT column_name(s)
FROM table1
LEFT JOIN table2
ON table1.column_name=table2.column_name;
```

```sql
SELECT column_name(s)
FROM table1
RIGHT JOIN table2
ON table1.column_name=table2.column_name;
```

两条指令类似，都是从表中返回匹配行，但是即使右/左表中没有匹配，也会返回该记录，没有匹配的字段为NULL

### `FULL OUTER JOIN`

![](./images/fullj.png)

```sql
SELECT column_name(s)
FROM table1
FULL OUTER JOIN table2
ON table1.column_name=table2.column_name;
```

左表和右表只要有一项匹配，就返回该记录

## `UNION`（合并查询结果）

```sql
SELECT column_name(s) FROM table1
UNION
SELECT column_name(s) FROM table2;
```

该命令用于合并查询结果（类似合集，且集合中元素不重复），两条`SELECT`语句必须有相同数量的列，而且对应列的数据类型要一致，列的顺序也要一致（一一对应）

显示的结果的列名为前一个`SELECT`语句中的列名

```sql
SELECT column_name(s) FROM table1
UNION ALL
SELECT column_name(s) FROM table2;
```

`UNION ALL`命令所合并的结果会有重复值

可以在每一条`SELECT`语句后加`WHERE`子句，但是`ORDER BY`只能先合并再排序（即只能对合并结果排序）

## `SELECT INTO/INSERT INTO SELECT`（复制表的记录）

```sql
SELECT column_name(s)
INTO newtable 
FROM table1;
```

该命令要求新表不存在，该命令会自动创建一个新表，同时可以指定新的列名；注意MySql不支持该语法

```sql
INSERT INTO table2
(column_name(s))
SELECT column_name(s)
FROM table1;
```



# 数据库操作

## 新建数据库和表

`CREATE DATABASE new_database_name`

```sql
CREATE TABLE new_table_name
(
column_name1 data_type(size),
column_name2 data_type(size),
column_name3 data_type(size),
....
);
```

* column_name：列名
* data_type：该列的数据类型
* size：该数据的最大位数

## 添加约束关系

SQL约束关系用于为表中的数据添加规则，如果有违反规则的操作，则终止该行为

### 在创建表时添加约束

```sql
CREATE TABLE table_name
(
column_name1 data_type(size) constraint_name, --约束名称
column_name2 data_type(size) constraint_name,
column_name3 data_type(size) constraint_name,
....
);
```

* `NOT NULL`：规定该字段不接受空值
* `UNIQUE`：保证该字段的值的唯一性，可同时指定多个列的联合为`UNIQUE`
* `PRIMARY KEY`：主键；主键必须包含唯一的值；主键列不能包含 NULL 值；每个表都应该有一个主键，并且每个表只能有一个主键，可同时指定多个列联合为`PRIMARY KEY`
* `FOREIGN KEY`：外键；一个表中的 FOREIGN KEY 指向另一个表中的 UNIQUE KEY(唯一约束的键)
* `CHECK`：用于规定字段的范围
* `DEFAULT`：用于向某一列插入默认值

实例分析：

```mysql
CREATE TABLE table1
(
    ID int NOT NULL;
    value1 int(255) NOT NULL, #int型，最大255个元素，不能为空
    value2 int(255) UNIQUE,
    value3 int(255),
    value4 int(255),
    CONSTRAINT uc_V UNIQUE(value3,value4), #多个列的UNIQUE约束
    PRIMARY KEY (ID), #定义ID为主键
    value5 int(255),
    CHECK (value5>100), #定义value5必须大于100
    value6 int(255) DEFAULT 100 #定义value6默认为100
    value7 int(255),
    FOREIGN KEY (value7) REFERENCES teble2(P_ID) #定义value7为外键，指向表table2的P_ID列
);
```

## `CREATE INDEX`（创建索引）

```sql
CREATE UNIQUE INDEX index_name --不加UNIQUE，则创建的索引可以有相同的值
ON table_name (column_name);
```

## `DROP`（删除索引，表或数据库）

```sql
DROP TABLE table_name;
```

```sql
DROP DATABASE base_name;
```

```mysql
ALTER TABLE table_name DROP INDEX index_name;
```

仅清空表:

```sql
TRUNCATE TABLE table_name
```

## `ALTER TABLE`（修改列）

```sql
ALTER TABLE table_name   --添加一个新列
ADD column_name datatype;
```

```sql
ALTER TABLE table_name  --删除列 
DROP COLUMN column_name;
```

```sql
ALTER TABLE table_name     --更新列的数据类型或者列名
MODIFY column_name datatype;
```



# 参考

* [菜鸟教程](https://www.runoob.com/sql/sql-tutorial.html)
* [廖雪峰的博客](https://www.liaoxuefeng.com/wiki/1177760294764384) 