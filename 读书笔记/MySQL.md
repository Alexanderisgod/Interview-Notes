## MYSQL



## 一. InnoDB存储引擎

1. 关键特性：

   - 插入缓冲
   - 两次写， double write
   - 自适应哈希索引（adaptive hash index）

   

2. 联合索引：对表上的多个列做索引。对多值判断时，采用联合索引；联合索引可以对第二个键值进行排序。

   

3. MySQL锁的类型：

   - 共享锁：允许事务读一行数据。
   - 排它锁：允许事务删除或更新一行数据。
   - 多粒度锁：允许在行级上的锁和表级上的锁同时存在。
   - 意向锁是表级别的锁。
   - 意向共享锁：事务想要获得一个表中某几行的共享锁；意向排它锁：获得一个表中某几行的排它锁。

   

4. SQL的命令执行顺序

   ```sql
   (8) SELECT (9)DISTINCT<Select_list>
   (1) FROM <left_table> (3) <join_type>JOIN<right_table>
   (2) ON<join_condition>
   (4) WHERE<where_condition>
   (5) GROUP BY<group_by_list>
   (6) WITH {CUBE|ROLLUP}
   (7) HAVING<having_condtion>
   (10) ORDER BY<order_by_list>
   (11) LIMIT<limit_number>
   ```

   查询从FROM开始执行，执行过程中，每个步骤都会为下一步骤生成一个虚拟表，作为下个步骤的输入。

   1. 对FROM的子句中的两个表执行一个笛卡尔乘积，生成virtual table 1。

   2. 采用ON条件筛选器，将ON中的逻辑表达式应用到 virtual table 1的各个行，筛选出满足ON中的逻辑表达式的行，生成virtual table2。

   3. 根据连接方式进一步操作，得到virtual table3。

      

   4. 采用WHERE筛选器，对virtual table3进行筛选，生成virtual table4。

      ==ON和WHERE==的区别在于：ON应用逻辑表达式，在第三步OUTER JOIn中还可以把移除的行重新添加回来，而WHERE移除是不可挽回。

      

   5. GROUP BY将 子句中相同属性的row合成一组，得到virtual table 5。

   6. 应用HAVING筛选器，生成virtual table 7。HAVING筛选器是唯一一个用来筛选组的筛选器。

   

   

5. 