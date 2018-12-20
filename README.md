# EVT
使用极端值理论(Extreme Value Theory)实现阈值动态自动化设置
# 介绍
我们的工作建立在2017 KDD "Anomaly Detection in Streams with Extreme Value Theory"论文的基础上，做了如下改进：
* 引入矩估计算法，加速计算。该算法比极大似然估计快100多倍
* 提出了更高层，更抽象的基于预测残差的算法框架，而DSPOT算法是我们提出框架的一种具体算法
* 我们强调了数据漂移对系统带来的影响，提出了批量更新的算法，有效应对数据漂移
# 应用
* 异常探测问题中，经常需要设置阈值，例如：内存的使用率大于90%时，判定为异常。这里阈值90%是人为设定的，需要用户有足够的使用经验，而且这种设定方式随机性很大，比如设置为89%或者91%似乎也是合理的。
* 现实应用中，每条KPI都需要手动设置不同的阈值，这是一项十分复杂和庞大的工作，如果我们能够只设定概率值q而无需设定阈值，那么会免除巨大的工作量。
![应用实例](https://github.com/DawnsonLi/EVT/blob/master/pic/1.png)
* 使用我们的方法只需定义异常事件发生的概率，而无需设置成百上千的阈值，以不变应万变
![应用实例](https://github.com/DawnsonLi/EVT/blob/master/pic/2.png)
* 使用示例:
这里给出一个应对数据漂移的算法运行结果示意图，上下黄色虚线分别对应算法自动设置的上下阈值。
![应用实例](https://github.com/DawnsonLi/EVT/blob/master/pic/middle_3.png)



