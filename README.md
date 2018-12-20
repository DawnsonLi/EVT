# EVT
使用极端值理论(Extreme Value Theory)实现阈值动态自动化设置
## 我们的工作建立在2017 KDD "Anomaly Detection in Streams with Extreme Value Theory"论文的基础上，做了如下改进：
* 引入矩估计算法，加速计算。该算法比极大似然估计快100多倍
* 提出了更高层，更抽象的基于预测残差的算法框架，而DSPOT算法是我们提出框架的一种具体算法
* 我们强调了数据漂移对系统带来的影响，提出了批量更新的算法，有效应对数据漂移

