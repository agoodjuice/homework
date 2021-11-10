# homework
人工智能安全作业
​	该仓库是基于Pytorch框架以及CIFAR 10训练集下构建的一个基础神经网络模型，主要是通过调整超参数来观察各个超参数的作用，以及超参数对训练所产生的不同影响

## 网络架构

​	为了更显著的观察各个超参的作用，所以构建一个具有2层卷积、2层池化、3层全连接的简单网络

​	超参基本设置

- Learing Rate = 0.01

- Epoch = 10

- Batch Size = 64

- Optimizer：SGD

- Criterion：CrossEntropyLoss

  每个EPOCH的loss
![image](https://github.com/agoodjuice/homework/blob/master/pic/raw_loss.jpg)

​	在2.3附近一直浮动，没有下降

​	最后在验证集上的准确率：`12.2100%` 

## 调整超参

### 调整EPOCH

> EPOCH: 通俗的说，就是将所有训练样本训练一次的结果

​	通过观察最开始的loss和acc，可以得出该模型没有学到什么东西...，考虑增加EPOCH，让其继续学习，这里把EPOCH设置成50

![image](https://github.com/agoodjuice/homework/blob/master/pic/epoch_50.jpg)

​	可以看到经过一段时间后，loss开始有了明显下降最低能达到`1.6`，并且acc相较一开始也有了显著提高，能够达到`39%`。说明增大EPOCH能提高我们模型的拟合程度

​	EPOCH=100

![image](https://github.com/agoodjuice/homework/blob/master/pic/epoch_100.jpg)

​	loss能够继续下降能够达到`1.4`，acc最终能够达到`47.1%`

​	EPOCH = 200

![image](https://github.com/agoodjuice/homework/blob/master/pic/epoch_200.jpg)

​	EPOCH = 350 

![image](https://github.com/agoodjuice/homework/blob/master/pic/epoch_350.jpg)

汇总如下

| EPOCH | LOSS |  ACC   |
| :---: | :--: | :----: |
|  10   | 2.3  | 12.21% |
|  50   | 1.6  |  39%   |
|  100  | 1.4  | 47.13% |
|  200  | 1.14 | 55.9%  |
|  350  | 0.88 | 59.1%  |

总结：

​	可以看出随着EPOCH的增加，我们的模型效果是在不断变好的，但是我们的EPOCH也不能设置的太大，因为训练过多的话，反而会产生过拟合，使得我们的泛化能力变差，像后面增加的EPOCH所带来的的ACC提升就没有一开始的明显	

### 调整优化器Optimizer

​	一开始使用的最普通的梯度下降法SGD，其中没有动量的概念，最大的缺点是下降速度慢，而且可能会在沟壑的两边持续震荡，停留在一个局部最优点。

​	为了抑制SGD的震荡，我们可以在梯度下降过程可以加入惯性，因此就有人提出了**SGD with Momentum**（SGDM）。后序又有人提出AdaGrad算法，独立地适应所有模型参数的学习率。

​	在上述调整EPOCH大小的时候，我们观察loss曲线可以看出虽然总体loss是在下降的，但是在下降的过程中上下波动不小，所以我们对普通的SGD进行替换，使用Adam，Adam其实就是前面各种优化器思想的集大成者，在SGD基础上增加了一阶动量和二阶动量。

| EPOCH | LOSS |  ACC   |
| :---: | :--: | :----: |
|  50   | 0.50 | 59.59% |
|  100  | 0.39 | 58.85% |

在只有训练50个EPOCH的时候，可以明显看到，使用Adam优化器的loss和acc有了巨大提高，但是当EPOCH设置100的时候，虽然loss有了下降，但是acc缺没有提高，说明模型可能已经产生了过拟合。



### 调整学习率LR

​	接下来再在Adam的基础上，调整学习率，看看有什么变化

> 学习率： 习率是指导我们，在梯度下降法中，如何使用损失函数的梯度调整网络权重的超参数。学习率如果过大，可能会使损失函数直接越过全局最优点，学习率如果过小，损失函数的变化速度很慢，会大大增加网络的收敛复杂度，并且很容易被困在局部最小值或者鞍点

LR = 0.001

| EPOCH | LOSS |  ACC   |
| :---: | :--: | :----: |
|  50   | 1.11 | 58.58% |
|  100  | 0.87 | 63.05% |

可以上面的结果进行比较，可以看到训练100个EPOCH时的数据还是有明显提升的，可见学习率调小后，说明模型收敛的速度在减慢，减缓了发生过拟合的速度





## 总结

各个不同的超参数之间会对模型的训练产生不同的影响 ，不能一味的叠加。需要我们根据模型的训练过程以及数据来进行一个动态调整

