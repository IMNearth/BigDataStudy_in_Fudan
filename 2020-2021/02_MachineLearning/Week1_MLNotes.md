## Week 1

### 1 统计学习 statistical learning

根据数据构建概率统计模型，运用模型对数据进行预测及分析。要素：（1）数据：能够被记录的都是数据，文本、图像（2）模型 model（3）策略 strategy（4）算法 algorithm。

#### 1.1 学习的类型

* <font color=blue>有监督学习 supervised learning</font> ：从标注数据中学习预测模型的机器学习问题。

  ##### (1) 输入和输出

  * 输入空间和输出空间，i.e. {身高、体重} $\rightarrow$ {学习成绩}。一般来说，输入空间 > 输出空间（用多个信息预测一个信息）。

  * 称每个具体的输入 (input) 为一个实例 (instance)，或观测 (observation)

    每个实例可以用一个特征向量 $\boldsymbol x = (x^{(1)}, ..., x^{(p)})^T \in \mathbb R^p$ 表示。

    通常输入变量用 $X$ 表示，输出变量用 $Y$ 表示。小写字母代表随机变量的一个取值。

  * 训练样本：称一个 $\boldsymbol{(x, y)}$ 数据对为一个样本(sample)。训练用的样本集合可以如下表示

  $$
  \{\boldsymbol{(x_1, y_1), ..., (x_n, y_n)}\}
  $$

  * 输出变量 $Y$ ：（1）若为连续值，则为回归问题 regression；（2）若为离散值，则为分类问题 classification。

  ---

  1. 训练样本 $\{\boldsymbol{(x_1, y_1), ..., (x_n, y_n)}\} \rightarrow$ 学习系统 $\rightarrow$ 模型：$\hat{P}(Y|X)$  或 $Y = \hat{f}(X)$ 

  2. 模型 $\rightarrow$ 预测系统：输入 $X_{n+1}$，返回 $y_{n+1} = arg\max_{y} \hat{P}(y|X_{n+1})$ 或 $y = \hat{f}(X_{n+1})$

  ---

  ##### (2) 联合概率分布

  随机变量 $X, Y$ 服从联合概率分布 $P(X, Y)$ 。样本$(x_i, y_i)$ 依据 $P(X, Y)$ 独立同分布的产生。

  ##### (3) 假设空间

  输入到输出的映射，由模型表示。假设空间就是输入到输出的映射的集合。

* <font color=blue>无监督学习 un-supervised learning</font>：

  从无标注数据中学习模型的机器学习任务。<u>本质是学习数据中的规律和潜在结构</u>。

  i.e. 聚类分析、降维分析。

* <font color=blue>强化学习 reinforcement learning</font>



#### 1.2 学习的步骤

1. 明确学习模型（函数集合 $\rightarrow$ 假设空间 hypothesis testing）
2. 评价控制（具体使用何种metric评价，损失如何度量）
3. 训练最优模型（训练细节）

#### 1.3 按模型进行分类

1. 概率模型 / 非概率模型：$P(y|x)$  -- 朴素贝叶斯		$y = \hat{f}(x)$  -- 神经网络

2. 线性模型 / 非线性模型：$\sum_{j=1}^{N}{\beta_ix_i}$

3. 参数化模型 / 非参数化模型



#### 1.4 统计学习方法的三要素

0. 数据：能够被记录的都是数据

1. 模型：模型都属于假设空间。 $\mathcal F=\{f \mid Y = f(X)\}$ 。可能存在参数 $\theta \in \mathbb R^{p}$

2. 策略：指按何种标准去选择最优模型

   (1) 损失函数和风险函数：损失函数度量一次预测的好坏；风险函数度量平均意义下预测的好坏

   * 损失函数 (Loss Function) -- $L(Y, f(x))$

     * 0-1损失函数: $L(Y, f(x))=1 \mbox{  if}\,\,\, Y=f(x) \,\,\,\mbox{else 0}$

     * 平方损失函数：$L(Y, {f}(x)) = (Y - {f}(x))^2$

     * 绝对值损失函数：$L(Y, {f}(x)) = |Y - {f}(x)|$

     * 对数损失函数/对数似然损失函数 (log-likelihood loss function)：
       $$
       L(Y, P(Y\mid X)) = -\log{P(Y\mid X)}
       $$

   * 风险函数 (Risk Function) 
     $$
     E(f) = E_P\{L(Y, f(X))\} = \int L(y, f(x))P(y, x)dydx
     $$
     但我们并不知道 $P(Y, X)$，此时就需要 $\rightarrow$ 经验风险（经验损失）
     $$
     R_{exp} = \frac{1}{N}\sum_{i=1}^{N}{L(y_i, f(x_i))}
     $$
     当 $N \rightarrow \infty$, 经验风险趋近于期望风险（大数定律）。

   (2) 经验风险最小化

   * 按经验风险最小化的原则选择模型
     $$
     f = arg\min_{f} \frac{1}{N}\sum_{i=1}^{N}{L(y_i, f(x_i))}
     $$
     当 $N$ 比较大时，效果较好。当 $N$ 较小时，容易出现过拟合 (over-fitting) 问题。

     》为了避免过拟合，可以添加正则化 $T(f)$
     $$
     R_{exp} = \frac{1}{N}\sum_{i=1}^{N}{L(y_i, f(x_i))}+\lambda \, T(f)
     $$
     》加入正则项的目的是<u>降低模型的复杂程度</u>(尽量少的拟合噪声)，使得model更robust

3. 算法：模型具体的计算方法（最优化问题的求解）



#### 1.5 模型评估和模型选择

* 训练误差和测试误差

  $Y = f(X)$  $\leftrightarrow$  训练误差  $R_{exp} = \frac{1}{N}\sum_{i=1}^{N}{L(y_i, f(x_i))}$   $\leftrightarrow$   训练集    $\leftrightarrow$    代表模型的拟合能力

  ​                    $\rightarrow$  测试误差  $R_{exp} = \frac{1}{N}\sum_{j=1}^{M}{L(y_j, f(x_j))}$   $\leftrightarrow$   测试集    $\leftrightarrow$    代表模型的预测能力

* 过拟合与模型选择

  * 过拟合：模型包含的参数过多，以至于模型对训练数据拟合较好，但对测试数据预测较差

  * [例] 给定训练数据集 $\{(x_1, y_1), ..., (x_N, y_N)\}, \quad N=10$ 

    用M次多项式 $f(\boldsymbol{x, w})=w_0+w_1x + ... +w_mx^m $拟合数据，损失函数为平方损失函数。

    最小二乘法求 $\boldsymbol{w}$ 等价于 $\boldsymbol w = arg\min_{\boldsymbol w}{L(\boldsymbol w)}$

