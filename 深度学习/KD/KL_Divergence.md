### KD Divergence (KL散度)

Function: measure the difference between two probability distributions over the same variable $x$ , a measure, called the Kullback-Leibler divergence, or simply , the KL divergence, has been popularly used in the data mining literature.

总而言之，言而总之：就是度量两种概率分布之间的差异

KL散度和 ==相对熵== 高度相关

Denoted:  $D_{KL}(p(x), q(x))$  , 表示用$q(x)$ 近似 $p(x)$ 分别丢失的 ==信息熵== 

$p(x), q(x)$都是对离散随机变量 $x$ 的分布度量：

离散公式如下： $D_{KL}(p(x)||q(x))=\sum\limits_{x\in X } p(x) ln\frac{p(x)}{q(x)}$

连续公式如下： $D_{KL}(p(x)||q(x))=\int_{a}^{b} p(x) ln\frac{p(x)}{q(x)} dx$

==Attention== : 

1. KL散度并不是 距离度量， ==不是对称的== 即是： $D_{KL}(p(x), q(x))\ne D_{KL}(q(x), p(x))$
2. $D_{KL}(P ||Q) ≥ 0$ and  $D_{KL}(P||Q) = 0$  if and only if P = Q

###  So what's relative entropy?



1. **Entropy**:  $H(x) = -p(x) log\ p(x)$

   如果 $x$ $符合p(x)$ 分布 $X\sim p(x)$,  期望 $E_pg(X)=\sum\limits_{x\in X}g(x)p(x)$ 

   so denote: $g(x) = log \frac{1}{p(X)}$

   因此 $H(x) = E_p log \frac{1}{p(X)}$

2. 联合熵： joint entropy

   $H(X, Y)$ 表示为离散变量 $(X, Y)$ 的分布的熵,

   $H(X, Y) = -\sum\limits_{x\in X}\sum\limits_{y\in Y} p(x, y) log\ p(x, y)$

   转换为：$H(X, Y)=-E\ log\ p(X, Y)$

   

3. 条件熵：condition entropy
   $$
   \begin{eqnarray}
   H(Y|X)	&=& \sum\limits_{x\in X} p(x) H(Y|X=x) \\
   &=& -\sum\limits_{x\in X} p(x) \sum\limits_{y\in Y} p(y|x) \ log\ p(y|x) \\
   &=& -\sum\limits_{x\in X} \sum\limits_{y\in Y} p(x, y)\ log \ p(y|x) \\
   &=& -E_p(x, y) \ log\ p(Y|X)
   \end{eqnarray}
   $$

4.  $H(X, Y) = H(X) + H(Y|X)$

   

5. 相对熵： relative entropy
   $$
   \begin{eqnarray}
   D(p||q)&=&\sum\limits_{x\in X} p(x)\ log\frac{p(x)}{q(X)} \\
   ~&=&E_p\ log \frac{p(X)}{q(X)}
   \end{eqnarray}
   $$