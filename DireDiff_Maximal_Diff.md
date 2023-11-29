# 方向导数、极大值定理 和 可微性

（接着一年前的上一期）

本节我们来简单介绍一下，**方向导数**、**次可微性** 和 **最优化条件**。还记得上一节我们介绍的次梯度 $\partial f$ 吗？有了次梯度我们就可以重新把实分析中的可微、梯度和导数联系在一起。让我们来看看凸分析里面是如何扩展这些数学工具的吧。

## 方向导数

***(方向导数的定义)*** 对于一个凸的真函数 $f:x\rightarrow (-\infty,\infty]$，(proper convex) 如果极限：

$$
f'(x;d):=\lim_{\alpha\to0^+}\frac{f(x+\alpha d)-f(x)}{\alpha}
$$

存在，那么方向导数 $f'(x;d)$ 则存在。

方向导数可以理解为带有方向的导数。这里的方向就是定义中的 $d$ 。根据不同的符号可以控制导数的方向。以往的导数定义的是一条与函数相切的直线，而方向导数则描述的是一条射线。所以可以描述更加宽松的情况，那就是也许该点并不可微，但是却存在方向导数。

**(凸函数的方向导数存在性)** 如果函数 $f$ 是凸的，则在 $f$ 的定义域内部的任意值 ( $x\in\text{int}\ \text{dom}f$）方向导数 $f'(x;d)$ 都存在，且等于 $\inf_{\alpha>0}\frac{f(x+\alpha d) - f(x)}{\alpha}$ 。

证明：

首先我们声明 $\alpha$ 的上确界为 $\beta:=\inf\alpha$，同时 $\lambda := \frac{\alpha}{\beta}$，$\Phi_d(\alpha):=\frac{f(x+\alpha d)-f(x)}{\alpha}$。

因为函数 $f$ 为凸函数，所以我们可以利用我们设定的 $\lambda$ 代入凸函数定义：

$$
\begin{aligned}
f(x+\alpha d) &= f(x+\lambda\beta d) = f(\lambda(x+\beta d)+(1-\lambda)x)\\
&\le \lambda f(x+\beta d)+(1-\lambda)f(x)\\
&\le \lambda (f(x+\beta d) - f(x)) + f(x)\\
\end{aligned}
$$

之后我们继续化简，可以得到：

$$
\begin{aligned}
f(x+\alpha d) &\le \lambda (f(x+\beta d) - f(x)) + f(x)\\
f(x+\alpha d) - f(x) &\le \frac{\alpha}{\beta} (f(x+\beta d) - f(x))\\
\frac{f(x+\alpha d) - f(x)}{\alpha} &\le \frac{ (f(x+\beta d) - f(x))}{\beta}\\
\Phi_d(\alpha)&\le\Phi_d(\beta)
\end{aligned}
$$

也就相当于 $f'(x;d):=\inf_{\alpha>0}\frac{f(x+\alpha d)-f(x)}{\alpha}$。 $\blacksquare$

***（凸函数方向导数的性质）***感兴趣的同学可以参考 [Amir Beck 教科书](http://archive.siam.org/books/mo25/) 中的 定理 3.21 - 3.25

1. 凸函数的方向导数是 **凸(convex)** 的
2. $f'(x;\beta d)=\beta f'(x;d)$
3. 对于函数 $f(x):=\max_{i\in I}f_i$ （函数集合 $I$ 有穷），其方向导数也一样是这些函数的最大值  $f'(x;d)=\max\{f_i(x;d)\}$
4. $f'(x;d)\le f(x+d)-f(x)$，因为我们之前证明了有 

$$
f'(x;d)=\inf_{\alpha>0}\frac{f(x+\alpha d) - f(x)}{\alpha}\le\Phi_d(1)=f(x+d)-f(x)
$$

## 极大值定理 （Max Formula）

接下来，有了方向导数的定义，我们可以推导出极大值定理。**极大值定理链接了次梯度以及方向导数，是一个非常有用的定理。**我们后面会多次用到极大值定理。

***（极大值定理）***对于一个凸的真函数 $f:X\to(-\infty,\infty]$， 对于所有 $x\in \text{int}\ \text{dom}f$，我们都有 

$$
f'(x;d)=\max\{\langle g,d\rangle|g\in\partial f(x)\}=\sup_{g\in\partial f(x)}\langle g, d\rangle
$$

证明：为了证明相等，我们不妨从 大于等于 和 小于等于 两遍开始推导。

**对于 “$\ge$“，**

因为  $g\in\partial f(x)$ 为次梯度，所以对 $\forall \alpha > 0$  我们根据次梯度定义可以得到 

$$
\begin{aligned}
f(x)+\langle g, \alpha d\rangle &\le f(x+\alpha d) \\
\langle g, d\rangle&\le \frac{f(x+\alpha d)-f(x)}{\alpha}\\
\langle g, d\rangle&\le f'(x;d)\\
\sup_{g\in\partial f(x)} \langle g, d\rangle&\le f'(x;d)
\end{aligned}
$$

**对于 “$\le$”**

我们可以设 $h(w):=f'(x;w)$，因为函数 $f$ 是凸函数，所以其方向导数也为凸。同时函数和函数次梯度的定义域 $\text{dom} h = \text{dom}\partial h = X$。

不妨设 $\tilde{g}\in\partial h(d),\ v \in X$，那么对于 $\forall \alpha\ge 0$，我们则有

$$
\begin{aligned}
\alpha f'(x;v)=f'(x;\alpha v)=h(\alpha v)&\ge h(d) + \langle \tilde{g}, \alpha v-d\rangle\\
\alpha f'(x;v)&\ge f'(x;d)+\langle \tilde{g}, \alpha v\rangle -\langle \tilde{g}, d\rangle\\
\alpha\big(f'(x;v)-\langle \tilde{g}, v\rangle\big)&\ge f'(x;d)-\langle \tilde{g}, d\rangle\\
\Leftrightarrow 0\le\frac{1}{\alpha} &\le\frac{f'(x;d)-\langle \tilde{g}, d\rangle}{f'(x;v)-\langle \tilde{g}, v\rangle}\\
\Leftrightarrow f'(x;d)&\ge\langle \tilde{g}, d\rangle
\end{aligned}
$$

所以我们根据凸函数的方向导数的性质中的第四条可以推导出：

$$
\begin{aligned}
f(y)&\ge f(x)+f'(x;y-x)\\
f(y)&\ge f(x) + h(y-x) \ge f(x)+\langle \tilde{g}, y-x\rangle \qquad \forall \tilde{g}\in\partial h(d)
\end{aligned}
$$

因此可以得出 $\tilde g \in \partial f(x)$ 。所以我们可以得到：

$$
f'(x;d)\le\langle \tilde g, d\rangle \le \sup_{g\in\partial f(x)}\langle g,d\rangle \le f'(x;d)
$$

因此，

$$
f'(x;d)=\max\{\langle g,d\rangle|g\in\partial f(x)\}=\sup_{g\in\partial f(x)}\langle g, d\rangle\qquad\blacksquare
$$

## 可微性

我们来回忆一下高数中的可微的定义：

如果一个函数 $f$ 是可微的 (differential)，那么对于在定义域内部的 $x\in\text{int}\ \text{dom}f$，我们有

$$
(\exists g\in X) \qquad \lim_{h\to 0}\frac{f(x+h)-f(x)-\langle g,h\rangle}{\|h\|}=0
$$

当 $g$ 是唯一的时候，那么函数 $f$ 在 $x$ 上是可微的，$g$ 是函数 $f$ 在 $x$ 上的梯度 $\nabla f(x)$ 。

***（可微处的方向梯度）***如果函数 $f$ 在 $x$ 上是可微的，$f'(x;d)=\langle \nabla{f(x)},d \rangle$ 。

证明： 很明显，在 $d=0$ 的时候 等式成立，那么我们不妨设 $d\neq 0$，那么根据微分定义：

$$
\begin{aligned}
0&=\lim_{\alpha\to 0^+}\frac{f(x+\alpha d)-f(x)-\langle \nabla f(x), \alpha d\rangle}{\alpha \|d\|}\\
&=\lim_{\alpha\to 0^+}\bigg[\frac{f(x+\alpha d)-f(x)}{\alpha \|d\|} - \frac{\langle \nabla f(x), d\rangle}{ \|d\|}\bigg]
\end{aligned}
$$

所以，结合方向导数的定义，我们可以得到

$$
\lim_{\alpha\to 0^+}\frac{f(x+\alpha d)-f(x)}{\alpha \|d\|} - \frac{\langle \nabla f(x), d\rangle}{\|d\|}=\frac{f'(x;d)}{\|d\|}-\frac{\langle \nabla f(x), d\rangle}{\|d\|}
$$

不难得到 $f'(x;d)=\langle \nabla f(x), d\rangle$ 。 $\blacksquare$

***(可微处的次梯度)*** 对于一个凸的真函数 $f:X\to(-\infty,\infty]$。对于 $x\in\text{int}\ \text{dom}f$，如果 $f$ 在 $x$ 上可微，那么其次梯度为一个单点 (singleton) 且 $\partial f(x)=\{\nabla f(x)\}$ 。反之，如果一个函数的次梯度为单点，那么这个函数在该处的梯度则为次梯度的唯一元素。

（偷懒就不证明了，感兴趣的同学可以参考  [Amir Beck 教科书](http://archive.siam.org/books/mo25/) 中的 定理 3.33。感性来说，既然次梯度是一个宽松的切线集合，只要函数都在切线一侧就满足次梯度的定义：那么对于极限情况，也就是这些切线都塌缩成一条的时候，就是一个严格的梯度。那么梯度存在的地方就是可微的。）

好了本节内容就到这里，主要复习了一下微分导数和次梯度的定义。对于各位读者来说可能过于基础，但是如果结合实分析来看凸分析中的这些定义扩展也是一种异域风情。希望各位能有所收获，我们下节再见👋。下节我们会介绍 **Lipschitz 连续性**、**次梯度的有界性** 和 **最优化条件**。