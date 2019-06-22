[TOC]

# Problem

Given an undirected graph $G=(V,E):V=[n]$, we wish to compute the *root vector* $\pi[1..n]$ :
$$
\pi[i]=\max_{j\in V:\kappa'(i,j)\geq1}j
$$
where $\kappa'(i,j)$ represents the edge-connectivity between $i$ and $j$ in $G$. We see that $\kappa'(i,j)\geq1\Leftrightarrow i$ and $j$ are in the same component in $G$.

Thus, intuitively, $\pi[i]$ represents the largest vertex label that is in the same component as $i$ is.

Here we may safely assume that $G$ is simple. Since self-loops and multi-edges won't affect the connectivity of $G$.

# Definition

Define $A[1..n,1..n]$ as the *adjacency matrix* :
$$
A[i,j]=\begin{cases}
1,\text{ if }i<j,(ij)\in E\\
0,\text{ otherwise}
\end{cases}
$$
We see that $A$ is strictly upper triangular.

Let $A^{(k)}$ be the adjacency matrix of the modified graph after $k$ iterations. Given any $A^{(k)}$, define $\pi^{(k)}[1..n]=\pi(A^{(k)})$ as the *parent vector* :
$$
\pi^{(k)}[i]=\max(i,\max_{j\in V}A^{(k)}[i,j]\cdot j)
$$
We see that $\forall i,k,\pi^{(k)}[i]\geq i$.

Also, given any $\pi^{(k)}$, define $P^{(k)}[1..n,1..n]=P(\pi^{(k)})$ as the *parent matrix* :
$$
P^{(k)}[i,j]=\begin{cases}
1,\text{ if }\pi^{(k)}[i]=j\\
0,\text{ otherwise}
\end{cases}
$$
We see that $P^{(k)}$ is upper triangular with each row having exactly one non-zero entry, $1$.

# Algorithm

Set $A^{(0)}=A$ and initialize $\pi^{(0)},P^{(0)}$ accordingly by definition. And set ${P^{(0)}}'=P^{(0)}$. Compute the following until convergence:
$$
\begin{aligned}
B^{(k)}[i,j]&=\max_{l\in V}{P^{(k-1)}}'[l,i]\cdot A^{(k-1)}[l,j]\\
A^{(k)}[i,j]&=\max(A^{(k)}[i,j],B^{(k)}[i,j])\\
\pi^{(k)}&=\pi(A^{(k)})\\
P^{(k)}&=P(\pi^{(k)})\\
{P^{(k)}}'[i,j]&=\max_{l\in V}P^{(k)}[i,l]\cdot P^{(k)}[l,j]
\end{aligned}
$$
Finally, set $\pi=\pi^{(k)}$ of the last iteration.

Intuitively, we hook vertices in the same component to the largest vertex in that component and keep iterating. This creates trees (stars) over iterations.

# Observation

## Hooking

In iteration $k$, $B^{(k)}[i,j]=1\Leftrightarrow\exists l<j:P^{(k-1)}[l,i]=A^{(k-1)}[l,j]=1$. This implies $\pi^{(k-1)}[l]=i\wedge(lj)\in E'$, where $E'$ is the edge set of the modified graph from the previous iteration, $k-1$. And $B^{(k)}[i,j]=0$ otherwise.

![Figure 1: B[i, j]](https://github.com/yx-z/Konnectivity/blob/master/notes/f1.png?raw=true)

Intuitively, $B^{(k)}$ represents the adjacency matrix of new edges we hooked in to the graph. And we will enforce be to be upper triangular as $A^{(k)}$.

These new edges shortcut vertices in the same component. We will then add these edges to $A^{(k)}$. Therefore, algebraically, $A^{(k)}[i,j]$ will be $\max(A^{(k)}[i,j],B^{(k)}[i,j],B^{(k)})$.

## Shortcutting

Notice that in any iteration $k$, we will perform shortcutting:
$$
\begin{aligned}
{P^{(k)}}'[i,j]=1&\Leftrightarrow P^{(k)}[i,l]=P^{(k)}[l,j]=1\\
&\Leftrightarrow\pi[i]=l,\pi[l]=j\\
\implies&\pi[i]=j
\end{aligned}
$$

# Semiring

It is easy to see that the update for $B^{(k)}[i,j]=\max_{l\in V}P^{(k-1)}[l,i]\cdot A^{(k-1)}[l,j]$ can be achieved by $B=(P^{(k-1)})^T\otimes A^{(k-1)}$ equipped with a Max-Times semiring. Similar with the shortcutting, we can do ${P^{(k)}}'={P^{(k)}}^2$ equipped with the same semiring.

# Cost Analysis

TODO

# Future Work

- Implement the algorithm in `ctf`.

# Supervertex

After each iteration $k$, We can further merge vertices with a common root together. We will then call the root supervertex, as the diagram shown below.

![Supervertex](https://github.com/yx-z/Konnectivity/blob/master/notes/f2.png?raw=true)

Algebraically, we can merge columns of $A^{(k)}$ and $P^{(k)}$.

# Improvement

$$
\begin{aligned}
P_{pre}&=P\\
B&=P^TAP\\
P&=PC:C\text{ is upper-triangular part of }B\\
P&=P^2\\
while&(\|P-P_{pre}\|>0)
\end{aligned}
$$

with max-times semiring, and symmetric $A$.

$B=B’P$ where $B’$ is $B$ in the original algorithm (previous context). So
$$
\begin{aligned}
&P^TAP[i,j]=1&\Leftrightarrow\\
%&\exists l:B’[i,l]=P[l,j]=1&\Leftrightarrow\\
&\exists l,m:P[m, i]=A[m,l]=P[l,j]=1&\Leftrightarrow\\
&\exists l,m:\pi[m]=i\wedge\pi[l]=j\wedge(ml)\in E&
\end{aligned}
$$
![Improved Hooking](https://github.com/yx-z/Konnectivity/blob/master/notes/f3.png?raw=true)

where $m\leq i,l\leq j$. We see that $B$ preserves symmetry.

Now consider the step $P=PC$. $P[i,j]=1\Leftrightarrow\exists l:P[i,l]=C[l,j]=1$. Since $C$ is upper triangular, we know $i\leq l\leq j$. This implies $\pi[i]=l$.

![Hook Parent](https://github.com/yx-z/Konnectivity/blob/master/notes/f4.png?raw=true)

# LACC

Let $A$ be the adjacency matrix and let $p$ represents the parent vector. Initialize $p_i=i$. We do the following until convergence of $p$ under the Max-Times semiring.

- $q=Ap:q_i=\max_jA_{ij}p_j$. Thus $q_i$ is the max parent among neighbors of $i$.
- $r_i=\max(p_i,q_i)$. Thus $r_i$ represents the root of the component that contains $i$.
- $P_{ij}=1,\forall i,j:p_i=j,0$ otherwise.
- $s=P^Tr:s_i=\max_jP^T_{ij}r_j=\max_jP_{ji}r_j$. Suppose $s_i=r_j$, we know that $P_{ji}=1\implies p_j=i$. Intuitively, if $i$ is the parent of $j$, then we hook $i$ to the root of $j$.
- $p_i=\max(s_i,p_i)$. We will then update the new parents for each vertex.

