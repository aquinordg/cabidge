# Random Data Generator Algorithm for Clustering (rdga_4k)

The package generates synthetic data for applications in clustering algorithms.

## Functions
### Categorical Binary Random Data

```sh
catbird(n_feat, feat_sig, rate, lmbd=.8, eps=.2, random_state=None)
```
#### Parameters:
`n_feat` int > 1: number of features;

`feat_sig` list: number of significant features (no noise), equal to the size of the rate list;

`rate` list: division of examples into clusters, equal to the size of the feat_sig list;

`lmbd` float 0,1: intersection factor between features;

`eps` float 0,1: feature noise rate;

`random_state` int: random seed.

#### Explanation:

Consider the following scenario. Each data sample represents a person. Each feature indicates whether such a person has a certain skill. We assume that people can be grouped in clusters such that:

- the presence/absence of a certain few features (called *dependent features*) are correlated; and

- other features may be present/absent independently.

Dependent features vary among clusters. The rationale is that, for a particular group of people, some features are more important than others. These important features appear more frequent than other features in the group. Moreover, they are positively or negatively correlated. For example, IT professionals usually share common skills: programming, data base skills, etc. On the other hand, the presence of some pairs of skills should be rare. For example, IT professionals who program rarely use the same programming language or development environment.

Thus, let $x_{ij}$ be the indicator whether the $i$-th person, belonging to cluster $c(i)$, has the $j$-th feature. We want to generate a dataset such that

$$P(x_{ij} = 1) \approx \begin{cases}
        \lambda & \text{ if } j \in \mathcal{F}_{c(i)} \\
        \epsilon & \text{ otherwise,}
    \end{cases}$$

where $0 < \epsilon < \lambda < 1$ and $\mathcal{F}_{c(i)}$ is the set of dependents features of the cluster $c(i)$.

We also want that the following statements to hold:

$$P(x_{ij}, x_{ij'}) = P(x_{ij})P(x_{ij'}) \forall j \not\in \mathcal{F}_{c(i)}\text{,}$$

$$P(x_{ij}) = P(x_{i'j}) \text{ if } c(i) = c(i')\text{, and }$$

$$P(x_{ij}, x_{ij'}) \neq P(x_{ij})P(x_{ij'}) \forall j, j' \in \mathcal{F}_{c(i)}\text{.}$$

To generate data with such properties, each cluster $c$ in the dataset is associated with an $m$ by $m$ matrix $W^c$ whose elements are

$$w^c_{pq} \sim \mathcal{N}(0, 1)\text{,}$$

and each sample $\vec{x}_{i} = (x_{i1}, x_{i2}, \dots, x_{im})$ such that $c(i) = c$ is associated with a vector $m$-dimensional $\vec{a}^i$ whose elements are

$$a_p^i \sim \mathcal{N}(0, 1)\text{.}$$

Then, let

$$\vec{b}^i = \frac{1}{\sqrt{m}} W^{c(i)} \times \vec{a}^i\text{,}$$

then, we generate the dependent features ($j \in \mathcal{F}^{c(i)}$)

$$x_{ij} = \begin{cases}
        1 & \text{ if } \Phi\!\left(b^i_j\right) < \lambda \\
        0 & \text{ otherwise,}
    \end{cases}$$ where the $\Phi$ is the cumulative distribution function of the standard normal distribution. The other features,
$j \not\in \mathcal{F}^{c(i)}$, are generated using
$$x_{ij} = \begin{cases}
        1 & \text{ if } u_{ij} \sim \mathcal{U}(0, 1) < \epsilon \\
        0 & \text{ otherwise,}
    \end{cases}$$

Remark: note that the product $XY$ of random variables $X, Y \sim \mathcal{N}(0, 1)$ is not a standard normal distribution. However, it is similar enough to use $\Phi$ to satisfy the first equation.

[$\epsilon = 0.1$; $\lambda = 0.9$]{#fig:g2 .image .placeholder
original-image-src="fig/e0.1_l0.9" original-image-title="fig:"
width="1\\linewidth"}

[$\epsilon = 0.1$; $\lambda = 0.8$]{#fig:g1 .image .placeholder
original-image-src="fig/e0.1_l0.8" original-image-title="fig:"
width="1\\linewidth"}

[$\epsilon = 0.2$; $\lambda = 0.9$]{#fig:g4 .image .placeholder
original-image-src="fig/e0.2_l0.9" original-image-title="fig:"
width="1\\linewidth"}

[$\epsilon = 0.2$; $\lambda = 0.8$]{#fig:g3 .image .placeholder
original-image-src="fig/e0.2_l0.8" original-image-title="fig:"
width="1\\linewidth"}
