Graph learning
==============================

Collection of models for learning networks from signals.

Clustering methods follow the [sklearn](https://scikit-learn.org/stable/) API.

## Installation

Clone the git repository and install with pip:
```
git clone https://github.com/LTS4/graph-learning.git
cd graph-learning
pip install .
```

## References

### Base Models

#### Smooth learning (LogModel)

> V. Kalofolias, “How to Learn a Graph from Smooth Signals,” in Proceedings of the 19th International Conference on Artificial Intelligence and Statistics, May 2016, pp. 920–929. https://doi.org/10.48550/arXiv.1601.02513.

> V. Kalofolias and N. Perraudin, “Large Scale Graph Learning From Smooth Signals,” presented at the International Conference on Learning Representations, Sep. 2018. Available: https://openreview.net/forum?id=ryGkSo0qYm

Part of the code is ported to Python from the Matlab implementation from https://github.com/epfl-lts2/gspbox, published under GNU General Public License v3.0.

#### LGRMF

> H. E. Egilmez, E. Pavez, and A. Ortega, “Graph learning with Laplacian constraints: Modeling attractive Gaussian Markov random fields,” in 2016 50th Asilomar Conference on Signals, Systems and Computers, Nov. 2016, pp. 1470–1474. https://doi.org/10.1109/ACSSC.2016.7869621.

### Clustering models

#### GLMM

> H. P. Maretic and P. Frossard, “Graph Laplacian Mixture Model,” IEEE Transactions on Signal and Information Processing over Networks, vol. 6, pp. 261–270, 2020, https://doi.org/10.1109/TSIPN.2020.2983139.

#### k-Graphs

> H. Araghi, M. Sabbaqi, and M. Babaie–Zadeh, “$K$-Graphs: An Algorithm for Graph Signal Clustering and Multiple Graph Learning,” IEEE Signal Processing Letters, vol. 26, no. 10, pp. 1486–1490, Oct. 2019, https://doi.org/10.1109/LSP.2019.2936665.

### Temporal graph learning

#### TGFA

> K. Yamada, Y. Tanaka, and A. Ortega, “Time-Varying Graph Learning with Constraints on Graph Temporal Variation,” Jan. 10, 2020, https://doi.org/10.48550/arXiv.2001.03346.


#### Temporal Multiresolution Graph Learning (GraphDictHier)

> K. Yamada and Y. Tanaka, “Temporal Multiresolution Graph Learning,” IEEE Access, vol. 9, pp. 143734–143745, 2021, https://doi.org/10.1109/ACCESS.2021.3120994.


### Dictionary Models

#### Parametric Dictionary Learning (GraphDictSpectral)

> D. Thanou, D. I. Shuman, and P. Frossard, “Parametric dictionary learning for graph signals,” in 2013 IEEE Global Conference on Signal and Information Processing, Dec. 2013, pp. 487–490. https://doi.org/10.1109/GlobalSIP.2013.6736921.

#### Graph Dictionary Signal Model (GraphDictLog, GraphDictBase)

> W. Cappelletti and P. Frossard, “Graph-Dictionary Signal Model for Sparse Representations of Multivariate Data,” Nov. 08, 2024, [arXiv:2411.05729](https://arXiv.org/abs/2411.05729)
