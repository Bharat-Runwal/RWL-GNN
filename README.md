# RWL-GNN 

A PyTorch implementation of "Robustifying Graph Neural Networks via Weighted Laplacian" Accepted at [SPCOM](https://ece.iisc.ac.in/~spcom/2022/) 2022. [(Slides)](https://drive.google.com/file/d/1Z0E9ualMfngrko_g1zdMu_VYB1IpcpCC/view?usp=sharing)

The code is based on Pytorch adversarial repository, DeepRobust [(https://github.com/DSE-MSU/DeepRobust)](https://github.com/DSE-MSU/DeepRobust) and [Pro-GNN](https://github.com/ChandlerBang/Pro-GNN)
[![][colab]][RWL-GNN]
<div align=center><img src="joint.png" width="700"/></div>

## Abstract 
- Graph neural network (GNN) is achieving remarkable performances in a variety of
application domains. However, GNN is vulnerable to noise and adversarial attacks
in input data. Making GNN robust against noises and adversarial attacks is an
important problem. The existing defense methods for GNNs are computationally
demanding, are not scalable, and are architecture dependent. In this paper, we
propose a generic framework for robustifying GNN known as Weighted Laplacian
GNN (RWL-GNN). The method combines Weighted Graph Laplacian learning
with the GNN implementation. The proposed method benefits from the positive
semi-definiteness property of Laplacian matrix, feature smoothness, and latent
features via formulating a unified optimization framework, which ensures the ad-
versarial/noisy edges are discarded and connections in the graph are appropriately
weighted. For demonstration, the experiments are conducted with Graph convo-
lutional neural network(GCNN) architecture, however, the proposed framework
is easily amenable to any existing GNN architecture. The simulation results with
benchmark dataset establish the efficacy of the proposed method over the state-
of-the-art methods, both in accuracy and computational efficiency. 

## Requirements
See that in https://github.com/DSE-MSU/DeepRobust/blob/master/requirements.txt

## Installation
To run the code, first you need to install DeepRobust:
```
pip install deeprobust
```
Or you can clone it and install from source code:
```
git clone https://github.com/DSE-MSU/DeepRobust.git
cd DeepRobust
python setup.py install
```

## Run the code
After installation, you can clone this repository
```
git clone https://github.com/Bharat-Runwal/RWL-GNN.git
cd RWL-GNN
python train.py --seed 10 --dataset cora  --attack meta --ptb_rate 0 --epoch 400 --alpha 1.0  --gamma 1.0 --lambda_ 0.001 --lr  1e-3
```
[colab]: <https://colab.research.google.com/assets/colab-badge.svg>
[RWL-GNN]: <https://github.com/Bharat-Runwal/RWL-GNN/blob/main/Demo_RWL_GNN.ipynb>


<!-- ## Cite
For more information, you can take a look at the [paper](https://arxiv.org/abs/2005.10203) or the detailed [code](https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/defense/prognn.py) shown in DeepRobust.

If you find this repo to be useful, please cite our paper. Thank you.
```
@inproceedings{jin2020graph,
  title={Graph Structure Learning for Robust Graph Neural Networks},
  author={Jin, Wei and Ma, Yao and Liu, Xiaorui and Tang, Xianfeng and Wang, Suhang and Tang, Jiliang},
  booktitle={26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD 2020},
  pages={66--74},
  year={2020},
  organization={Association for Computing Machinery}
}
``` -->
