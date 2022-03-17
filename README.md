# Weakly-Supervised 3D Graph Deep Learning Framework 
The code in this toolbox implements the "Graph Convolutional Networks for Hyperspectral Image Classification". More specifically, it is detailed as follow.


# Description
Point cloud registration is important and critical for the applications of changing detection, deformation monitoring, etc., which is also challenging due to the vast clustered points and the irregular and complex structures of spatial objects. Aiming at this problem, we designed a semi-supervised 3-dimensional (3D) graph deep learning framework of point cloud registration which can simultaneously learn detector (attention expression) and descriptor (deep feature) for point cloud matching by combining the weakly-supervised way. The detector and descriptor are respectively constructed to extract the keypoints and describe the deep feature of each keypoint. In the framework, we innovatively combined the graph convolutional networks (GCNs) and attention mechanism into PointNet++ structure to form a novel weakly-supervised 3D graph deep learning framework, which can effectively extract keypoints and descriptor of each keypoint. In the training process of the learning framework, we rotated point cloud randomly to form the training data by a weakly-supervised way and integrated a siamese network into the training model. Besides, the proposed method designed the point set-based neural networks into a uniform deep learning model to construct 3D deep features and keypoints. In the experiments, our method can achieve better results of point cloud registration in comparisons with the state-of-the-art methods, which illustrated the advantages of the proposed method.

# environment

The training process is conducted under the TensorFlow framework and Ubuntu 16.04 cuda9.2.
We also use MATLAB scripts for evaluation and processing of data.


# Prerequisites

Before using the model, you first need to compile the customized tf_ops in the folder tf_ops (we use the customized grouping and sampling ops from PointNet++).

Check and execute tf_xxx_compile.sh under each subfolder. Update the python and nvcc file if necessary. The scripts has been updated for TF1.4, so if you're using TF version < 1.4, refer to the original script provided with PointNet++ for compilation.

# Training
#### Preparation of data
You need to build the triplet training set in advance by following the paper or other methods. 
#### Training
Start training:  ` python train_graph.py ` 

# Testing
To evaluate trained model, you may use  `python test_graph.py` 


# Citation
Please kindly cite the papers if this code is useful and helpful for your research.

Sun, L., Zhang, Z., Zhong, R., Chen, D., Zhang, L., Zhu, L., ... & Wangc, Y. (2022). A Weakly Supervised Graph Deep Learning Framework for Point Cloud Registration. IEEE Transactions on Geoscience and Remote Sensing.

```
@ARTICLE{sun2021graph,
    title={A Weakly Supervised Graph Deep Learning Framework for Point Cloud Registration},
    author={Sun, Lan and Zhang, Zhenxin and Zhong, Ruofei and Chen, Dong and Zhang, Liqiang and Zhu, Lin and Wang, Qiang and Wangb, Guo and Zou, Jianjun and Wangc, Yu},
    journal={IEEE Transactions on Geoscience and Remote Sensing}, 
    year={2022},
    volume={},
    number={},
    pages={1-1},
    doi={10.1109/TGRS.2022.3145474}} 
  ```
#### We will release the code as soon as possible
