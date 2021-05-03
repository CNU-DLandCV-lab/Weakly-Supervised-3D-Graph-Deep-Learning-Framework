# Weakly-Supervised 3D Graph Deep Learning Framework 

#### Description
Point cloud registration is important and critical for the applications of changing detection, deformation monitoring, etc., which is also challenging due to the vast clustered points and the irregular and complex structures of spatial objects. Aiming at this problem, we designed a semi-supervised 3-dimensional (3D) graph deep learning framework of point cloud registration which can simultaneously learn detector (attention expression) and descriptor (deep feature) for point cloud matching by combining the weakly-supervised way. The detector and descriptor are respectively constructed to extract the keypoints and describe the deep feature of each keypoint. In the framework, we innovatively combined the graph convolutional networks (GCNs) and attention mechanism into PointNet++ structure to form a novel weakly-supervised 3D graph deep learning framework, which can effectively extract keypoints and descriptor of each keypoint. In the training process of the learning framework, we rotated point cloud randomly to form the training data by a weakly-supervised way and integrated a siamese network into the training model. Besides, the proposed method designed the point set-based neural networks into a uniform deep learning model to construct 3D deep features and keypoints. In the experiments, our method can achieve better results of point cloud registration in comparisons with the state-of-the-art methods, which illustrated the advantages of the proposed method.

#### environment
The training process is conducted under the TensorFlow framework and Ubuntu 16.04 cuda9.2.

#### We will release the code as soon as possible
