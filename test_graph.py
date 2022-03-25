
# coding: utf-8
import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.getcwd()
BASE_DIR = os.path.dirname("__file__")
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tensorflow as tf
import numpy as np
import struct
import tf_util
from datetime import datetime
from pointnet_util import pointnet_sa_module
#from pointnet_util import chevb_model
from pointnet_util import feature_detection_module  
from pointnet_util import *
import h5py
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import BallTree
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling_nd'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import numpy as np

#Parameters and Inputs
BATCH_SIZE=8
LOG_FOUT = open(os.path.join( 'log_train.txt'), 'w')
LOG_FOUT.write(str("log")+'\n')
BASE_LEARNING_RATE=0.1
DECAY_STEP=10
DECAY_RATE=0.7
OPTIMIZER="adam"
MAX_EPOCH=100
scale=0
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(200000)
BN_DECAY_CLIP = 0.99
MAX_KEYPOINT=50
NUM_POINT=4096
KeyPoint=256
nms_radius=0.5
min_response_ratio=1e-2
NUM=100
cloud1="/data/slan/data/data_test_cloud/gazaba/Hokuyo_0.csv"
cloud2="/data/slan/data/data_test_cloud/gazaba/Hokuyo_1.csv"

def chevb_model(xyz, points, npoint, radius, nsample,scope ,mlp, bn_decay,output_dim,bn=True, pooling='max',is_training=True,k=5):
    with tf.variable_scope(scope) as sc:
        with tf.variable_scope("sample_and_group") as sc:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, None, True, True)
        # grouped_xyz this is local xyz wrt to each center point
        # dim B N K 3
        # controls using what feature for covariance computation
        with tf.variable_scope("Laplacian") as sc:
            local_cord = grouped_xyz
            in_shape = new_points.get_shape().as_list()
            W = spec_graph_util.get_adj_mat_dist_euclidean(local_cord[:,:,:,0:3] , flag_normalized = True)
            W = tf.identity(W, name='adjmat')
            W_knn = spec_graph_util.cov_mat_k_nn_graph(W, k = 32 )
            W_knn = spec_graph_util.corv_mat_setdiag_zero(W_knn)
            W_knn = tf.identity(W_knn, name='adjmat_knn')
            L = spec_graph_util.corv_mat_laplacian0(W_knn , flag_normalized = True)
            L = tf.identity(L, name='laplacian')
            
        with tf.variable_scope("MLP") as sc:
            for i, num_out_channel in enumerate(mlp):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay)

            x=tf.matmul(L,new_points)
            #gcn
        with tf.variable_scope("GCN") as sc:
            '''
            x2=x
            W_attention=tf.get_variable('weights_attention',[x.shape[-1],x.shape[-1]],initializer=tf.random_normal_initializer(mean=0, stddev=1),dtype=tf.float32)      
            print(x.shape)
            x1=tf.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],x.shape[3]))
            new_points_=tf.matmul(x1,W_attention)
            new_points_=tf.reshape(new_points_,(x.shape[0],x.shape[1],x.shape[2],x.shape[-1]))           
            #gcn attention 
            attention=tf.nn.softmax(new_points_)
            x=attention*x2
            '''
            W_spec=tf.get_variable('weights_spec',[x.shape[-1],output_dim],initializer=tf.random_normal_initializer(mean=0, stddev=1),dtype=tf.float32)      
            x1=tf.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],x.shape[3]))
            new_points_=tf.matmul(x1,W_spec)
            new_points_=tf.reshape(new_points_,(x.shape[0],x.shape[1],x.shape[2],output_dim))
           # new_points_=tf.transpose(new_points_,[1,2,0,3])
            new_points_=tf.nn.relu(new_points_)
            new_points_ = tf.reduce_max(new_points_, axis=[2], keep_dims=True)
            new_points_ = tf.squeeze(new_points_, [2])            
            print("new_points_ shape",new_points_.shape)
    return new_xyz, new_points_, idx

def DataSet(filename,search_radius,num=1024,interval=0):
    data=pd.read_csv(filename,sep=',')
    data=data[::4]
    print(data)
    data=data[data.columns[1:4]]
    data.to_csv('b.txt',sep=' ')

    #print(data)
    if(interval!=0): 
        data=np.array(data)
        data=data.reshape((1,data.shape[0],data.shape[1]))
        new_xyz=gather_point(data,farthest_point_sample(data.shape[1]/interval, data))
        with tf.Session() as sess:
            result=new_xyz.eval()
        data=result[0]
   # data=data[::interval]
   # print((data).shape)
    #print(data)
    data=np.array(data)
    tree = BallTree(np.array(data), leaf_size=2)
    re=[]
    index_set=[]
    real_xyz=[]
    for j in range(NUM):
        index=int(len(data)/NUM)*j
        index_set.append(index)
        index_raw_data_set=find_near_points(tree,data[index],num)
        re.append(data[index_raw_data_set]-data[index])
        real_xyz.append(data[index_raw_data_set])
    return re,index_set,real_xyz

def DataSet_all(filename,search_radius,num=1024,interval=0):
    data=pd.read_csv(filename,sep=',')
    data=data[data.columns[1:4]]

    if(interval!=0): 
        data=np.array(data)
        data=data.reshape((1,data.shape[0],data.shape[1]))
        new_xyz=gather_point(data,farthest_point_sample(data.shape[1]/interval, data))
        with tf.Session() as sess:
            result=new_xyz.eval()
    data=result[0]
    print(data.shape)
    #print(data)
    data=np.array(data)
    tree = BallTree(np.array(data), leaf_size=2)
    re=[]
    index_set=[]
    real_xyz=[]
    for j in range(len(data)):
        index=j
        index_set.append(index)
        index_raw_data_set=find_near_points(tree,data[index],num)
        re.append(data[index_raw_data_set]-data[index])
        real_xyz.append(data[index_raw_data_set])
    return re,index_set,real_xyz

def find_near_points(tree,point,num_point=1):
    point=point.reshape(-1,3)
    dist, ind = tree.query(point, k=num_point) 
    return ind

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
   # labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl
'''
def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    l0_xyz=point_cloud
    l0_points=None

    
    l1_xyz, l1_points, l1_indices=chevb_model(l0_xyz, l0_points, 1024, 0.2,nsample=32,scope='layer1',mlp=[64,64,128], output_dim=128,bn=True, pooling='max',
                                              is_training=is_training, bn_decay=bn_decay)
    l2_xyz, l2_points, l2_indices=chevb_model(l1_xyz, l1_points, 512, 0.4,nsample=32,scope='layer2',mlp=[128,128,256], output_dim=256,bn=True, pooling='max',
                                              is_training=is_training, bn_decay=bn_decay)
    l3_xyz, l3_points, l3_indices=chevb_model(l2_xyz, l2_points, 256, 0.8,nsample=32,scope='layer3',mlp=[256,256,512], output_dim=512, bn=True, pooling='max', 
is_training=is_training, bn_decay=bn_decay)
    #keypoints, idx, attention=feature_detection_module(l2_xyz, l2_points, num_clusters=256, radius=0.8, is_training=is_training,
                  #                                     mlp=[64,64,128], mlp2=[128, 64],num_samples=32, bn_decay=bn_decay)
    keypoints, idx, attention=feature_detection_module(l2_xyz, l2_points, num_clusters=256, radius=0.8, is_training=is_training,
                                                       mlp=[64,64,128], mlp2=[128, 256],num_samples=32, bn_decay=bn_decay)

    attention=l3_points
    
    attention = tf.reshape(l3_points, [batch_size, -1])
    attention = tf_util.fully_connected(attention, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    attention = tf_util.dropout(attention, keep_prob=0.5, is_training=is_training, scope='dp1')
    attention = tf_util.fully_connected(attention, 128, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    attention = tf_util.dropout(attention, keep_prob=0.5, is_training=is_training, scope='dp2')
    attention = tf_util.fully_connected(attention, 64, bn=True, is_training=is_training, scope='fc3', bn_decay=bn_decay)
    attention = tf_util.dropout(attention, keep_prob=0.5, is_training=is_training, scope='dp3')
    attention = tf_util.fully_connected(attention, l3_points.shape[1], bn=True, is_training=is_training, scope='fc4', bn_decay=bn_decay)

    #attention = tf_util.fully_connected(attention, 128, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
   # print(attention.shape)
   # Globe_W=tf.squeeze(Globe_W)
  #  attention=tf.expand_dims(attention,axis=2)
   # attention=tf.matmul(Globe_W,attention)
    #attention=tf.squeeze(attention)
  #  keypoints, idx, attention=feature_detection_module(l3_xyz, l3_points, 128,radius=0.6, is_training=is_training, mlp=[256,256,512],
  #                                                                      mlp2=[],num_samples=32,bn_decay=bn_decay)
   # print(attention.shape,attention.shape)
    #attention2=tf.matmul(attention,attention2)
   # attention=tf.reshape(attention,[batch_size,l3_points.shape[1],l3_points.shape[2]])

    #attention=tf.reduce_max(attention,axis=2)
    #attention=tf.reshape(attention,[-1,1])

    #print("attention.shape",attention.shape)
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [512,512,256], is_training, bn_decay, scope='fa_layer1')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,256,128], is_training, bn_decay, scope='fa_layer2')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer3')

    return l3_points, l3_xyz,attention
'''
def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    l0_xyz=point_cloud
    l0_points=None
    l1_xyz, l1_points, l1_indices=pointnet_sa_module(l0_xyz, l0_points, 512, 0.2,nsample=32,scope='layer1',mlp=[64,64,128],mlp2=[128,256], bn=True, pooling='max',
                                             group_all=False, is_training=is_training, bn_decay=bn_decay)
    l2_xyz, l2_points, l2_indices=chevb_model(l1_xyz, l1_points, 256, 0.4,nsample=32,scope='layer2',mlp=[128,128,256], output_dim=128,bn=True, pooling='max',
                                              is_training=is_training, bn_decay=bn_decay)
    l3_xyz, l3_points, l3_indices=chevb_model(l2_xyz, l2_points, 128, 0.8,nsample=32,scope='layer3',mlp=[256,256,512], output_dim=128, bn=True, pooling='max', 
is_training=is_training, bn_decay=bn_decay)
    # Set abstraction layers
    
    keypoints, idx, attention=feature_detection_module(l0_xyz, l0_points, num_clusters=128, radius=0.8, is_training=is_training,
                                                       mlp=[64,64,128], mlp2=[128, 256],num_samples=32, bn_decay=bn_decay)
    return l3_points, l3_xyz,attention

def triple_loss(pred1,margin=1):
    anchor_output=pred1[:BATCH_SIZE]
    positive_output=pred1[BATCH_SIZE:BATCH_SIZE*2]
    negative_output=pred1[BATCH_SIZE*2:]
    d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1)
    d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), 1)
    loss = tf.maximum(0.0, margin + d_pos - d_neg)
    loss = tf.reduce_mean(loss)
    return loss

'''
def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    l0_xyz=point_cloud
    l0_points=None
    # Set abstraction layers
    l1_xyz, l1_points, l1_indices=pointnet_sa_module_chev(l0_xyz, l0_points, npoint=256, radius=0.8, nsample=64,mlp=[64,128],mlp2=[128,256],mlp3=[256,256], is_training=is_training,
                            scope='layer1')
    keypoints, idx, attention=feature_detection_module(l0_xyz, l0_points, num_clusters=256, radius=0.8, is_training=is_training,
                                                       mlp=[64,64,128], mlp2=[128, 256],num_samples=32, bn_decay=bn_decay)

    print(l1_points.shape,attention.shape)
    return l1_points, l1_xyz,attention
'''

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def log_string_record(out_str , file):
    file.write(out_str+'\n')
    file.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def eval():
    with tf.Graph().as_default():
        with tf.device('/gpu:1'):
            pointclouds_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
          #  batch = tf.get_variable('batch', [],
           #     initializer=tf.constant_initializer(0), trainable=False)
            #bn_decay1 = get_bn_decay(batch)
            print("___________model begin")

            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            # Get model and loss 
            pred1,xyz,attention = get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            print("model compete")
        saver = tf.train.Saver()
    # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        saver.restore(sess, '/data/slan/code/registration_pointnet_graph/gazaba/4096/log/graph_2.ckpt')
        
    ops = {'pointclouds_pl': pointclouds_pl,
               'is_training_pl': is_training_pl,
                'xyz':xyz,
               'pred1': pred1,
           'attention':attention,
    #            'xyz1_':xyz1_,
    #            'xyz2_':xyz2_,
               }
    xyz1,feature1,xyz2,feature2,xyz_nms1,xyz_nms2=test(sess,ops)
    return xyz1,feature1,xyz2,feature2,xyz_nms1,xyz_nms2

def test(sess_model,ops):
    xyz1=[]
    feature1=[]
    xyz2=[]
    feature2=[]
    attention1=[]
    attention2=[]
    xyz1_=[]
    xyz2_=[]
    lxyz1=[]
    lxyz2=[]
    #feature
    for i in range(int(len(data1))):
        data_set1=data1[BATCH_SIZE*i-BATCH_SIZE:i*BATCH_SIZE]
        feed_dict = {ops['pointclouds_pl']:data1[i].reshape((1,data1[i].shape[0],data1[i].shape[1])),
                     ops['is_training_pl']: False,
                }
        xyz,feature,attention= sess_model.run( [ops['xyz'],ops['pred1'],ops['attention']], feed_dict=feed_dict)
        feature1.append(feature)
        attention1.append(attention)
        xyz1_.append(xyz)

    for i in range(1,int(len(data2))):
        data_set2=data2[BATCH_SIZE*i-BATCH_SIZE:i*BATCH_SIZE]

        feed_dict = {ops['pointclouds_pl']: data2[i].reshape((1,data2[i].shape[0],data2[i].shape[1])),
                     ops['is_training_pl']: False,
                }
        xyz,feature,attention== sess_model.run( [ops['xyz'],ops['pred1'],ops['attention']], feed_dict=feed_dict)
        feature2.append(feature)
        attention2.append(attention)
        xyz2_.append(xyz)

    xyz1_ = np.concatenate(xyz1_, axis=1)
    attention1 = np.concatenate(attention1, axis=1)
    xyz2_ = np.concatenate(xyz2_, axis=1)
    attention2 = np.concatenate(attention2, axis=1)
    for i in range(1,int(len(data1))):
       # print(i)
        data_set=real_xyz1[BATCH_SIZE*i-BATCH_SIZE:i*BATCH_SIZE]
        feed_dict = {ops['pointclouds_pl']: real_xyz1[i].reshape((1,real_xyz1[i].shape[0],real_xyz1[i].shape[1])),
                     ops['is_training_pl']: False,
                }
        xyz,feature= sess_model.run( [ops['xyz'],ops['pred1'],], feed_dict=feed_dict)
        xyz1.append(xyz)

    for i in range(1,int(len(data2))):
        data_set=real_xyz2[BATCH_SIZE*i-BATCH_SIZE:i*BATCH_SIZE]

        feed_dict = {ops['pointclouds_pl']: real_xyz2[i].reshape((1,real_xyz2[i].shape[0],real_xyz2[i].shape[1])),
                     ops['is_training_pl']: False,
                }
        xyz,feature,lxyz1_,lxyz2_= sess_model.run( [ops['xyz'],ops['pred1'],ops['xyz1_'],ops['xyz2_']], feed_dict=feed_dict)
        xyz2.append(xyz)
        lxyz1.append(lxyz1_)
        lxyz2.append(lxyz2_)
    #xyz1 = np.concatenate(xyz1, axis=1)
    #xyz2 = np.concatenate(xyz2, axis=1)
   # print(xyz1.shape,attention1.shape)
   # xyz_nms1, attention_nms1, num_keypoints1 = nms(xyz1, attention1)
   # xyz_nms2, attention_nms2, num_keypoints2 = nms(xyz2, attention2)
    xyz_nms1,xyz_nms2=[],[]
    return xyz1,feature1,xyz2,feature2,xyz_nms1,xyz_nms2,lxyz1,lxyz2

def test(sess_model,ops):
    print(data1.shape,real_xyz1.shape)
    def test_data(data):
        xyz_re=[]
        feature_re=[]
        attention_re=[]
        print("test data",data.shape)
        for i in range (1,int(len(data)/BATCH_SIZE)):  
            data_set1=data[BATCH_SIZE*i-BATCH_SIZE:i*BATCH_SIZE]
            feed_dict = {ops['pointclouds_pl']:data_set1,
                         #data_set1,
                     ops['is_training_pl']: False,
                }
            xyz,feature,attention= sess_model.run( [ops['xyz'],ops['pred1'],ops['attention']], feed_dict=feed_dict)
            feature_re.append(feature)
            attention_re.append(attention)
            xyz_re.append(xyz)   
        print(len(xyz_re))
        xyz_re=np.concatenate(xyz_re)
        feature_re=np.concatenate(feature_re)
        attention_re=np.concatenate(attention_re)
        print(xyz_re.shape)
        return  xyz_re,feature_re,attention_re

    print("data1",data1.shape)
    xyz1_,feature1,attention1=test_data(data1)
    xyz2_,feature2,attention2=test_data(data2)
    xyz1,feature1_,attention1_=test_data(real_xyz1)
    xyz2,feature2_,attention2_=test_data(real_xyz2)
    print("xyz1.shape attention.shape",xyz1.shape,attention1.shape)
    xyz1=xyz1.reshape((1,xyz1.shape[0]*xyz1.shape[1],xyz1.shape[2]))
    xyz2=xyz2.reshape((1,xyz2.shape[0]*xyz2.shape[1],xyz2.shape[2]))
    attention1=attention1.reshape((1,attention1.shape[0]*attention1.shape[1]))    
    attention2=attention2.reshape((1,attention2.shape[0]*attention2.shape[1]))

    xyz_nms1,xyz_nms2=attention1,attention2
    print(xyz1.shape)
    print(xyz1.shape)
    xyz_nms1, attention_nms1, num_keypoints1 = nms(xyz1, attention1)
    xyz_nms2, attention_nms2, num_keypoints2 = nms(xyz2, attention2)

    def find_new_xyz(filename,xyz,num=4096,interval=0):
        data=pd.read_csv(filename,sep=',')
        data=data[data.columns[1:4]]
        data=data[::4]
        if(interval!=0): 
            data=np.array(data)
            data=data.reshape((1,data.shape[0],data.shape[1]))
            new_xyz=gather_point(data,farthest_point_sample(data.shape[1]/interval, data))
            with tf.Session() as sess:
                result=new_xyz.eval()
        #data=result[0]
        print(len(data))
        data=np.array(data)
        tree = BallTree(np.array(data), leaf_size=2)
        xyz_set,real_xyz=[],[]
        for i in range(len(xyz)):
            index_raw_data_set=find_near_points(tree,xyz[i],num)
            xyz_set.append(data[index_raw_data_set]-xyz[i])
            real_xyz.append(data[index_raw_data_set])
        xyz_set=np.concatenate(xyz_set)
        real_xyz=np.concatenate(real_xyz)
        return xyz_set,real_xyz

    print("xyz_nms1",xyz_nms1.shape)
    xyz_nms1=np.squeeze(xyz_nms1)   
    xyz_nms2=np.squeeze(xyz_nms2) 
    xyz_nms1=xyz_nms1.reshape((-1,3))
    xyz_nms2=xyz_nms2.reshape((-1,3))

    pointset1,real_xyz11=find_new_xyz(cloud1,xyz_nms1,NUM_POINT,interval=scale)
    pointset2,real_xyz22=find_new_xyz(cloud2,xyz_nms2,NUM_POINT,interval=scale)
    pointset1=np.array(pointset1)
    pointset2=np.array(pointset2)
    pointset1=np.squeeze(pointset1)
    pointset2=np.squeeze(pointset2)
    real_xyz11=np.array(real_xyz11)
    real_xyz22=np.array(real_xyz22)
    real_xyz11=np.squeeze(real_xyz11)
    real_xyz22=np.squeeze(real_xyz22)
    print("feed network",pointset1.shape)
    xyz1_,feature1,attention1=test_data(pointset1)
    xyz2_,feature2,attention2=test_data(pointset2)
    xyz1,feature1_,attention1_=test_data(real_xyz11)
    xyz2,feature2_,attention2_=test_data(real_xyz22)
    print(xyz1.shape,feature1.shape,xyz2.shape)  
    return xyz1,feature1,xyz2,feature2,xyz_nms1,xyz_nms2

def nms(xyz, attention):
    #print(xyz.shape,attention.shape)
    num_models = xyz.shape[0]  # Should be equals to batch size
    num_keypoints = [0] * num_models

    xyz_nms = np.zeros((num_models, KeyPoint, 3), xyz.dtype)
    attention_nms = np.zeros((num_models, KeyPoint), xyz.dtype)

    for i in range(num_models):
        print("xyz",xyz[i, :, :].shape,"attention",attention.shape)
        nbrs = NearestNeighbors(n_neighbors=50, algorithm='ball_tree').fit(xyz[i, :, :])
        distances, indices = nbrs.kneighbors(xyz[i, :, :])
        knn_attention = attention[i, indices]
        outside_ball = distances > nms_radius
        knn_attention[outside_ball] = 0.0
        is_max = np.where(np.argmax(knn_attention, axis=1) == 0)[0]
        # Extract the top k features, filtering out weak responses
        attention_thresh = np.max(attention[i, :]) * min_response_ratio
        is_max_attention = [(attention[i, m], m) for m in is_max if attention[i, m] > attention_thresh]
        is_max_attention = sorted(is_max_attention, reverse=True)
        max_indices = [m[1] for m in is_max_attention]
        if len(max_indices) >= KeyPoint:
            max_indices = max_indices[:KeyPoint]
            num_keypoints[i] = len(max_indices)
        else:
            num_keypoints[i] = len(max_indices)  # Retrain original number of points
            max_indices = np.pad(max_indices, (0, KeyPoint- len(max_indices)), 'constant',
                                 constant_values=max_indices[0])
        xyz_nms[i, :, :] = xyz[i, max_indices, :]
        attention_nms[i, :] = attention[i, max_indices]
        
    return xyz_nms, attention_nms, num_keypoints


if __name__=='__main__':
    #read data
    data1,index1,real_xyz1=DataSet(cloud1,5,NUM_POINT,scale)
    data1=np.array(data1)
    real_xyz1=np.array(real_xyz1)
    #print(data1.shape,real_xyz1.shape)
    data2,index2,real_xyz2=DataSet(cloud2,5,NUM_POINT,scale)
    data2=np.array(data2)
    real_xyz2=np.array(real_xyz2)
    #print(data2.shape,real_xyz2.shape)
    data1=np.squeeze(data1)
    data2=np.squeeze(data2)
    real_xyz1=np.squeeze(real_xyz1)
    real_xyz2=np.squeeze(real_xyz2)
    #evaluation
    xyz1, feature1, xyz2, feature2, xyz_nms1, xyz_nms2 = eval()

    layer1 = pd.DataFrame(xyz_nms1)
    layer2 = pd.DataFrame(xyz_nms2)
    layer1.to_csv('keypoint1.txt')
    layer2.to_csv('keypoint2.txt')
    
    xyz1_can = np.concatenate(xyz1)
    feature1_can = np.concatenate(feature1)
    xyz2_can = np.concatenate(xyz2)
    feature2_can = np.concatenate(feature2)
    xyz1_can = np.squeeze(xyz1_can)
    feature1_can = np.squeeze(feature1_can)
    xyz2_can = np.squeeze(xyz2_can)
    feature2_can = np.squeeze(feature2_can)

    # save keypoints and descriptor
    with open("test1" + "keypoint.bin", "wb") as fd:
        fd.write(struct.pack('d', len(xyz1_can)))
        print(len(xyz1_can))
        for X in xyz1_can:
            # input_dtat=data_xyz.iloc[X]
            fd.write(struct.pack('d', X[0]))
            fd.write(struct.pack('d', X[1]))
            fd.write(struct.pack('d', X[2]))

    with open("feature1" + ".des.bin", "wb") as fd:
        fd.write(struct.pack('d', len(feature1_can)))
        print(len(feature1_can))
        fd.write(struct.pack('d', int(len(feature1_can[0]))))
        #  test_data=nd.array(test_data)
        for X in (feature1_can):
            for i in X:
                fd.write(struct.pack('d', i))

    with open("test2" + "keypoint.bin", "wb") as fd:
        fd.write(struct.pack('d', len(xyz2_can)))
        print(len(xyz2_can))
        for X in xyz2_can:
            # input_dtat=data_xyz.iloc[X]
            fd.write(struct.pack('d', X[0]))
            fd.write(struct.pack('d', X[1]))
            fd.write(struct.pack('d', X[2]))

    with open("feature2" + ".des.bin", "wb") as fd:
        fd.write(struct.pack('d', len(feature2_can)))
        print(len(feature2_can))
        fd.write(struct.pack('d', int(len(feature1_can[0]))))
        #  test_data=nd.array(test_data)
        for X in (feature2_can):
            for i in X:
                fd.write(struct.pack('d', i))

