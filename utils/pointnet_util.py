""" Origin: PointNet++ Layers

Author: Charles R. Qi
Date: November 2017
"""

""" Added: LocalSpecGCN Layers

Author: Chu Wang
Date: July 2018
"""



import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling_nd'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util
import spec_graph_util
from spec_graph_util import spec_conv2d
from spec_graph_util import spec_conv2d_modul
from spec_graph_util import spec_hier_cluster_pool


def sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec=None, knn=False , use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        tnet_spec: dict (keys: mlp, mlp2, is_training, bn_decay), if None do not apply tnet
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if tnet_spec is not None:
        grouped_xyz = tnet(grouped_xyz, tnet_spec)

    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        if use_xyz:
            new_points = grouped_xyz
        else:
            new_points = None

    return new_xyz, new_points, idx, grouped_xyz



def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', tnet_spec=None, knn=False, use_xyz=True ):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    with tf.variable_scope(scope) as sc:
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec, knn, use_xyz)


        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay)
        if pooling=='avg':
            new_points = tf_util.avg_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='avgpool1')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg1'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True)
        elif pooling=='min':
            new_points = tf_util.max_pool2d(-1*new_points, [1,nsample], stride=[1,1], padding='VALID', scope='minpool1')
        elif pooling=='max_and_avg':
            avg_points = tf_util.max_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='maxpool1')
            max_points = tf_util.avg_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='avgpool1')
            new_points = tf.concat([avg_points, max_points], axis=-1)
        elif pooling=='cluster_pool':
            new_points = spec_cluster_pool(new_points, pool_method = 'max')
        elif pooling=='hier_cluster_pool':
            new_points = spec_hier_cluster_pool(new_points, pool_method = 'max', csize = csize)
        elif pooling=='hier_cluster_pool_ablation':
            new_points = spec_hier_cluster_pool_ablation(new_points, pool_method = init_pooling, csize = csize, recurrence = r)


        if mlp2 is None: mlp2 = []
        for i, num_out_channel in enumerate(mlp2):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv_post_%d'%(i), bn_decay=bn_decay)
        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx



# set abstraction layer with spectral graph convolution.
def pointnet_sa_module_spec(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', tnet_spec=None, knn=False, use_xyz=True , spec_conv_type = 'mlp', structure = 'spec', useloc_covmat = True, csize = None ):
    ''' PointNet Set Abstraction (SA) Module
        Input:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor
        npoint: int32 -- #points sampled in farthest point sampling
        radius: float32 -- search radius in local region
        nsample: int32 -- how many points in each local region
        mlp: list of int32 -- output size for MLP on each point
        mlp2: list of int32 -- output size for MLP on each region
        group_all: bool -- group all points into one PC if set true, OVERRIDE
        npoint, radius and nsample settings
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features

        # special to spectrual conv
        useloc_covmat: use location xyz ONLY for local neighbourhood's covariance matrix computation; Set false to use all point features including xyz + network feats.

        Return:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
        idx: (batch_size, npoint, nsample) int32 -- indices for local regions
        '''
    with tf.variable_scope(scope) as sc:
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec, knn, use_xyz)

        # grouped_xyz this is local xyz wrt to each center point
        # dim B N K 3
        # controls using what feature for covariance computation
        if useloc_covmat:
            local_cord = grouped_xyz
        else:
            local_cord = None

        if structure == 'spec':
            mlp_spec = mlp
            new_points, UT  = spec_conv2d(inputs = new_points, num_output_channels = mlp_spec,
                                             local_cord = local_cord,
                                             bn=bn, is_training=is_training,
                                             scope='spec_conv%d'%(0), bn_decay=bn_decay)

        if structure == 'spec-modul':
            mlp_spec = mlp
            new_points, UT  = spec_conv2d_modul(inputs = new_points, num_output_channels = mlp_spec,
                                             local_cord = local_cord,
                                             bn=bn, is_training=is_training,
                                             scope='spec_conv%d'%(0), bn_decay=bn_decay)



        if pooling=='avg':
            new_points = tf_util.avg_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='avgpool1')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg1'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True)
        elif pooling=='min':
            new_points = tf_util.max_pool2d(-1*new_points, [1,nsample], stride=[1,1], padding='VALID', scope='minpool1')
        elif pooling=='max_and_avg':
            avg_points = tf_util.max_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='maxpool1')
            max_points = tf_util.avg_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='avgpool1')
            new_points = tf.concat([avg_points, max_points], axis=-1)
        elif pooling=='hier_cluster_pool':
            new_points = spec_hier_cluster_pool(new_points, pool_method = 'max', csize = csize)


        if mlp2 is None: mlp2 = []
        for i, num_out_channel in enumerate(mlp2):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv_post_%d'%(i), bn_decay=bn_decay)
        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx
    
'''
def chevb_model(xyz, points, npoint, radius, nsample,scope ,mlp, bn_decay,bn=True, pooling='max',is_training=True):
    with tf.variable_scope(scope) as sc:
       # if group_all:
           # nsample = xyz.get_shape()[1].value
         #   new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
     #   else:
        with tf.variable_scope("sample_and_group") as sc:

            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, None, True, True)
        # grouped_xyz this is local xyz wrt to each center point
        # dim B N K 3
        # controls using what feature for covariance computation
        with tf.variable_scope("MLP") as sc:
            for i, num_out_channel in enumerate(mlp):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay)
        with tf.variable_scope("Laplacian") as sc:
            local_cord = grouped_xyz
            in_shape = new_points.get_shape().as_list()
    # get graph adj matrix
            W = spec_graph_util.get_adj_mat_dist_euclidean(local_cord[:,:,:,0:3] , flag_normalized = True)
            W = tf.identity(W, name='adjmat')
            W_knn = spec_graph_util.cov_mat_k_nn_graph(W, k = 32 )
            W_knn = spec_graph_util.corv_mat_setdiag_zero(W_knn)
            W_knn = tf.identity(W_knn, name='adjmat_knn')
            L = spec_graph_util.corv_mat_laplacian0(W_knn , flag_normalized = True)
            L = tf.identity(L, name='laplacian')
            x=tf.matmul(L,new_points)
            W_spec=tf.get_variable('weights_spec',[npoint,x.shape[-2],x.shape[-1]],initializer=tf.constant_initializer(0.0),dtype=tf.float32)
            new_points_=x*W_spec
            new_points_ = tf.reduce_max(new_points_, axis=[2], keep_dims=True)
            new_points_ = tf.squeeze(new_points_, [2]) # (batch_size, npoints, mlp2[-1])
    #new_points_=tf.matmul(x,W_spec)
    return new_xyz, new_points_, idx

def chevb_model(xyz, points, npoint, radius, nsample,scope ,mlp, bn_decay,output_dim,bn=True, pooling='max',is_training=True,k=5):
    with tf.variable_scope(scope) as sc:
        
       # if group_all:
           # nsample = xyz.get_shape()[1].value
         #   new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
     #   else:
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
         
            W_spec=tf.get_variable('weights_spec',[x.shape[-1],output_dim],initializer=tf.random_normal_initializer(mean=0, stddev=1),dtype=tf.float32)
            print(x.shape)
            x=tf.transpose(x,[2,0,1,3])
            x1=tf.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],x.shape[3]))            
            new_points_=tf.matmul(x1,W_spec)
            new_points_=tf.reshape(new_points_,(x.shape[0],x.shape[1],x.shape[2],output_dim))
            new_points_=tf.transpose(new_points_,[1,2,0,3])
       
            for i, num_out_channel in enumerate(output_dim):
                x = tf_util.conv2d(x, num_out_channel, [1,1],
                                    padding='VALID', stride=[1,1],
                                    bn=bn, is_training=is_training,
                                    scope='convGCN%d'%(i), bn_decay=bn_decay)
        x=tf.nn.relu(x)
        x = tf.reduce_max(x, axis=[2], keep_dims=True)
        #print(x.shape)
        x = tf.squeeze(x, [2])
    return new_xyz, x, idx

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
            x2=x
            #gcn
            W_attention=tf.get_variable('weights_attention',[x.shape[-1],output_dim],initializer=tf.random_normal_initializer(mean=0, stddev=1),dtype=tf.float32)      
            print(x.shape)
            x1=tf.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],x.shape[3]))
            new_points_=tf.matmul(x1,W_attention)
            new_points_=tf.reshape(new_points_,(x.shape[0],x.shape[1],x.shape[2],output_dim))
           # new_points_=tf.transpose(new_points_,[1,2,0,3])
          #  new_points_=tf.nn.relu(new_points_)
           # new_points_ = tf.reduce_max(new_points_, axis=[3], keep_dims=True)
            #new_points_ = tf.squeeze(new_points_, [3])            
            #gcn attention 
            attention=tf.nn.softmax(new_points_)
            print("new_points_ shape",new_points_.shape,x.shape)
            x=attention*x2
            W_spec=tf.get_variable('weights_spec',[x.shape[-1],output_dim],initializer=tf.random_normal_initializer(mean=0, stddev=1),dtype=tf.float32)      
            print(x.shape)
            #x=tf.transpose(x,[2,0,1,3])
            x1=tf.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],x.shape[3]))
            new_points_=tf.matmul(x1,W_spec)
            new_points_=tf.reshape(new_points_,(x.shape[0],x.shape[1],x.shape[2],output_dim))
           # new_points_=tf.transpose(new_points_,[1,2,0,3])
            new_points_=tf.nn.relu(new_points_)
            new_points_ = tf.reduce_max(new_points_, axis=[2], keep_dims=True)
            new_points_ = tf.squeeze(new_points_, [2])            
            print("new_points_ shape",new_points_.shape)
            
            
    return new_xyz, new_points_, idx




def chevb_model(xyz, points, npoint, radius, nsample,scope ,mlp, bn_decay,output_dim,bn=True, pooling='max',is_training=True,K=10):
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
            W_knn = spec_graph_util.cov_mat_k_nn_graph(W, k = nsample )
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
           # 
            
            #cheby begin    
                
           # new_points = tf.reduce_max(new_points, axis=[2], name='maxpool')
            
            print(new_points.get_shape(),L.shape)
            B,N,M,Fin=new_points.get_shape()
            x0=new_points
            x = tf.expand_dims(x0, 0)
            def concat(x, x_):
                x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
                return tf.concat([x, x_], axis=0)  # K x M x Fin*N

            if K > 1:
                x1 = tf.matmul(L, x0)
                print(x.shape,x1.shape)
                x = concat(x, x1)
            for k in range(2, K):
                x2 = 2 * tf.matmul(L, x1) - x0
                x = concat(x, x2)
                x0, x1 = x1, x2
        # K x N x M x Fin
            x = tf.transpose(x, perm=[1, 2, 3,4, 0])  # N x M x Fin x K
            x = tf.reshape(x, [x.shape[0] * x.shape[1]*x.shape[2], x.shape[3] * K]) 
        
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
            W = tf.get_variable('weights_cheby',[Fin * K, output_dim],initializer=tf.random_normal_initializer(mean=0, stddev=1),dtype=tf.float32)
            x = tf.matmul(x, W) 
            x=tf.reshape(x, [B,N, M, output_dim])
            #cheby end

            x=tf.nn.relu(x)
            x=tf.reduce_max(x, axis=[2])
    return new_xyz, x, idx


'''

def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1
def feature_detection_module(xyz, points, num_clusters, radius, is_training, mlp, mlp2, num_samples=64, use_bn=True):
    """ Detect features in point cloud

    Args:
        xyz (tf.Tensor): Input point cloud of size (batch_size, ndataset, 3)
        points (tf.Tensor): Point features. Unused in 3DFeat-Net
        num_clusters (int): Number of clusters to extract. Set to -1 to use all points
        radius (float): Radius to consider for feature detection
        is_training (tf.placeholder): Set to True if training, False during evaluation
        mlp: list of int32 -- output size for MLP on each point
        mlp2: list of int32 -- output size for MLP on each region. Set to None or [] to ignore
        num_samples: Maximum number of points to consider per cluster
        use_bn: bool -- Whether to perform batch normalization

    Returns:
        new_xyz: Cluster centers
        idx: Indices of points sampled for the clusters
        attention: Output attention weights
        orientation: Output orientation (radians)
        end_points: Unused

    """
    end_points = {}
   # new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec, knn, use_xyz)
    new_xyz = gather_point(xyz, farthest_point_sample(num_clusters, xyz)) # Sample point centers
    new_points, idx = query_and_group_points(xyz, points, new_xyz, num_samples, radius, knn=False, use_xyz=True,
                                             normalize_radius=True, orientations=None)  # Extract clusters

    # Pre pooling MLP
    for i, num_out_channel in enumerate(mlp):
        new_points = tf_util.conv2d(new_points, num_out_channel, kernel_size=[1, 1], stride=[1, 1], padding='VALID',
                            bn=use_bn, is_training=is_training,
                            scope='conv%d' % (i) )

    # Max Pool
    new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True)

    # Max pooling MLP
    if mlp2 is None: mlp2 = []
    for i, num_out_channel in enumerate(mlp2):
        new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=use_bn, is_training=is_training,
                            scope='conv_post_%d' % (i))
    attention = tf_util.conv2d(new_points, 1, [1, 1], stride=[1, 1], padding='VALID',
                        bn=False, scope='attention')
    attention = tf.squeeze(attention, axis=[2, 3])
 # new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
        #                                padding='VALID', stride=[1,1],
    #                                    bn=bn, is_training=is_training,
         #                               scope='conv%d'%(i), bn_decay=bn_decay)

    return new_xyz, idx, attention
def query_and_group_points(xyz, points, new_xyz, nsample, radius, knn=False,
                           use_xyz=True, normalize_radius=True, orientations=None):

    if knn:
        _, idx = knn_point(nsample, xyz, new_xyz)
        pts_cnt = nsample  # Hack. By right should make sure number of input points < nsample
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)

    tf.summary.histogram('pts_cnt', pts_cnt)

    # Group XYZ coordinates
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)
    grouped_xyz = grouped_xyz - tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization
    if normalize_radius:
        grouped_xyz /= radius  # Scale normalization
    # 2D-rotate via orientations if necessary
    if orientations is not None:
        cosval = tf.expand_dims(tf.cos(orientations), axis=2)
        sinval = tf.expand_dims(tf.sin(orientations), axis=2)
        grouped_xyz = tf.stack([cosval * grouped_xyz[:, :, :, 0] + sinval * grouped_xyz[:, :, :, 1],
                                -sinval * grouped_xyz[:, :, :, 0] + cosval * grouped_xyz[:, :, :, 1],
                                grouped_xyz[:, :, :, 2]], axis=3)

    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, nsample, channel)
        if use_xyz:

            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_points, idx

def feature_detection_module(xyz, points, num_clusters, radius, is_training, mlp, mlp2, bn_decay, num_samples=32,bn=True):
    """ Detect features in point cloud

    Args:
        xyz (tf.Tensor): Input point cloud of size (batch_size, ndataset, 3)
        points (tf.Tensor): Point features. Unused in 3DFeat-Net
        num_clusters (int): Number of clusters to extract. Set to -1 to use all points
        radius (float): Radius to consider for feature detection
        is_training (tf.placeholder): Set to True if training, False during evaluation
        mlp: list of int32 -- output size for MLP on each point
        mlp2: list of int32 -- output size for MLP on each region. Set to None or [] to ignore
        num_samples: Maximum number of points to consider per cluster
        use_bn: bool -- Whether to perform batch normalization

    Returns:
        new_xyz: Cluster centers
        idx: Indices of points sampled for the clusters
        attention: Output attention weights
        orientation: Output orientation (radians)
        end_points: Unused

    """
    
    
    new_xyz, new_points, idx, grouped_xyz = sample_and_group(num_clusters, radius, num_samples, xyz, points, None, True, True)
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
    # Pre pooling MLP
    with tf.variable_scope("MLP") as sc:
            for i, num_out_channel in enumerate(mlp):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay)

    # Max Pool
    new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True)

    # Max pooling MLP
    if mlp2 is None: mlp2 = []
    for i, num_out_channel in enumerate(mlp2):
        new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=bn, is_training=is_training,
                            scope='conv_post_%d' % (i))

        
    with tf.variable_scope("GCN") as sc:
        x=new_points
        
        x2=x
        W_attention=tf.get_variable('weights_attention',[x.shape[-1],x.shape[-1]],initializer=tf.random_normal_initializer(mean=0, stddev=1),dtype=tf.float32)      
            #print(x.shape)
        x1=tf.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],x.shape[3]))
        new_points_=tf.matmul(x1,W_attention)
        new_points_=tf.reshape(new_points_,(x.shape[0],x.shape[1],x.shape[2],x.shape[-1]))           
            #gcn attention 
        attention=tf.nn.softmax(new_points_)
        x=attention*x2
        
        W_spec=tf.get_variable('weights_spec',[x.shape[-1],256],initializer=tf.random_normal_initializer(mean=0, stddev=1),dtype=tf.float32)      
        x1=tf.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],x.shape[3]))
        new_points_=tf.matmul(x1,W_spec)
        new_points_=tf.reshape(new_points_,(x.shape[0],x.shape[1],x.shape[2],256))
           # new_points_=tf.transpose(new_points_,[1,2,0,3])
        new_points_=tf.nn.relu(new_points_)
        new_points_ = tf.reduce_max(new_points_, axis=[2], keep_dims=True)
        #new_points_ = tf.squeeze(new_points_, [2])  
    attention = tf_util.conv2d(new_points_, 1, [1, 1], stride=[1, 1], padding='VALID',
                        bn=False, scope='attention')
    print("attention",new_points_.shape)
    attention = tf.squeeze(attention, axis=[2, 3])

    
    
 #   orientation_xy = tf_util.conv2d(new_points, 2, [1, 1], stride=[1, 1], padding='VALID',
   #                         activation=None, bn=False, scope='orientation', reuse=False)
   # orientation_xy = tf.squeeze(orientation_xy, axis=2)
   # orientation_xy = tf.nn.l2_normalize(orientation_xy, dim=2, epsilon=1e-8)
   # orientation = tf.atan2(orientation_xy[:, :, 1], orientation_xy[:, :, 0])

    return new_xyz, idx, attention
def pointnet_sa_module_chev(xyz, points, npoint, radius, nsample, mlp, mlp2, mlp3, is_training, scope, bn=True, bn_decay=None,
                       tnet_spec=None, knn=False, use_xyz=True,
                       keypoints=None, orientations=None, normalize_radius=True, final_relu=True):
    """ PointNet Set Abstraction (SA) Module. Modified to remove unneeded components (e.g. pooling),
        normalize points based on radius, and for a third layer of MLP

    Args:
        xyz (tf.Tensor): (batch_size, ndataset, 3) TF tensor
        points (tf.Tensor): (batch_size, ndataset, num_channel)
        npoint (int32): #points sampled in farthest point sampling
        radius (float): search radius in local region
        nsample (int): Maximum points in each local region
        mlp: list of int32 -- output size for MLP on each point
        mlp2: list of int32 -- output size for MLP after max pooling concat
        mlp3: list of int32 -- output size for MLP after second max pooling
        is_training (tf.placeholder): Indicate training/validation
        scope (str): name scope
        bn (bool): Whether to perform batch normalizaton
        bn_decay: Decay schedule for batch normalization
        tnet_spec: Unused in Feat3D-Net. Set to None
        knn: Unused in Feat3D-Net. Set to False
        use_xyz: Unused in Feat3D-Net. Set to True
        keypoints: If provided, cluster centers will be fixed to these points (npoint will be ignored)
        orientations (tf.Tensor): Containing orientations from the detector
        normalize_radius (bool): Whether to normalize coordinates [True] based on cluster radius.
        final_relu: Whether to use relu as the final activation function

    Returns:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
        idx: (batch_size, npoint, nsample) int32 -- indices for local regions

    """

    with tf.variable_scope(scope) as sc:
        if npoint is None:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
           # new_xyz, new_points, idx, grouped_xyz, end_points = sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec,
           #                                                                      knn, use_xyz,
            #                                                                     keypoints=keypoints,
             #                                                                    orientations=orientations,
              #                                                                   normalize_radius=normalize_radius)
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, None, True, True)
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='1conv%d'%(i), bn_decay=bn_decay)
        
        
        with tf.variable_scope("Laplacian1") as sc:
            local_cord = grouped_xyz
            in_shape = new_points.get_shape().as_list()
            W = spec_graph_util.get_adj_mat_dist_euclidean(local_cord[:,:,:,0:3] , flag_normalized = True)
            W = tf.identity(W, name='adjmat')
            W_knn = spec_graph_util.cov_mat_k_nn_graph(W, k = nsample )
            W_knn = spec_graph_util.corv_mat_setdiag_zero(W_knn)
            W_knn = tf.identity(W_knn, name='adjmat_knn')
            L = spec_graph_util.corv_mat_laplacian0(W_knn , flag_normalized = True)
            L = tf.identity(L, name='laplacian')
            x=tf.matmul(L,new_points)
           
            
            x2=x
            W_attention=tf.get_variable('weights_attention',[x.shape[-1],mlp[-1]],initializer=tf.random_normal_initializer(mean=0, stddev=1),dtype=tf.float32)      
            print(x.shape)
            #x=tf.transpose(x,[2,0,1,3])
            x1=tf.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],x.shape[3]))
            new_points_=tf.matmul(x1,W_attention)
            new_points_=tf.reshape(new_points_,(x.shape[0],x.shape[1],x.shape[2],mlp[-1]))
       
            #gcn attention 
            attention=tf.nn.softmax(new_points_)
            print("new_points_ shape",new_points_.shape,x.shape)
            x=attention*x2
            W_spec=tf.get_variable('weights_spec',[x.shape[-1],mlp[-1]],initializer=tf.random_normal_initializer(mean=0, stddev=1),dtype=tf.float32)      
            print(x.shape)
            x1=tf.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],x.shape[3]))
            new_points=tf.matmul(x1,W_spec)
            new_points=tf.reshape(new_points,(x.shape[0],x.shape[1],x.shape[2],mlp[-1]))
            new_points=tf.nn.relu(new_points)
           # new_points_ = tf.reduce_max(new_points_, axis=[2], keep_dims=True)
            #new_points_ = tf.squeeze(new_points_, [2])            
            print("new_points_ shape",new_points_.shape)
        # Max pool
        pooled = tf.reduce_max(new_points, axis=[2], keep_dims=True)
        pooled_expand = tf.tile(pooled, [1, 1, new_points.shape[2], 1])

        # Concatenate
        new_points = tf.concat((new_points, pooled_expand), axis=3)

        if mlp2 is None: mlp2 = []
        for i, num_out_channel in enumerate(mlp2):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='2conv%d'%(i), bn_decay=bn_decay)
        with tf.variable_scope("Laplacian2") as sc:
            local_cord = grouped_xyz
            in_shape = new_points.get_shape().as_list()
            W = spec_graph_util.get_adj_mat_dist_euclidean(local_cord[:,:,:,0:3] , flag_normalized = True)
            W = tf.identity(W, name='adjmat')
            W_knn = spec_graph_util.cov_mat_k_nn_graph(W, k = nsample )
            W_knn = spec_graph_util.corv_mat_setdiag_zero(W_knn)
            W_knn = tf.identity(W_knn, name='adjmat_knn')
            L = spec_graph_util.corv_mat_laplacian0(W_knn , flag_normalized = True)
            L = tf.identity(L, name='laplacian')
            x=tf.matmul(L,new_points)
           
            
            x2=x
            W_attention=tf.get_variable('weights_attention',[x.shape[-1],mlp2[-1]],initializer=tf.random_normal_initializer(mean=0, stddev=1),dtype=tf.float32)      
            print(x.shape)
            #x=tf.transpose(x,[2,0,1,3])
            x1=tf.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],x.shape[3]))
            new_points_=tf.matmul(x1,W_attention)
            new_points_=tf.reshape(new_points_,(x.shape[0],x.shape[1],x.shape[2],mlp2[-1]))
       
            #gcn attention 
            attention=tf.nn.softmax(new_points_)
            print("new_points_ shape",new_points_.shape,x.shape)
            x=attention*x2
            W_spec=tf.get_variable('weights_spec',[x.shape[-1],mlp2[-1]],initializer=tf.random_normal_initializer(mean=0, stddev=1),dtype=tf.float32)      
            print(x.shape,W_spec.shape)
            x1=tf.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],x.shape[3]))
            new_points=tf.matmul(x1,W_spec)
            print(new_points.shape)
            new_points=tf.reshape(new_points,(x.shape[0],x.shape[1],x.shape[2],mlp2[-1]))
            new_points=tf.nn.relu(new_points)
            pooled = tf.reduce_max(new_points, axis=[2], keep_dims=True)
            pooled_expand = tf.tile(pooled, [1, 1, new_points.shape[2], 1])

        # Concatenate
        #new_points = tf.concat((new_points, pooled_expand), axis=3)
        # Max pool again
        #new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True)

        if mlp3 is None:
            mlp3 = []
        for i, num_out_channel in enumerate(mlp3):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='3conv%d'%(i), bn_decay=bn_decay)
        with tf.variable_scope("Laplacian3") as sc:
            local_cord = grouped_xyz
            in_shape = new_points.get_shape().as_list()
            W = spec_graph_util.get_adj_mat_dist_euclidean(local_cord[:,:,:,0:3] , flag_normalized = True)
            W = tf.identity(W, name='adjmat')
            W_knn = spec_graph_util.cov_mat_k_nn_graph(W, k = nsample )
            W_knn = spec_graph_util.corv_mat_setdiag_zero(W_knn)
            W_knn = tf.identity(W_knn, name='adjmat_knn')
            L = spec_graph_util.corv_mat_laplacian0(W_knn , flag_normalized = True)
            L = tf.identity(L, name='laplacian')
            x=tf.matmul(L,new_points)
           
            
            x2=x
            W_attention=tf.get_variable('weights_attention',[x.shape[-1],mlp3[-1]],initializer=tf.random_normal_initializer(mean=0, stddev=1),dtype=tf.float32)      
            print(x.shape)
            #x=tf.transpose(x,[2,0,1,3])
            x1=tf.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],x.shape[3]))
            new_points_=tf.matmul(x1,W_attention)
            new_points_=tf.reshape(new_points_,(x.shape[0],x.shape[1],x.shape[2],mlp3[-1]))
       
            #gcn attention 
            attention=tf.nn.softmax(new_points_)
            print("new_points_ shape",new_points_.shape,x.shape)
            x=attention*x2
            W_spec=tf.get_variable('weights_spec',[x.shape[-1],mlp3[-1]],initializer=tf.random_normal_initializer(mean=0, stddev=1),dtype=tf.float32)      
            print(x.shape)
            x1=tf.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],x.shape[3]))
            new_points=tf.matmul(x1,W_spec)
            new_points=tf.reshape(new_points,(x.shape[0],x.shape[1],x.shape[2],mlp3[-1]))
            new_points=tf.nn.relu(new_points)
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True)
            new_points = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp2[-1])

        return new_xyz, new_points, idx
'''
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
            x2=x
            #gcn
        with tf.variable_scope("GCN") as sc:
            
            W_attention=tf.get_variable('weights_attention',[x.shape[-1],x.shape[-1]],initializer=tf.random_normal_initializer(mean=0, stddev=1),dtype=tf.float32)      
            print(x.shape)
            x1=tf.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],x.shape[3]))
            new_points_=tf.matmul(x1,W_attention)
            new_points_=tf.reshape(new_points_,(x.shape[0],x.shape[1],x.shape[2],x.shape[-1]))           
            #gcn attention 
            attention=tf.nn.softmax(new_points_)
            x=attention*x2
            
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
'''

def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:                                                                                                      
            xyz1: (batch_size, ndataset1, 3) TF tensor                                                              
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1                                           
            points1: (batch_size, ndataset1, nchannel1) TF tensor                                                   
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point                                                 
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)#克里金插值
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1