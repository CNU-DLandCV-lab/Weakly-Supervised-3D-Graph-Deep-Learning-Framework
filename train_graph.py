import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.getcwd()
BASE_DIR = os.path.dirname("__file__")
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling_nd'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
from pointnet_util import sample_and_group
import tensorflow as tf
import numpy as np
import tf_util
import spec_graph_util
from datetime import datetime
from pointnet_util import *
import h5py
from tensorflow.python.client import device_lib
from scipy.sparse.linalg.eigen.arpack import eigsh
print (device_lib.list_local_devices())
from tensorflow.python.client import device_lib

#Inputs and Parameters
bas_dir=r"/data/slan/data/train_data/"
file_list=os.listdir(r"/data/slan/data/train_data/")
def load_h5(h5_filename):
    f = h5py.File(h5_filename,'r')
    data_set1 = f['s_1']
    data_set2 = f['s_2']
    return data_set1,data_set2
data1,data2=[],[]
file_pwd=[bas_dir+i for i in file_list]
for i in file_pwd:
    dataset1,dataset2=load_h5(i)
    data1.append(dataset1)
    data2.append(dataset2)
print("concatnate")
data1 = np.concatenate(data1, 0)
data2 = np.concatenate(data2, 0)
print("concatnate end")

BATCH_SIZE=8
NUM_POINT=4096
LOG_FOUT = open(os.path.join( 'log_train.txt'), 'w')
LOG_FOUT.write(str("log")+'\n')
BASE_LEARNING_RATE=0.001
DECAY_STEP=1000
DECAY_RATE=0.7
OPTIMIZER="adam"
MAX_EPOCH=100
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(200000)
BN_DECAY_CLIP = 0.99

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
   # labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl

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
    # Set abstraction layers
    
    keypoints, idx, attention=feature_detection_module(l2_xyz, l2_points, num_clusters=256, radius=0.8, is_training=is_training,
                                                       mlp=[64,64,128], mlp2=[128, 512],num_samples=32, bn_decay=bn_decay)

    #print(l1_points.shape,attention.shape)
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

def pairwise_dist(A, B):
    ''' Computes pairwise distance

    :param A: (B x N x D) containing descriptors of A
    :param B: (B x N x D) containing descriptors of B
    :return: (B x N x N) tensor. Element[i,j,k] denotes the distance between the jth descriptor in ith model of A,
             and kth descriptor in ith model of B
    '''
    A = tf.expand_dims(A, 2)
    B = tf.expand_dims(B, 1)
    print("a shape ",A.shape," B ",B.shape)
    dist = tf.reduce_sum(tf.squared_difference(A, B), 3)
    return dist

def get_loss(features, attention,margin=1):
        """ Computes the attention weighted alignment loss as described in our paper.

        Args:
            xyz: Keypoint coordinates (Unused)
            features: List of [anchor_features, positive_features, negative_features]
            anchor_attention: Attention from anchor point clouds
            end_points: end_points, which will be augmented and returned

        Returns:
            loss, end_points
        """
        anchors=features[:BATCH_SIZE]
        positives=features[BATCH_SIZE:BATCH_SIZE*2]
        negatives=features[BATCH_SIZE*2:]
        anchor_attention=attention[:BATCH_SIZE]
        with tf.variable_scope("alignment") as sc:
            positive_dist = pairwise_dist(anchors, positives)
            negative_dist = pairwise_dist(anchors, negatives)
            best_positive = tf.reduce_min(positive_dist, axis=2)
            best_negative = tf.reduce_min(negative_dist, axis=2)
        
        with tf.variable_scope("triplet_loss") as sc:
            attention_sm = anchor_attention / tf.reduce_sum(anchor_attention, axis=1)[:, None]
            print(attention_sm.shape)
            sum_positive = tf.reduce_sum(attention_sm * best_positive, 1)
            sum_negative = tf.reduce_sum(attention_sm * best_negative, 1)
            tf.summary.histogram('normalized_attention', attention_sm)
        #    end_points['normalized_attention'] = attention_sm
          #  end_points['sum_positive'] = sum_positive
           # end_points['sum_negative'] = sum_negative
            triplet_cost = tf.maximum(0., sum_positive - sum_negative + margin)
            loss = tf.reduce_mean(triplet_cost)
        tf.summary.scalar('loss', loss)
        
        return loss

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

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:1'):
            pointclouds_pl = placeholder_inputs(BATCH_SIZE*3, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)
            # Get model and loss
            pred,xyz,attention = get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            print('pred shape ',pred.shape)
            with tf.variable_scope("LOSS") as sc:
                #loss = triple_loss(pred)
               # loss=get_loss(pred,attention)
                loss=get_loss(pred,attention)
                tf.summary.scalar('loss', loss)
            print ("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        writer = tf.summary.FileWriter("log", sess.graph)
        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(BASE_DIR, 'log/train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(BASE_DIR, 'log/test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        ops = {'pointclouds_pl': pointclouds_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
              }

        eval_acc_max_so_far = -1
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer,epoch)
           # eval_acc_epoch = eval_one_epoch(sess, ops, test_writer)
            # Save the variables to disk.
       #     if epoch % 10 == 0:
            save_path = saver.save(sess,"indoor/log/graph_2.ckpt")
            log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess_model, ops, train_writer,num_step):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    log_string(str(datetime.now()))
    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,3))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    label1=np.ones((BATCH_SIZE,1))
    label2=np.zeros((BATCH_SIZE,1))
    label3=np.concatenate((label1,label2),0)
    counter=0
    print(int(len(data1)/32))
   # np.random.shuffle(data1)
   # np.random.shuffle(data2)
    for i in range(1,1+int(len(data1)/BATCH_SIZE)):
        #no shuffle points
        data_set1=data1[BATCH_SIZE*i-BATCH_SIZE:i*BATCH_SIZE]
        data_set2=data2[BATCH_SIZE*i-BATCH_SIZE:i*BATCH_SIZE]
      #  print(32*i-BATCH_SIZE)
       # shuffle_points()
     #   data_set=data
       # print(data_set1.shape,data_set2.shape,data1.shape,32*i-BATCH_SIZE,BATCH_SIZE)
        index_shuffle=[i for i in range(BATCH_SIZE)]
        np.random.shuffle(index_shuffle)
        index_shuffle=np.array(index_shuffle)  
        different_data=data_set2[index_shuffle]

        data_set1=np.concatenate((data_set1,data_set2,different_data),0)
        data_set1=np.squeeze(data_set1)
        #print(data_set1)
        feed_dict = {ops['pointclouds_pl']: data_set1,
                     ops['is_training_pl']: is_training,
                }
        
        summary, step,loss_set,pred= sess_model.run( [ops['merged'], ops['step'],ops['loss'],ops['pred']], feed_dict=feed_dict)
        loss_sum=loss_sum+loss_set
       # print(loss_set)
        train_writer.add_summary(summary, i+num_step*len(data1)/32)
        counter=counter+1
    #    print(np.array(tf.reduce_sum(pred1-pred2)))
    print(num_step," loss ",loss_sum/len(data_set1)/32)
    
'''

#spectral gcn chevb
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
'''
#gcn
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


if __name__=='__main__':
    train()
