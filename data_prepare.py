

# coding: utf-8
import h5py
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import utils
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import math
import scipy.linalg as linalg
import matplotlib as mpl 
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt 
import random

def find_near_points(x,y,z,data,search_radius):
    x1=x+search_radius
    x_1=x-search_radius
    y1=y+search_radius
    y_1=y-search_radius
    z1=z+search_radius
    z_1=z-search_radius
    pointset=data[(data['x']<x1) & (data['x']>x_1) & (data['y']<y1) & (data['y']>y_1) & (data['z']<z1) & (data['z']>z_1)]
    return pointset


def find_near_points1(data,tree,search_radius):
    data=np.array(data)
    pointset=tree.query_radius(data.reshape((1,3)),search_radius)
    return pointset


def computeTDF(voxel_grid_occ,voxel_grid_dim_x,voxel_grid_dim_y,
               voxel_grid_dim_z,voxel_grid_orgin_x,voxel_grid_orgin_y,voxel_grid_orgin_z,voxel_size,trunc_margin):
    voxel_grid_TDF=np.zeros((voxel_grid_dim_x,voxel_grid_dim_y,voxel_grid_dim_z))
    mod = SourceModule("""
__global__ void ComputeTDF( float * voxel_grid_occ, float * voxel_grid_TDF) {

    int voxel_idx = threadIdx.x+blockIdx.x*blockDim.x;
      if (voxel_idx > (27000))
        return;
    int voxel_grid_dim_x,voxel_grid_dim_y,voxel_grid_dim_z;
    voxel_grid_dim_x=voxel_grid_dim_y=voxel_grid_dim_z=30;
    int pt_grid_z = (int)floor((float)voxel_idx / ((float)voxel_grid_dim_x * (float)voxel_grid_dim_y));
  int pt_grid_y = (int)floor(((float)voxel_idx - ((float)pt_grid_z * (float)voxel_grid_dim_x * (float)voxel_grid_dim_y)) / (float)voxel_grid_dim_x);
  int pt_grid_x = (int)((float)voxel_idx - ((float)pt_grid_z * (float)voxel_grid_dim_x * (float)voxel_grid_dim_y) - ((float)pt_grid_y * (float)voxel_grid_dim_x));

  int search_radius = (int)round(15 / 0.1);

  if (voxel_grid_occ[voxel_idx] > 0) {a
    voxel_grid_TDF[voxel_idx] = 1.0f; // on surface
    return;
  }
    // Find closest surface point
    for (int iix = max(0, pt_grid_x - search_radius); iix < min(voxel_grid_dim_x, pt_grid_x + search_radius + 1); ++iix)
    for (int iiy = max(0, pt_grid_y - search_radius); iiy < min(voxel_grid_dim_y, pt_grid_y + search_radius + 1); ++iiy)
      for (int iiz = max(0, pt_grid_z - search_radius); iiz < min(voxel_grid_dim_z, pt_grid_z + search_radius + 1); ++iiz) {
        int iidx = iiz * voxel_grid_dim_x * voxel_grid_dim_y + iiy * voxel_grid_dim_x + iix;
        if (voxel_grid_occ[iidx] > 0) {
          float xd = (float)(pt_grid_x - iix);
          float yd = (float)(pt_grid_y - iiy);
          float zd = (float)(pt_grid_z - iiz);
          float dist = sqrtf(xd * xd + yd * yd + zd * zd) / (float)search_radius;
          if ((1.0f - dist) > voxel_grid_TDF[voxel_idx] && (1.0f - dist) >0)
            voxel_grid_TDF[voxel_idx] = 1.0f - dist;
        }
      }
}
""")
    func = mod.get_function("ComputeTDF")   
    func(drv.In(voxel_grid_occ),  drv.InOut(voxel_grid_TDF),
         block=( 300,1,1) ,grid=(90,1))
    return voxel_grid_TDF


def OneTDF1(data,voxel_size,voxel_grid_dim_x=30,voxel_grid_dim_y=30,voxel_grid_dim_z=30):
 #   print(data.iloc[0,0],data.iloc[0,1],data.iloc[0,2])
    voxel_grid_orgin_x=data.iloc[0,0]-voxel_grid_dim_x/2*voxel_size
    voxel_grid_orgin_y=data.iloc[0,1]-voxel_grid_dim_y/2*voxel_size
    voxel_grid_orgin_z=data.iloc[0,2]-voxel_grid_dim_z/2*voxel_size
   # print(data.iloc[0],voxel_grid_orgin_x,voxel_grid_orgin_y,voxel_grid_orgin_z)
    counter=0
    voxel_grid_occ=np.zeros((voxel_grid_dim_x,voxel_grid_dim_y,voxel_grid_dim_z))
    x=np.array(data['x'])
    y=np.array(data['y'])
    z=np.array(data['z'])
    voxel_size_gpu=np.array(voxel_size)
    gpu_tid_size=np.array(len(x))

   # print(voxel_size_gpu)
    voxel_grid_occ_list=np.zeros((len(x),))
    #print("y_occ_shape",y.shape)
    Nx=np.ones(1,dtype=np.float32)
    Nx[0]=voxel_grid_orgin_x
    Ny=np.ones(len(x),dtype=np.float32)
    Ny[0]=voxel_grid_orgin_y
    Nz=np.ones(len(x),dtype=np.float32)
    Nz[0]=voxel_grid_orgin_z
 #   print(Nx)
    voxel_grid_occ_list=produce_occ(x.astype(np.float32),y.astype(np.float32),z.astype(np.float32),voxel_grid_occ.astype(np.float32),
                                    Nx.astype(np.float32),Ny.astype(np.float32),Nz.astype(np.float32),voxel_size_gpu.astype(np.float32))
   # print(voxel_grid_occ)
   # voxel_grid_TDF=computeTDF(voxel_grid_occ,voxel_grid_dim_x,voxel_grid_dim_y,
         #      voxel_grid_dim_z,voxel_grid_orgin_x,voxel_grid_orgin_y,voxel_grid_orgin_z,voxel_size,5)
    return voxel_grid_occ_list


def produce_occ(x,y,z,voxel_grid_occ,voxel_grid_orgin_x,voxel_grid_orgin_y,voxel_grid_orgin_z,voxel_size):
    gpu_tid_size=np.array(len(x))
    #print("gpu x ",x)
    #print('gpu_tid_size',gpu_tid_size)
    mod = SourceModule("""
__global__ void ComputeTDF( float *x,float *y,float *z,float * voxel_grid_occ, 
float* voxel_grid_orgin_x, float* voxel_grid_orgin_y, float* voxel_grid_orgin_z,float* voxel_size,float* gpu_tid_size) {
        int tid = threadIdx.x+blockIdx.x*blockDim.x;
       // if(tid>gpu_tid_size[0])
    //        return;
        int pt_grid_x=round((x[tid]-voxel_grid_orgin_x[0])/voxel_size[0]);
        int pt_grid_y=round((y[tid]-voxel_grid_orgin_y[0])/voxel_size[0]);     
        int pt_grid_z=round((z[tid]-voxel_grid_orgin_z[0])/voxel_size[0]);
       // voxel_grid_occ[tid]=pt_grid_y;
        if(pt_grid_x<30 && pt_grid_y<30 &&pt_grid_z<30&& pt_grid_x>0 && pt_grid_y>0 &&pt_grid_z>0)
            voxel_grid_occ[pt_grid_z*30*30+pt_grid_y*30+pt_grid_x]=1;
}
""")
    func = mod.get_function("ComputeTDF")   
    func(drv.InOut(x),drv.In(y),drv.In(z),drv.InOut(voxel_grid_occ),
         drv.In(voxel_grid_orgin_x), drv.In(voxel_grid_orgin_y),drv.In(voxel_grid_orgin_z),drv.In(voxel_size),drv.In(gpu_tid_size),
         block=( 500,1,1) ,grid=(100,1))
   # print(x)
    return voxel_grid_occ


for index in range(1,45):
    data_registrated=pd.read_csv(r"/data/global_frame/"+"PointCloud"+str(index)+".csv",sep=',')
    data_befor_registrate=pd.read_csv(r"/data/local_frame/"+"Hokuyo_"+str(index)+".csv",sep=',')
    print("PointCloud"+str(index)+".csv")
    data_registrated=data_registrated[data_registrated.columns[1:4]]
    data_befor_registrate=data_befor_registrate[data_befor_registrate.columns[1:4]]
    dset1=[]
    dset2=[]
    print(data_befor_registrate.shape,data_registrated.shape)
    index_shuffle=[i for i in range(len(data_befor_registrate))]
    np.random.shuffle(index_shuffle)
    index_shuffle=np.array(index_shuffle)
    counter=0
    for i in index_shuffle[:1000]:
        #print(i)
        if(i==0):
            continue
        x,y,z=data_registrated.iloc[i]
        pointset=find_near_points(x,y,z,data_registrated,0.1*15)
        pointset.iloc[0]=data_registrated.iloc[i]
        dset1.append(OneTDF1(pointset,0.1))
        counter=counter+1
    for i in index_shuffle[:1000]:
        if(i==0):
            continue
        i=i-1
        x,y,z=data_befor_registrate.iloc[i]
        pointset=find_near_points(x,y,z,data_befor_registrate,0.1*15)
        pointset.iloc[0]=data_befor_registrate.iloc[i]
        dset2.append(OneTDF1(pointset,0.1))

    f = h5py.File(r"/data/trainsets/"+"learn_h5py"+str(index)+".h5", 'w')
    f['anchor']=dset1
    f['pos']=dset2
    f.close()
