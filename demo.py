# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 22:10:11 2021

@author: Amin
"""

from Demix.dNMF import DeformableNMF, ExponentialFP, SimulatedVideoDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import visualization as V
import numpy as np
import torch

# %% Simulation
K = 6
T = 40
times = np.arange(3)
sz = torch.tensor([50,50,2])
C = torch.rand(K,T)

efp = ExponentialFP(sz,K,T,positions=None)
A_tC,A_t,grid,reg = efp(times,C)


dataset = SimulatedVideoDataset(K=K,T=T,sz=sz,shape_std=3,traj_snr=[-100,-100,-100],
                                traj_means=[0,0,0],density=.9,bg_snr=-130)

batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

V.visualize_image(dataset.video[:,:,:,0].max(2)[0])
V.visualize_trajectory(dataset.positions,dataset.positions)

# %%
dnmf = DeformableNMF(sz,K,T,positions=dataset.positions[:,:,0])
optimizer = optim.Adam([dnmf.fp.beta], lr=1e-5)

for i in range(20):
    dnmf.update_motion(dataloader,optimizer,gamma=1,epochs=10)
    A_t,Y_i,Y = dnmf.update_footprints(dataloader,batch_size,sz,gamma_c=0,iter_c=10)


# %%
Y = Y.max(2)
Y_i = Y_i.max(2)
A_t = A_t.max(2)

# %%
file = ''
save = True

V.visualize_temporal(dnmf.C.cpu().numpy(),titlestr='C',save=save,file=file+'temporal')
V.visualize_spatial(dnmf.fp.A.cpu().numpy().max(2),RGB=save,save=save,file=file+'spatial-'+str(i))

V.visualize_video(video=Y[:,:,None,:]/Y.max(),save=save,file=file+'original.mp4')
V.visualize_video(video=Y_i[:,:,None,:]/Y_i.max(),save=save,file=file+'registered.mp4')
V.visualize_video(video=A_t[:,:,0,:][:,:,None,:]/A_t.max(),save=save,file=file+'pf-sample.mp4')
V.visualize_video(video=(Y-Y_i)[:,:,None,:]/(Y-Y_i).max(),save=save,file=file+'motion-resid.mp4')
