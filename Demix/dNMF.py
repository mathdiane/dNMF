# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:08:46 2021

@author: Amin
"""
from Methods.Demix.WUtils import Simulator
from torch.utils.data import Dataset
import torch.nn.functional as F
from scipy.io import loadmat
import torch.nn as nn
import numpy as np
import scipy as sp
import torch

device = 'cuda'

class ExponentialFP(nn.Module):
    def __init__(self,sz,K,T,positions=None):
        super().__init__()
        
        flow_id = torch.cat(torch.where(torch.ones(list(sz)))).reshape([3]+list(sz)).permute([1,2,3,0]).float().to(device)
        transformed = ExponentialFP.quadratic_basis(flow_id).to(device)
        self.beta = torch.cat((torch.zeros(1,3),
                          torch.eye(3),
                          torch.zeros(6,3)),0)[:,:,None].repeat(1,1,T).to(device)
        self.beta.requires_grad=True
        
        self.sigma = (torch.ones(K)*3).to(device)
        if positions is None:
            self.pos = (1+torch.rand(K,3)*sz[None,:]).to(device)
        else:
            self.pos = positions.to(device)
            
        
        self.flow_id = flow_id
        self.transformed = transformed
        self.sz = sz.to(device)
        self.A = torch.exp((-(self.flow_id[:,:,:,:,None] - self.pos.T[None,None,None,:,:])**2/
                 self.sigma[None,None,None,None,:]**2).sum(3))


        pass
        
    # Spatial transformer network forward function
    @staticmethod
    def quadratic_basis(P):
        return torch.cat((P[:,:,:,0][:,:,:,None]*0+1,P[:,:,:,:],P[:,:,:,:]*P[:,:,:,:],
                          P[:,:,:,0][:,:,:,None]*P[:,:,:,1][:,:,:,None],
                          P[:,:,:,0][:,:,:,None]*P[:,:,:,2][:,:,:,None],
                          P[:,:,:,1][:,:,:,None]*P[:,:,:,2][:,:,:,None]),3)
        
    def forward(self, times, C):
        grid = torch.einsum('mnza,abt->mnzbt', self.transformed, self.beta[:,:,times])
        grid = 2*grid/(self.sz[None,None,None,:,None]-1)-1
        A_t = F.grid_sample(self.A.permute([3,2,1,0])[None,:,:,:,:][[0]*len(times),:,:,:,:]
                            ,grid.permute([4,2,1,0,3]),align_corners=True).permute([0,1,4,3,2])
        A_tC = torch.einsum('tkmnz,kt->tmnz',A_t,C[:,times].to(device))
        # A_tC = torch.einsum('kb,bkmn->bmn',self.C[:,times],A_t)
        reg = torch.tensor([ExponentialFP.log_det_jac(self.beta[:,:,t],self.sz-1)**2+
                            ExponentialFP.log_det_jac(self.beta[:,:,t],self.sz*0)**2 for t in times])
        return A_tC,A_t,grid,reg
        
    @staticmethod
    def regularizer(A, B, epsilon=.1):
        return epsilon*(ExponentialFP.det_jac(B,A.min(0)[0])-1)**2 \
             + epsilon*(ExponentialFP.det_jac(B,A.max(0)[0])-1)**2
    
    @staticmethod
    def spatial_pushforward(dl,batch_size,sz,device,model):
        K = model.C.shape[0]
        A_t = np.zeros((sz[0],sz[1],sz[2],K,len(dl)*batch_size))
        grid = np.array(np.where(np.ones((sz[0],sz[1],sz[2])))).T.reshape(sz[0],sz[1],sz[2],3)
        Y_i = np.zeros((sz[0],sz[1],sz[2],len(dl)*batch_size))
        
        Y = np.zeros((sz[0],sz[1],sz[2],len(dl)*batch_size))
        
        for batch_idx, data in enumerate(dl):
            _,test_,flow_,_= model.fp(data[1].tolist(),model.C)
            
            flow_ = flow_[:,:,:,[0,1,2]]
            for d in range(flow_.shape[3]):
                flow_[:,:,:,d] = ((flow_[:,:,:,d]+1)/2)*sz[d]

            test_ = test_.detach().cpu().numpy()
            A_t[:,:,:,:,batch_idx*batch_size:batch_idx*batch_size+test_.shape[0]] = np.transpose(test_,[2,3,4,1,0])
            Y[:,:,:,batch_idx*batch_size:batch_idx*batch_size+data[0].shape[0]] = np.transpose(data[0],[1,2,3,0])
            
            flow_ = flow_.detach().cpu().numpy()
            for b in range(data[0].shape[0]):
                Y_i[:,:,:,batch_idx*batch_size+b] = ExponentialFP.image_iwarp(data[0][b,:,:,:],flow_[:,:,:,:,b],grid)
                
        return A_t, Y_i, Y
    
    @staticmethod
    def image_iwarp(im,flow,grid):
        X = np.array([flow[:,:,:,0].reshape(-1), 
                      flow[:,:,:,1].reshape(-1), 
                      flow[:,:,:,2].reshape(-1)]).T
        Y = im.reshape(-1)
        interp = sp.interpolate.NearestNDInterpolator(X,Y)
        mapped = interp(grid).reshape(im.shape)
        return mapped



    @staticmethod
    def log_det_jac(B,P):
        x,y,z = P[0],P[1],P[2]
        
        a = B[1,0]+2*B[4,0]*x+B[7,0]*y+B[9,0]*z
        b = B[2,0]+2*B[5,0]*y+B[7,0]*x+B[8,0]*z
        c = B[3,0]+2*B[6,0]*z+B[8,0]*y+B[9,0]*x
        d = B[1,1]+2*B[4,1]*x+B[7,1]*y+B[9,1]*z
        e = B[2,1]+2*B[5,1]*y+B[7,1]*x+B[8,1]*z
        f = B[3,1]+2*B[6,1]*z+B[8,1]*y+B[9,1]*x
        g = B[1,2]+2*B[4,2]*x+B[7,2]*y+B[9,2]*z
        h = B[2,2]+2*B[5,2]*y+B[7,2]*x+B[8,2]*z
        i = B[3,2]+2*B[6,2]*z+B[8,2]*y+B[9,2]*x
        
        det = torch.log(abs(a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)))
        return det
    
class DeformableNMF:
    
    def __init__(self,sz,K,T,positions=None):
        self.SpatialModel = ExponentialFP
        self.fp = self.SpatialModel(sz=sz,K=K,T=T,positions=positions)
        
        self.C = torch.rand((K,T)).to(device)
        self.A = torch.rand((K,sz[0],sz[1])).to(device)
        
        if positions is not None:
            grid = np.array(np.where(np.ones((sz[0],sz[1],sz[2])))).T
            self.D = 1-np.exp(-.01*sp.spatial.distance.cdist(grid,positions).reshape(sz[0],sz[1],sz[2],K))
        else:
            self.D = None
        
    @staticmethod
    def update_temporal(A_t,C,Y,gamma=None):
        A_ts = np.einsum('mnzkt,mnzlt->klt', A_t, A_t)
        C1 = np.einsum('mnzkt,mnzt->kt', A_t, Y)
        C2 = np.einsum('klt,lt->kt', A_ts, C)
        if gamma is not None:
            reg = np.hstack((C[:,0][:,None],C[:,:-1])) + np.hstack((C[:,1:],C[:,-1][:,None]))
            C1 += gamma*reg
            C2 += 2*gamma*C
        C = C*C1/(C2+1e-32)
        return C
    
    @staticmethod
    def update_spatial(A,C,Y_i,D=None,gamma=None):
        C_s = np.einsum('kt,pt->kp',C,C)
        A1 = np.einsum('mnt,kt->mnk',Y_i,C)
        A2 = np.einsum('mnk,kp->mnp',A,C_s)
        
        if D is not None:
            A2 += gamma*D
        A = A*A1/(A2+1e-32)
        return A
    
    
    def update_footprints(self,testloader,batch_size,sz,gamma_c=1e-2,gamma_a=1e0,iter_c=10):
        with torch.no_grad():
            A_t,Y_i,Y = self.SpatialModel.spatial_pushforward(testloader,batch_size,sz,device,self)
        
        with torch.no_grad():
            C = self.C.detach().cpu().numpy()
            # A = self.A.permute([1,2,3,0]).detach().cpu().numpy()
            # Y = Y.squeeze()
        
        for i in range(iter_c):
            C = DeformableNMF.update_temporal(A_t,C,Y,gamma=gamma_c)
            # A = DeformableNMF.update_spatial(A,C,Y_i,D=self.D,gamma=gamma_a)
        
        # self.A = torch.tensor(A.transpose([2,0,1])).float().to(device)
        self.C = torch.tensor(C).float().to(device)
        
        return A_t,Y_i,Y
        
    def update_motion(self,dataloader,optimizer,gamma=0,epochs=20):
        for epoch in range(1,epochs+1):
            print('Epoch ' + str(epoch))
            self.fp.train()
            for batch_idx, data in enumerate(dataloader):
                optimizer.zero_grad()
                A_tC,A_C,flow,reg= self.fp(data[1].tolist(),self.C)
                recon = F.mse_loss(A_tC,data[0].to(device))
                loss = recon+gamma*reg.mean()
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('Recon: ' + str(recon))
                    print('Reg: ' + str(reg))

class SimulatedVideoDataset(Dataset):
    def __init__(self,K=20,T=30,sz=[100,100,5],shape_std=9,traj_means=[0,0,0],
                 traj_snr=[-2,-2,-2],density=.9,bg_snr=-2):
        """
        """
        video,positions,traces = Simulator.generate_quadratic_video(K=K,T=T,sz=sz,shape_std=shape_std,
                    traj_means=traj_means,traj_snr=traj_snr,density=density,bg_snr=bg_snr)

        self.video = video.float()
        self.positions = positions
        self.traces = traces
        
    def __len__(self):
        return self.video.shape[3]

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.video[:,:,:,idx]
        sample[sample<0] = 0

        return sample,idx
    
    
class NeuroPALVideoDataset(Dataset):
    def __init__(self,file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        vid_mat = loadmat(file+'\\data.mat')
        self.video = np.array(vid_mat['data'][::2,::2,::10,:100]).astype(np.float32)
        
        pos_mat = loadmat(file+'\\traces_n.mat')
        self.positions = torch.tensor(pos_mat['positions']).float()-1
        self.positions[:,0,:] /= 2
        self.positions[:,1,:] /= 2
        self.positions[:,2,:] /= 10
        
        self.names = pos_mat['neuron_names'][0]
        
    def __len__(self):
        return self.video.shape[3]

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.video[:,:,:,idx]
        sample[sample<0] = 0

        return sample,idx

