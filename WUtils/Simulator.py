# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:11:25 2020

@author: Amin
"""

from scipy.stats import multivariate_normal
from torch.distributions import normal
from scipy.sparse import rand
from . import Utils
import numpy as np
import torch
import math

# %%

def generate_quadratic_video(K,T,sz=[20,20,1],shape_std=3,traj_means=[.0,.0,.0],
                             traj_snr=[-3,-3,-3],density=.1,bg_snr=-1,traces='exp'):
    """Simulate a video of active and moving neurons using Gaussian shapes
        and quadratic transofrmation for motion trajectories
    """
    
    positions = simulate_quadratic_sequential_trajectory(K,T,traj_means,traj_snr,sz)
    
    if traces == 'exp':
        traces = simulate_exponential_traces(K,T,density)
    elif traces == 'mixed':
        traces = simulate_mixed_traces(K,T,density)
    
    sz = np.array([sz[0],sz[1],sz[2],T])
    
    bg_std = np.sqrt(10**(bg_snr/10)) # video is normalized to have power = 1
    video = torch.zeros([sz_ for sz_ in sz])
    noise_video = bg_std*normal.Normal(0,1).sample(sz)
    sz[3] = 1
    for t in range(T):
        for k in range(K):
            patch = simulate_cell(sz,positions[k,:,t].squeeze().numpy(),shape_std*np.eye(3), np.array([traces[k,t]]),np.array([0]),np.array([0]),0)
            video[:,:,:,t] = video[:,:,:,t] + torch.tensor(patch).float()[:,:,:,0]
    video /= (video**2).sum()
    video += noise_video
    
    return video/video.max(),positions,traces


def quadratic_basis(I):
    I_p = torch.cat((torch.ones((len(I),1)),I,I*I,(I[:,0]*I[:,1]).unsqueeze(1), \
                     (I[:,0]*I[:,2]).unsqueeze(1),(I[:,1]*I[:,2]).unsqueeze(1)),1)
    return I_p

def simulate_quadratic_sequential_trajectory(K,T,means=[.0,.0,.0],snr=[-2,-2,-2],sz=[20,20,1]):
    B0 = torch.tensor([[means[0],1,0,0,0,0,0,0,0,0], \
                       [means[1],0,1,0,0,0,0,0,0,0], \
                       [means[2],0,0,1,0,0,0,0,0,0]]).t().float()
        
    std = [np.sqrt(10**(snr[i]/10))*(sz[i]) for i in range(len(snr))]
    a = normal.Normal(0,1).sample((T,3,10))
    b = torch.tensor([std]).t()*a.permute([1,0,2]).permute([2,0,1])
    
    betas = B0[:,:,np.newaxis] + b
    sz = torch.tensor(sz).float()
    I = ((sz-1)/2)*torch.rand(K,3)
    I += (sz-1)/4
    
    positions = torch.zeros(K,3,T)
    positions[:,:,0] = I
    
    for t in range(1,T):
        positions[:,:,t] = quadratic_basis(positions[:,:,t-1])@betas[:,:,t]
        
    return positions

def simulate_quadratic_trajectory(K,T,variances=[.001,.001,.00001],sz=[20,20,1]):
    B0 = torch.tensor([[0,1,0,0,0,0,0,0,0,0], \
                       [0,0,1,0,0,0,0,0,0,0], \
                       [0,0,0,1,0,0,0,0,0,0]]).t().float()
         
    a = torch.cumsum(normal.Normal(0,1).sample((T,3,10)),0)
    b = torch.tensor([variances]).t()*a.permute([1,0,2]).permute([2,0,1])
    
    betas = B0[:,:,np.newaxis] + b
    sz = torch.tensor(sz).float()
    I = (sz-1)*torch.rand(K,3)
    I[:,[0,1]] = I[:,[0,1]] + 4
    I_p = quadratic_basis(I)
    
    positions = torch.zeros(K,3,T)
    
    for t in range(T):    
        positions[:,:,t] = I_p@betas[:,:,t]
        
    return positions

def simulate_mixed_traces(K,T,density=.1,b=1):
    traces = b*np.ones((K,T))
    kernel = np.exp(np.arange(0,-3,-.3))
    for k in range(K):
        
        start = int(k*(T/K))
        if k == K-1:
            end = int((k+1)*(T/K))
        else:
            end = int((k+1)*(T/K))+10
        
        a = rand(1, end-start, density=density, format='csr')
        a.data[:] = 1
        traces[k,start:end] = traces[k,start:end] + np.convolve(np.array(a.todense()).flatten(), kernel, 'same')
        
        
    return traces
        
def simulate_exponential_traces(K,T,density=.1):
    traces = np.random.rand(K,T)
    
    kernel = np.exp(np.arange(0,-3,-.3))
    for k in range(K):
        a = rand(1, T+len(kernel)-1, density=density, format='csr')
        a.data[:] = 1
        traces[k,:] = np.convolve(np.array(a.todense()).flatten(), kernel, 'valid')
        
    return traces

def simulate_cell(sz, mean, cov, color, noise_mean, noise_std, trunc):
    pos = np.array(np.where(np.ones(sz[0:3])))
    var = multivariate_normal(mean=mean, cov=cov)
    p = var.pdf(pos.T)*(2*np.pi)**(1.5)*np.linalg.det(cov)**.5
    if p.size > 1:
        p[p < np.percentile(p, trunc)] = 0
    prob = p.reshape(sz[0:3])
    volume = np.zeros(sz)
    
    for channel in range(sz[3]):
        volume[:, :, :, channel] = color[channel]*prob + noise_mean[channel] + noise_std[channel]*np.random.randn(*sz[0:3])
        
    return volume



def simulate_trajectory(t, obj, mean, cov):
    var = multivariate_normal(mean=np.array([0,0,0]), cov=cov)
    trajectory = np.cumsum(var.rvs(size=(t,obj)), axis=0)
    if obj == 1 & t == 1:
        trajectory = np.reshape(trajectory, (1,1,trajectory.shape[0]))
    elif obj == 1:
        trajectory = np.reshape(trajectory, (trajectory.shape[0],1,trajectory.shape[1]))
    elif t == 1:
        trajectory = np.reshape(trajectory, (1,trajectory.shape[0],trajectory.shape[1]))
    
    for objidx in range(obj):
        trajectory[:,objidx,:] = trajectory[:,objidx,:] + mean[objidx,:]
    return trajectory


def get_roi_signals(video,P,window=np.array([3,3,0])):
    signals = np.zeros((P.shape[0],P.shape[2]))
    for t in range(P.shape[2]):
        for k in range(P.shape[0]):
            pos = P[k,:,t].round().numpy().astype(int)
            signals[k,t] = np.nanmean(Utils.subcube(video[:,:,:,t].unsqueeze(3),pos,window))
            
    return signals



def generate_random_video(cellnum=10, rndpos=1, rndrot=1, trunc = 60, sz=np.array([64,64,1,3,32]),                          cellsz = np.array([15,15,1,3]), cov=np.array([[7,0,0],[0,2,0],[0,0,0.000001]]), noisestd=1):
    
    border = sz[0:3]-cellsz[0:3]
    border[border<0]=0
    centers = np.tile((cellsz[0:3]-1)/2,(cellnum, 1))+ np.random.rand(cellnum,3)*np.tile(border,(cellnum, 1))
    
    if rndpos:
        trajectory = (simulate_trajectory(sz[4], cellnum, centers,np.array([[3.,0.3,0],[0.3,1.4,0], [0,0,0.000001]]))).astype(int)
        
    else:
        trajectory = np.tile(centers.astype(int), (sz[4],1,1))
    
    trajectory[trajectory<0]=0
    colors = np.random.rand(cellnum, sz[3])
    colors = colors/np.sum(colors)
    video = noisestd*np.random.rand(*sz)
    center = (cellsz[0:3]/2).astype(int);
    
    rotsig = 0.01
    from scipy.stats import multivariate_normal
    if rndrot:
        var = multivariate_normal(mean=np.array([0,0,0]), cov=[[.01,0,0],[0,.01,0],[0,0,0.01]])
        rotrnd = np.cumsum(var.rvs(size=(sz[4],cellnum)), axis=0)
    else:
        var = multivariate_normal(mean=np.array([0,0,0]), cov=[[1,0,0],[0,1,0],[0,0,1]])
        rotrnd = np.tile(np.cumsum(var.rvs(size=(1,cellnum)), axis=0), (sz[4],1,1))
        
    if cellnum == 1:
        rotrnd = np.reshape(rotrnd, (rotrnd.shape[0],1,3))
        
    for cellidx in range(cellnum):
        for t in range(sz[4]):
            rotmat = rotation_matrix(rotrnd[t,cellidx,0],[0,0,1])
            rotcov = rotmat[0:3,0:3].transpose()@cov@rotmat[0:3,0:3]
            cell = simulate_cell(cellsz,center,rotcov,colors[cellidx,:].squeeze(),np.array([0,0,0]),np.array([0,0,0]), trunc)
            video[:,:,:,:,t] = video[:,:,:,:,t] + Utils.placement(sz[0:3], trajectory[t,cellidx,:].squeeze(), cell)

    video = video/max(video.flatten())
    
    return video, trajectory, rotrnd, colors, cellnum, cellsz, sz, trunc, cov, rotsig

def compute_snr_intensity(density,cov=2*np.eye(3),T=20,bg_std=.0001):
    maxC = np.array([simulate_exponential_traces(1,T,density).max() for i in range(10)]).mean()
    center = (np.sqrt(np.linalg.eigvals(cov))*3).astype(int)
    sz = (center*2).tolist()
    sz.append(1)
    maxA = simulate_cell(sz, center.tolist(), cov, [1], [0], [0], 0).max()
    SNR = 2*(np.log10(maxC)+np.log10(maxA)-np.log10(bg_std))
    return SNR

def compute_snr_motion(stds=[1e-3,1e-3,1e-5]):
    B0 = np.array([[0,1,0,0,0,0,0,0,0,0],
                   [0,0,1,0,0,0,0,0,0,0],
                   [0,0,0,1,0,0,0,0,0,0]])
    SNR = np.log((B0**2).sum()) - np.log(stds[0]**2*B0.size/3+stds[1]**2*B0.size/3+stds[2]**2*B0.size/3)
    return SNR

def compute_snr_positions(positions):
    return np.log((positions[:,:,0]**2).sum()) - np.log(np.array([((positions[:,:,t] - positions[:,:,0])**2).sum() for t in range(1,positions.shape[2])]).mean())
    
def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.
        Taken from https://github.com/cgohlke/transformations
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.
        Taken from https://github.com/cgohlke/transformations
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data
    return None

