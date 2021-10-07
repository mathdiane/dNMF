# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:11:25 2020

@author: Amin
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import multivariate_normal
from torch.distributions import normal
from scipy.sparse import rand
from . import Utils
import numpy as np
import torch
import math

# %%

def generate_video(K,T,sz=[20,20,1],shape_std=3,density=.1,bg_snr=-1,traces='exp',
                    motion='sq',motion_par={'means':[.0,.0,.0],'snr':[-3,-3,-3]}):
    
    """Simulate a video of active and moving neurons using Gaussian shapes
        and various motion trajectory models
        
        Args:
            K (integer): Number of neurons
            T (integer): Number of time points
            sz (list): Size of the simulated video
            shape_std (float): Standard deviation of the neuron's spherical 
                covariance which determines how big the neurons are
            motion (string): Motion model, choose between 'sq'=sequential 
                quadratic, 'q'=independent quadratic, and 'gp'=gaussian process
            motion_par (dict): Parameters of motion (refer to each motion
                 generators to find corresponding parameters)
            density (float): Density of the spikes in the generated neural 
                activity traces
            bg_snr (float): Signal to noise ratio (in dB) of the video 
                background noise
            traces (string):'exp' corresponds to exponentially decaying signals
                
        Returns:
            video (numpy.ndarray): The resulting generated video wish shape
                [sz[0] sz[1] sz[2] T]
            positions (numpy.ndarray): Ground truth positions of the simulated
                neurons in time, shape is [K 3 T]
            traces (numpy.ndarray): Ground truth simulated traces, shape is
                [K T]
    """
    # Generate motion trajectory by sequentially transforming point clouds from
    # one time point to the next
    if motion == 'qs': # Quadratic Sequential
        positions = simulate_quadratic_sequential_trajectory(K,T,motion_par['means'],motion_par['snr'],sz)
    if motion == 'q': # Quadratic
        positions = simulate_quadratic_trajectory(K,T,motion_par['means'],motion_par['snr'],sz)
    if motion == 'gp': # Gaussian Process
        positions = generate_gp_motion(K,T,motion_par['sigma'],motion_par['ls'],sz)
        
    # Generate traces
    if traces == 'exp':
        traces = simulate_exponential_traces(K,T,density)
    
    sz = np.array([sz[0],sz[1],sz[2],T])
    
    # Generate video based on the traces, positions, and other features
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
    """Generate Nx10 quadratic basis from a Nx3 position array
    
    Args:
        I (numpy.ndarray): Nx3 positions array to be transferred to quadratic
            representation
    
    Returns:
        I_p (numpy.ndarray): Nx10 quadratic basis representation
            (1,x,y,z,x*2,x*y,x*z,y*z)
    """
    
    I_p = torch.cat((torch.ones((len(I),1)),I,I*I,(I[:,0]*I[:,1]).unsqueeze(1), \
                     (I[:,0]*I[:,2]).unsqueeze(1),(I[:,1]*I[:,2]).unsqueeze(1)),1)
    return I_p

def simulate_quadratic_sequential_trajectory(K,T,means=[.0,.0,.0],snr=[-2,-2,-2],sz=[20,20,1]):
    """Generate point cloud motion trajectory sequentially using quadratic 
        transformation of each frame to the next
        
    Args:
        K (integer): Number of neurons
        T (integer): Number of time points
        sz (list): Size of the simulated video
        means (list): Mean of the motion trajectory in x,y,z 
            dimensions (measured in pixels), if nonzero then there will 
            be a constant shift in neurons motion trajectory along 
            specified dimensions (used as the constant term in the 
            quadratic transformation used for generating motion)
        snr (list): The signal to noise ratio of the motion trajectory
            noise in dB (added as white noise to the quadratic 
             transformation used for generating motion)
    
    Returns:
        positions (numpy.ndarray): Positions of the simulated neurons in time, 
            shape is [K 3 T]
    """
    
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

def simulate_quadratic_trajectory(K,T,snr=[-2,-2,-2],sz=[20,20,1]):
    """Generate point cloud motion trajectory using quadratic transformation 
        of the first frame to the t-th frame
        
    Args:
        K (integer): Number of neurons
        T (integer): Number of time points
        sz (list): Size of the simulated video
        snr (list): The signal to noise ratio of the motion trajectory
            noise in dB (added as white noise to the quadratic 
             transformation used for generating motion)
    """
    
    B0 = torch.tensor([[0,1,0,0,0,0,0,0,0,0], \
                       [0,0,1,0,0,0,0,0,0,0], \
                       [0,0,0,1,0,0,0,0,0,0]]).t().float()
        
    std = [np.sqrt(10**(snr[i]/10))*(sz[i]) for i in range(len(snr))]
    a = torch.cumsum(normal.Normal(0,1).sample((T,3,10)),0)
    b = torch.tensor([std]).t()*a.permute([1,0,2]).permute([2,0,1])
    
    betas = B0[:,:,np.newaxis] + b
    sz = torch.tensor(sz).float()
    I = (sz-1)*torch.rand(K,3)
    I[:,[0,1]] = I[:,[0,1]] + 4
    I_p = quadratic_basis(I)
    
    positions = torch.zeros(K,3,T)
    
    for t in range(T):    
        positions[:,:,t] = I_p@betas[:,:,t]
        
    return positions

        
def simulate_exponential_traces(K,T,density=.1,b=1):
    """Generate traces for neural activities based on exponentially decaying 
        signals
        
        Args:
            K (integer): Number of neurons
            T (integer): Number of time points
            density (float): Density of the spikes in the generated neural 
                activity traces
        Returns:
            traces (numpy.ndarray): Simulated traces, shape is [K T]
    """
    
    traces = b+0*np.random.rand(K,T)
    
    kernel = np.exp(np.arange(0,-3,-.3))
    for k in range(K):
        a = rand(1, T+len(kernel)-1, density=density, format='csr')
        a.data[:] = 1
        traces[k,:] += np.convolve(np.array(a.todense()).flatten(), kernel, 'valid')
        
    return traces

def simulate_cell(sz, mean, cov, color, noise_mean, noise_std, trunc):
    """Simulate the image of one cell in 3 dimensions
    """
    
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
    """Extract cell signals by averaging pixels in a cube
    """
    
    signals = np.zeros((P.shape[0],P.shape[2]))
    for t in range(P.shape[2]):
        for k in range(P.shape[0]):
            pos = P[k,:,t].round().numpy().astype(int)
            signals[k,t] = np.nanmean(Utils.subcube(video[:,:,:,t].unsqueeze(3),pos,window))
            
    return signals



def generate_random_video(cellnum=10, rndpos=1, rndrot=1, trunc = 60, sz=np.array([64,64,1,3,32]),                          cellsz = np.array([15,15,1,3]), cov=np.array([[7,0,0],[0,2,0],[0,0,0.000001]]), noisestd=1):
    """Generate video of randomly moving cells based on affine transformation
    """
    
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
    """Computing the SNR of cell activities based on noise covariance (used
        for the previous version of the bioRxiv paper)
    """
    
    maxC = np.array([simulate_exponential_traces(1,T,density).max() for i in range(10)]).mean()
    center = (np.sqrt(np.linalg.eigvals(cov))*3).astype(int)
    sz = (center*2).tolist()
    sz.append(1)
    maxA = simulate_cell(sz, center.tolist(), cov, [1], [0], [0], 0).max()
    SNR = 2*(np.log10(maxC)+np.log10(maxA)-np.log10(bg_std))
    return SNR

def compute_snr_motion(stds=[1e-3,1e-3,1e-5]):
    """Computing the SNR of cell activities based on noise stds (used
        for the previous version of the bioRxiv paper)
    """
    
    B0 = np.array([[0,1,0,0,0,0,0,0,0,0],
                   [0,0,1,0,0,0,0,0,0,0],
                   [0,0,0,1,0,0,0,0,0,0]])
    SNR = np.log((B0**2).sum()) - np.log(stds[0]**2*B0.size/3+stds[1]**2*B0.size/3+stds[2]**2*B0.size/3)
    return SNR

def compute_snr_positions(positions):
    """Computing the SNR of neural positions (used for the previous 
        version of the bioRxiv paper)
    """
    
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

def generate_gp_motion(K,T=100,sigma=[10,10,10],ls=[10,10,10],sz=[10,10,1]):
    """Generate motion trajectories based on Gaussian Process and RBF kernels
    
    Args:
        K (integer): Number of neurons
        T (integer): Number of time points
        sz (list): Size of the simulated video
        sigma (list): Controls the amount of motion in each direction
        ls (list): Length scale, used to determine the coherency of motion in
            different directions 
    
    Returns:
        positions (numpy.ndarray): Positions of the simulated neurons in time, 
            shape is [K 3 T]
    """
    
    A = np.random.rand(K,3)*np.array(sz)
    
    kernels = [sigma[0]*RBF(ls[0]),
               sigma[1]*RBF(ls[1]),
               sigma[2]*RBF(ls[2])]
    
    
    gps = [[]]*3
    for d in range(3):
        gps[d] = GaussianProcessRegressor(kernel=kernels[d], n_restarts_optimizer=9)
        gps[d].predict(A[:,d][:,None])
    
    S = np.array([A[:,d][:,None]+gps[d].sample_y(A[:,d][:,None],n_samples=T).squeeze().copy() for d in range(3)]).T 
    return torch.tensor(S.transpose(1,2,0)).float()
    