# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:11:25 2020

@author: Amin
"""


from scipy.ndimage import affine_transform
from . import Transformations
import numpy as np
import torch
 # Helper Function for Grabbing a Patch from a Multicolor Volume
# To test, uncomment the plotting part.


def subcube(cube, loc_r, center):
    sz = np.array(cube.shape[0:3])
    
    rel = np.floor(center);
    reu = np.floor(center);
    
    
    sz = sz.astype(int)
    loc = loc_r.astype(int)
    
    rel = rel.astype(int)
    reu = reu.astype(int)
    center = center.astype(int)
    
    rel[loc-center<0] = loc[loc-center<0];
    reu[loc+center-sz+1>0] = sz[loc+center-sz+1>0]-loc[loc+center-sz+1>0]-1;
    
    
    patch = cube[loc[0]-rel[0]: loc[0]+reu[0]+1,loc[1]-rel[1]: loc[1]+reu[1]+1,loc[2]-rel[2]: loc[2]+reu[2]+1,:]
    newcenter = patch.shape[0:3];
    
    if torch.is_tensor(patch):
        patch = patch.clone()
        for d in range(patch.shape[3]):
            patch[:,:,:,d] = torch.tensor(affine_transform(patch[:,:,:,d],np.eye(3),offset=loc_r-loc))
    else:
        patch = patch.copy()
        for d in range(patch.shape[3]):
            patch[:,:,:,d] = affine_transform(patch[:,:,:,d],np.eye(3),offset=loc_r-loc)
        
    if any(newcenter[0:3] != 2*np.floor(center)+1):
        pre = (np.floor(center)-rel).astype(int)
        post = (np.floor(center)-reu).astype(int)
        if torch.is_tensor(patch):
            patch = torch.tensor(np.pad(patch, ((pre[0], post[0]), (pre[1], post[1]), (pre[2], post[2]),(0,0)), 'constant'))
        else:
            patch = np.pad(patch, ((pre[0], post[0]), (pre[1], post[1]), (pre[2], post[2]),(0,0)), 'constant')
            
    return patch



# # Helper Function for Placing a Subcube in a Cube
# This function is used for placing the cells in the location determined by the brownian motion.
# To test, uncomment the plotting part.


def placement(sz, loc, F1):
    loc = np.floor(loc);
    
    center = np.floor(np.array(F1.shape[0:3])/2)
    
    rel = np.floor(center);
    reu = np.floor(center);
    
    rel[loc-center<0] = loc[loc-center<0];
    reu[loc+center-sz+1>0] = sz[loc+center-sz+1>0]-loc[loc+center-sz+1>0]-1;
    
    F = np.zeros(sz.tolist()+[F1.shape[3]]);
    
    loc = loc.astype(int)
    rel = rel.astype(int)
    reu = reu.astype(int)
    center = center.astype(int)
    
    
    F[loc[0]-rel[0]: loc[0]+reu[0]+1, loc[1]-rel[1]: loc[1]+reu[1]+1, loc[2]-rel[2]: loc[2]+reu[2]+1,:] =  F1[center[0]-rel[0]: center[0]+reu[0]+1,center[1]-rel[1]: center[1]+reu[1]+1,center[2]-rel[2]: center[2]+reu[2]+1,:];
    
    return F


def superpose(vol, loc, F1):
    loc = np.floor(loc).astype(int)
    sz = np.array(vol.shape)[0:3]
    
    
    
    center = np.floor(np.array(F1.shape[0:3])/2).astype(int)
    
    rel = np.floor(center).astype(int)
    reu = np.floor(center).astype(int)
    rel[loc-center<0] = loc[loc-center<0]
    reu[loc+center-sz+1>0] = sz[loc+center-sz+1>0]-loc[loc+center-sz+1>0]-1
    
    F = np.zeros(sz.tolist()+[F1.shape[3]]);
    
    loc = loc.astype(int)
    rel = rel.astype(int)
    reu = reu.astype(int)
    center = center.astype(int)
    
    
    F[loc[0]-rel[0]: loc[0]+reu[0]+1, loc[1]-rel[1]: loc[1]+reu[1]+1, loc[2]-rel[2]: loc[2]+reu[2]+1,:] = \
        F[loc[0]-rel[0]: loc[0]+reu[0]+1, loc[1]-rel[1]: loc[1]+reu[1]+1, loc[2]-rel[2]: loc[2]+reu[2]+1,:] + \
        F1[center[0]-rel[0]: center[0]+reu[0]+1,center[1]-rel[1]: center[1]+reu[1]+1,center[2]-rel[2]: center[2]+reu[2]+1,:]
    
    return F

def max_project(video, color_by_depth=False, cut_points=None):
    if len(video.shape) == 5:
        mp = video.max(4)
    else:
        mp = video
        
    mp = (mp-mp.min())/(mp.max()-mp.min())
    
    if color_by_depth:
        mp = np.stack((mp[:,:,0:cut_points[0],:].max(2).squeeze(), \
                            mp[:,:,cut_points[0]+1:cut_points[1],:].max(2).squeeze(), \
                            mp[:,:,cut_points[1]+1:,:].max(2).squeeze()), axis=-1)
    else:
        mp = mp.max(2).squeeze()
        
    
    return mp




def obs2image(X1,I1,scale=1):
    
    if torch.is_tensor(X1):
        X1 = X1.data.numpy()
        I1 = I1.data.numpy()
        
    X1 = X1*scale
    
    X1 = X1.astype(int)
    
    X1[X1 < 0] = 0
    max_coor = X1.max(0)
    
    shape = np.append(max_coor+1,I1.shape[1])
    
    I = I1.reshape(1,-1).squeeze()
    X = np.repeat(X1,I1.shape[1],0).squeeze()
    C = np.tile(np.arange(I1.shape[1]), X1.shape[0]).astype(int).squeeze()
    
    ind = np.ravel_multi_index((X[:,0], X[:,1], X[:,2], C), shape)
    
    recon = np.zeros((np.prod(shape))); 
    recon[ind] = I/I.max(); recon = recon.reshape(shape)
    
    return recon
    
    
def pairwise_distances(x,y):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist