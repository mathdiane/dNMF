# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:10:26 2020

@author: Amin
"""
from Demix.MotionCorrect import MotionCorrect
from Utils import Simulator
from Demix.dNMF import dNMF
import numpy as np
import torch
import time

# %% Choosing the parameters
params = {'n_trials':5, 'noise_level':1e-4, 'sigma_inv':.2, 
          'radius':100, 'step_S':.1, 'gamma':0, 'stride_factor':2, 'density':.1, 'varfact':5,
          'traj_means':[.0,.0,.0], 'traj_variances':[2e-4,2e-4,1e-5], 'sz':[20,20,1], 
          'K':20, 'T':20, 'roi_window':[4,4,0]}

# %% Simulating video
video,positions,traces = Simulator.generate_quadratic_video(K=params['K'],T=params['T'],sz=params['sz'],\
                                                varfact=params['varfact'],traj_variances=params['traj_variances'],\
                                                traj_means=params['traj_means'],density=params['density'],bg_noise=params['noise_level'])
neuron_names = [str(i) for i in range(positions.shape[0])]

# %% Running ROI using ground truth tracking
start = time.time()
roi_signals = Simulator.get_roi_signals(video,positions,window=np.array(params['roi_window']))
end = time.time()
print('ROI finished in ' + str(end-start) + ' seconds')


# %% Running normcorre motion correction
start = time.time()
shape = video.shape[:3]

strides = (shape[0]//params['stride_factor']+1,shape[1]//params['stride_factor']+1,shape[2]//params['stride_factor']+1)
overlaps = (shape[0]//(2*params['stride_factor'])+1,shape[1]//(2*params['stride_factor'])+1,shape[2]//(2*params['stride_factor'])+1)
max_shifts = (shape[0]//params['stride_factor']+1,shape[1]//params['stride_factor']+1,shape[2]//params['stride_factor']+1)



data = video.numpy().transpose([3,0,1,2])
m = MotionCorrect(data,strides=strides,
                  overlaps=overlaps,max_shifts=max_shifts,
                  pw_rigid=True,is3D=True,border_nan='copy')

m.motion_correct()
A = m.apply_shifts_frame(data,positions[:,:,0].numpy(),1)
A = torch.tensor(A).float()

# %% Running NMF on motion corrected data
n_nmf = dNMF(torch.tensor(m.mc[0]).float(),positions=A[:,:,np.newaxis],\
            radius=params['radius'],step_S=params['step_S'],gamma=params['gamma'],
            use_gpu=False,initial_p=A,sigma_inv=params['sigma_inv'],
            method='1->t', verbose=True)
n_nmf.optimize(lr=.1,n_iter=0,n_iter_c=20)
end = time.time()
print('normcorre-nmf finished in ' + str(end-start) + ' seconds')

# %% Running dNMF
start = time.time()
dnmf = dNMF(video,positions=positions[:,:,0][:,:,np.newaxis],\
    radius=params['radius'],step_S=params['step_S'],gamma=params['gamma'],
    use_gpu=False,initial_p=positions[:,:,0],sigma_inv=params['sigma_inv'],
    method='1->t', verbose=True)

dnmf.optimize(lr=1e-4,n_iter=20,n_iter_c=2)
end = time.time()
print('dNMF finished in ' + str(end-start) + ' seconds')

# %% Visualizing dNMF results
dnmf.visualize_tracks('result',video)
dnmf.visualize_stats('result')
dnmf.visualize_neurons('result', ['0','1','2'], neuron_names, video)