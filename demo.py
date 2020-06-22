# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:10:26 2020

@author: Amin
"""
import matplotlib
matplotlib.rcParams['font.family'] = "Times New Roman"

from Methods.ID import AutoDetect, AutoID
from Methods.ID.Atlas import Atlas
from DataHandling import DataHandler
from Methods.Demix import Demix
from Methods import Preprocess
import numpy as np
import glob

# %% Loading the image of a worm and extracting cell centers

file = 'C:\\Users\\Amin\\Desktop\\Projects\\WormAutoID\\data\\Hobert\\Best otIs669 YA\\Best D-V\\Training Set 1\\38_YApSP.mat'
data, scale, bodypart = DataHandler.load_neuropal(file)

data_zscored = Preprocess.zscore(data)
if bodypart == 'head':
    k = 200
    neurons = ['IL2VR', 'BAGR', 'M5', 'M1', 'RMED', 'I6', 'VB1', 'IL2R', 'AVG', 'VB2', 'AWCL', 'ADAL', 'ADEL']
elif bodypart == 'tail':
    k = 60
    neurons = ['DVA', 'PLMR', 'VA12','PHSO1L', 'DA8', 'PQR', 'PVNR']

    
data_smooth, p = AutoDetect.local_max(data_zscored, .1, k, scale=scale)
AutoDetect.visualize(data_smooth,p,scale=1/scale)


im = DataHandler.load_neuropal_id(file[:-4]+'_ID_STRUCT.mat')
im.update_colors_readout(data_smooth)
p_gt = np.concatenate((im.get_positions(im.scale),im.get_colors_readout()),1)
AutoDetect.visualize(data_smooth,p_gt,1/im.scale)



# %% User guided labeling and segmentation

file = 'C:\\Users\\Amin\\Desktop\\Projects\\WormAutoID\\codes\\Data\\Models\\atlas.mat'
atlas = DataHandler.load_atlas(file,bodypart)


image_ind = np.array([im.get_annotations().index(x) for x in neurons if x in im.get_annotations()])

atlas_aligned = AutoID.supervised_align(atlas,  neurons=neurons, im=im)
atlas_aligned['sigma'] = atlas_aligned['sigma']
AutoDetect.visualize(data_smooth,atlas_aligned['mu'],1/im.scale)

bgmm = AutoID.bgmm_auto_id(data_smooth, atlas_aligned, n_iter=10, do_sinkhorn=True, do_update_sigma=True, update_prior=True, \
                           annot=neurons, annot_mu=p_gt[image_ind,:6], scale=im.scale)
bgmm.visualize_segmentation(.99/im.scale)
bgmm.compute_accuracy(im.get_annotations(), im.get_positions(im.scale), 1, radius=3)


p_mtx,_ = AutoID.compute_ot_ent(atlas_aligned['mu'], bgmm.mu, np.ones((bgmm.K)), np.ones((bgmm.K)), .1)
bgmm.compute_ranking_accuracy(im.get_annotations(), im.get_positions(im.scale), p_mtx, radius=3, top_k=3)

# %% Automated labeling and segmentation
file = 'C:\\Users\\Amin\\Desktop\\Projects\\WormAutoID\\codes\\Data\\Models\\atlas.mat'
atlas = DataHandler.load_atlas(file,bodypart)

#p_s = p[:,0:6]
#p_s[:,:3] = p[:,:3]*scale.T

#transformed,beta0,beta

atlas_aligned,_ = AutoID.gw_align(atlas,p_gt[:,0:6],n_iter=100,degree=1)
#AutoID.visualize((atlas['mu'],p[:,0:6],atlas_aligned['mu']))
AutoDetect.visualize(data_smooth,atlas_aligned['mu'],scale=1/im.scale)


bgmm = AutoID.bgmm_auto_id(data_smooth, atlas_aligned, update_prior=True, scale=im.scale)
bgmm.visualize_segmentation(.99/im.scale)
bgmm.compute_accuracy(im.get_annotations(), im.get_positions(im.scale), 1, radius=3)

p_mtx,_ = AutoID.compute_gw_ent(atlas_aligned['mu'], bgmm.mu, atlas_aligned['mu'][:,3:].sum(1), bgmm.mu[:,3:].sum(1), 1)
bgmm.compute_ranking_accuracy(im.get_annotations(), im.get_positions(im.scale), p_mtx, top_k=5)

# %% Loading video and demixing signals

file = 'C:\\Users\\Amin\\Desktop\\Projects\\WormTraces\\data\\1024_tail_06'
dnmf, video, neuron_names, scale = Demix.demix_video(file)
dnmf.save_results(file,[],[],[],[])
dnmf.visualize_tracks(file, video)
dnmf.visualize_neurons(file, ['ALN','PHSO1','PHC','PVN'], neuron_names, video)

dnmf, video = Demix.demix_simulated_video()
dnmf.visualize_neurons('C:\\Users\\Amin\\Desktop\\simulation', ['1','2','3','4'], [str(i) for i in range(dnmf.A.shape[0])], video)
dnmf.visualize_tracks('C:\\Users\\Amin\\Desktop\\simulation', video)



# %% Loading annotated images and training an atlas

main_folder = 'C:\\Users\\Amin\\Desktop\\Projects\\WormAutoID\\data\\Hobert\\Best otIs669 YA\\';

subfolders = ['Best D-V\\Training Set 1\\',
              'Best D-V\\Training Set 2\\',
              'Best L-R\\Training Set 1\\',
              'Best L-R\\Training Set 2\\']

files = [glob.glob(main_folder+subfolder+'*_STRUCT.mat') for subfolder in subfolders]
files = [item for sublist in files for item in sublist]
ims = [DataHandler.load_neuropal_id(file) for file in files]
ims_head = [im for im in ims if im.bodypart == 'head']
ims_tail = [im for im in ims if im.bodypart == 'tail']

atlas_head = Atlas.train_atlas(ims_head)
Atlas.visualize(atlas_head,ims_head,'')

atlas_tail = Atlas.train_atlas(ims_tail)
Atlas.visualize(atlas_tail,ims_tail,'')