# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:10:26 2020

@author: Amin
"""
import matplotlib
matplotlib.rcParams['font.family'] = "Times New Roman"

from Methods.ID.Atlas import Atlas
from Demix import Demix
import numpy as np
import glob


# %% Simulating video and demixing signals

dnmf, video = Demix.demix_simulated_video()
dnmf.visualize_neurons('', ['1','2','3','4'], [str(i) for i in range(dnmf.A.shape[0])], video)
dnmf.visualize_tracks('', video)
