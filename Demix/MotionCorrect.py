#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

The functions apply_shifts_dft, register_translation, _compute_error, _compute_phasediff, and _upsampled_dft are from
SIMA (https://github.com/losonczylab/sima), licensed under the  GNU GENERAL PUBLIC LICENSE, Version 2, 1991.
These same functions were adapted from sckikit-image, licensed as follows:

Copyright (C) 2011, the scikit-image team
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are
 met:

  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in
     the documentation and/or other materials provided with the
     distribution.
  3. Neither the name of skimage nor the names of its contributors may be
     used to endorse or promote products derived from this software without
     specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
 INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.

"""

from skimage.transform import resize as resize_sk
from skimage.transform import warp as warp_sk
from Demix.MotionCorrect import *
from scipy.spatial import distance
from numpy.fft import ifftshift
from past.utils import old_div
from cv2 import idft as ifftn
from cv2 import dft as fftn
from builtins import range
from builtins import zip
from builtins import map
from builtins import str
from typing import List
import numpy as np
import pylab as pl
import itertools
import logging
import cv2

opencv = True

#%%
class MotionCorrect(object):
    """
        class implementing motion correction operations
       """

    def __init__(self, video, min_mov=None, dview=None, max_shifts=(6, 6), niter_rig=1, splits_rig=1, num_splits_to_process_rig=None,
                 strides=(96, 96), overlaps=(32, 32), splits_els=1, num_splits_to_process_els=None,
                 upsample_factor_grid=4, max_deviation_rigid=3, shifts_opencv=True, nonneg_movie=True, gSig_filt=None,
                 use_cuda=False, border_nan=True, pw_rigid=False, num_frames_split=80, var_name_hdf5='mov',is3D=False,
                 indices=(slice(None), slice(None))):
        """
        Constructor class for motion correction operations

        Args:
           min_mov: int16 or float32
               estimated minimum value of the movie to produce an output that is positive

           dview: ipyparallel view object list
               to perform parallel computing, if NOne will operate in single thread

           max_shifts: tuple
               maximum allow rigid shift

           niter_rig':int
               maximum number of iterations rigid motion correction, in general is 1. 0
               will quickly initialize a template with the first frames

           splits_rig': int
            for parallelization split the movies in  num_splits chuncks across time

           num_splits_to_process_rig: list,
               if none all the splits are processed and the movie is saved, otherwise at each iteration
               num_splits_to_process_rig are considered

           strides: tuple
               intervals at which patches are laid out for motion correction

           overlaps: tuple
               overlap between pathes (size of patch strides+overlaps)

           pw_rigig: bool, default: False
               flag for performing motion correction when calling motion_correct

           splits_els':list
               for parallelization split the movies in  num_splits chuncks across time

           num_splits_to_process_els: list,
               if none all the splits are processed and the movie is saved  otherwise at each iteration
                num_splits_to_process_els are considered

           upsample_factor_grid:int,
               upsample factor of shifts per patches to avoid smearing when merging patches

           max_deviation_rigid:int
               maximum deviation allowed for patch with respect to rigid shift

           shifts_opencv: Bool
               apply shifts fast way (but smoothing results)

           nonneg_movie: boolean
               make the SAVED movie and template mostly nonnegative by removing min_mov from movie

           use_cuda : bool, optional
               Use skcuda.fft (if available). Default: False

           border_nan : bool or string, optional
               Specifies how to deal with borders. (True, False, 'copy', 'min')

           num_frames_split: int, default: 80
               Number of frames in each batch. Used when cosntructing the options
               through the params object

           var_name_hdf5: str, default: 'mov'
               If loading from hdf5, name of the variable to load

            is3D: bool, default: False
               Flag for 3D motion correction

            indices: tuple(slice), default: (slice(None), slice(None))
               Use that to apply motion correction only on a part of the FOV

       Returns:
           self

        """

        if type(video) is not list:
            video = [video]

        self.video = video
        self.dview = dview
        self.max_shifts = max_shifts
        self.niter_rig = niter_rig
        self.splits_rig = splits_rig
        self.num_splits_to_process_rig = num_splits_to_process_rig
        self.strides = strides
        self.overlaps = overlaps
        self.splits_els = splits_els
        self.num_splits_to_process_els = num_splits_to_process_els
        self.upsample_factor_grid = upsample_factor_grid
        self.max_deviation_rigid = max_deviation_rigid
        self.shifts_opencv = bool(shifts_opencv)
        self.min_mov = min_mov
        self.nonneg_movie = nonneg_movie
        self.gSig_filt = gSig_filt
        self.use_cuda = bool(use_cuda)
        self.border_nan = border_nan
        self.pw_rigid = bool(pw_rigid)
        self.var_name_hdf5 = var_name_hdf5
        self.is3D = bool(is3D)
        self.indices = indices

    def motion_correct(self, template=None):
        """general function for performing all types of motion correction. The
        function will perform either rigid or piecewise rigid motion correction
        depending on the attribute self.pw_rigid and will perform high pass
        spatial filtering for determining the motion (used in 1p data) if the
        attribute self.gSig_filt is not None. A template can be passed, and the
        output can be saved as a memory mapped file.

        Args:
            template: ndarray, default: None
                template provided by user for motion correction

        Returns:
            self
        """
        # TODO: Review the docs here, and also why we would ever return self
        #       from a method that is not a constructor
        if self.min_mov is None:
            if self.gSig_filt is None:
                self.min_mov = self.video[0].min()

        if self.pw_rigid:
            self.motion_correct_pwrigid(template=template)
            if self.is3D:
                # TODO - error at this point after saving
                b0 = np.ceil(np.max([np.max(np.abs(self.x_shifts_els)),
                                     np.max(np.abs(self.y_shifts_els)),
                                     np.max(np.abs(self.z_shifts_els))]))
            else:
                b0 = np.ceil(np.maximum(np.max(np.abs(self.x_shifts_els)),
                                    np.max(np.abs(self.y_shifts_els))))
        else:
            self.motion_correct_rigid(template=template)
            b0 = np.ceil(np.max(np.abs(self.shifts_rig)))
        self.border_to_0 = b0.astype(np.int)
        return self

    def motion_correct_rigid(self, template=None) -> None:
        """
        Perform rigid motion correction

        Args:
            template: ndarray 2D (or 3D)
                if known, one can pass a template to register the frames to

        Important Fields:
            self.total_template_rig: template updated by iterating  over the chunks

            self.templates_rig: list of templates. one for each chunk

            self.shifts_rig: shifts in x and y (and z if 3D) per frame
        """
        logging.debug('Entering Rigid Motion Correction')
        logging.debug(-self.min_mov)  # XXX why the minus?
        self.total_template_rig = template
        self.templates_rig:List = []
        self.shifts_rig:List = []
        self.mc:List = []

        for video_cur in self.video:
            _total_template_rig, _templates_rig, _shifts_rig, _mc = motion_correct_batch_rigid(
                video_cur,
                self.max_shifts,
                dview=self.dview,
                splits=self.splits_rig,
                num_splits_to_process=self.num_splits_to_process_rig,
                num_iter=self.niter_rig,
                template=self.total_template_rig,
                shifts_opencv=self.shifts_opencv,
                add_to_movie=-self.min_mov,
                nonneg_movie=self.nonneg_movie,
                gSig_filt=self.gSig_filt,
                use_cuda=self.use_cuda,
                border_nan=self.border_nan,
                var_name_hdf5=self.var_name_hdf5,
                is3D=self.is3D,
                indices=self.indices)
            if template is None:
                self.total_template_rig = _total_template_rig

            self.templates_rig += _templates_rig
            self.shifts_rig += _shifts_rig
            self.mc += _mc

    def motion_correct_pwrigid(self, template:np.ndarray=None, show_template:bool=False) -> None:
        """Perform pw-rigid motion correction

        Args:

            template: ndarray 2D (or 3D)
                if known, one can pass a template to register the frames to

            show_template: boolean
                whether to show the updated template at each iteration

        Important Fields:
            self.templates_els: template updated by iterating  over the chunks
            self.x_shifts_els: shifts in x per frame per patch
            self.y_shifts_els: shifts in y per frame per patch
            self.z_shifts_els: shifts in z per frame per patch (if 3D)
            self.coord_shifts_els: coordinates associated to the patch for
            values in x_shifts_els and y_shifts_els (and z_shifts_els if 3D)
            self.total_template_els: list of templates. one for each chunk

        Raises:
            Exception: 'Error: Template contains NaNs, Please review the parameters'
        """
        
        num_iter = 1
        if template is None:
            logging.info('Generating template by rigid motion correction')
            self.motion_correct_rigid()
            self.total_template_els = self.total_template_rig.copy()
        else:
            self.total_template_els = template

        self.templates_els:List = []
        self.x_shifts_els:List = []
        self.y_shifts_els:List = []
        if self.is3D:
            self.z_shifts_els:List = []
            

        self.coord_shifts_els:List = []
        for video_cur in self.video:
            new_template_els, _templates_els,\
                _x_shifts_els, _y_shifts_els, _z_shifts_els, _coord_shifts_els, _ = motion_correct_batch_pwrigid(
                    video_cur, self.max_shifts, self.strides, self.overlaps, -self.min_mov,
                    dview=self.dview, upsample_factor_grid=self.upsample_factor_grid,
                    max_deviation_rigid=self.max_deviation_rigid, splits=self.splits_els,
                    num_splits_to_process=None, num_iter=num_iter, template=self.total_template_els,
                    shifts_opencv=self.shifts_opencv, nonneg_movie=self.nonneg_movie, gSig_filt=self.gSig_filt,
                    use_cuda=self.use_cuda, border_nan=self.border_nan, var_name_hdf5=self.var_name_hdf5, is3D=self.is3D,
                    indices=self.indices)
            if not self.is3D:
                if show_template:
                    pl.imshow(new_template_els)
                    pl.pause(.5)
            if np.isnan(np.sum(new_template_els)):
                raise Exception(
                    'Template contains NaNs, something went wrong. Reconsider the parameters')

            if template is None:
                self.total_template_els = new_template_els
            
            
            self.templates_els += _templates_els
            self.x_shifts_els += _x_shifts_els
            self.y_shifts_els += _y_shifts_els
            if self.is3D:
                self.z_shifts_els += _z_shifts_els
            self.coord_shifts_els += _coord_shifts_els
#            self.mc += _mc
            
    def apply_shifts_frame(self, video, points, t):
        
        xyz_grid = np.array([it[0:3] for it in sliding_window_3d(video[0,:,:,:], self.overlaps, self.strides)])
        dims_grid = tuple(np.max(np.stack(xyz_grid, axis=1), axis=1) - np.min(
                        np.stack(xyz_grid, axis=1), axis=1) + 1)
        shifts_x = np.stack([np.reshape(_sh_, dims_grid, order='C').reshape(-1).astype(
                        np.float32) for _sh_ in self.x_shifts_els], axis=0)
        shifts_y = np.stack([np.reshape(_sh_, dims_grid, order='C').reshape(-1).astype(
                        np.float32) for _sh_ in self.y_shifts_els], axis=0)
        shifts_z = np.stack([np.reshape(_sh_, dims_grid, order='C').reshape(-1).astype(
                        np.float32) for _sh_ in self.z_shifts_els], axis=0)
        xyz_centers = np.array([it[3:6]+np.array(self.strides)/2 for it in sliding_window_3d(video[0,:,:,:], self.overlaps, self.strides)])
        shift_indices = distance.cdist(xyz_centers,points).argmin(0)
        
        A = points.copy()
        A[:,0] = A[:,0] + shifts_x[t,shift_indices]
        A[:,1] = A[:,1] + shifts_y[t,shift_indices]
        A[:,2] = A[:,2] + shifts_z[t,shift_indices]
        
        return A
        
    def apply_shifts_points(self, video, points):
        P_T = np.zeros((points.shape[0], points.shape[1], video.shape[0]))
        xyz_grid = np.array([it[0:3] for it in sliding_window_3d(video[0,:,:,:], self.overlaps, self.strides)])
        dims_grid = tuple(np.max(np.stack(xyz_grid, axis=1), axis=1) - np.min(
                        np.stack(xyz_grid, axis=1), axis=1) + 1)
        shifts_x = np.stack([np.reshape(_sh_, dims_grid, order='C').reshape(-1).astype(
                        np.float32) for _sh_ in self.x_shifts_els], axis=0)
        shifts_y = np.stack([np.reshape(_sh_, dims_grid, order='C').reshape(-1).astype(
                        np.float32) for _sh_ in self.y_shifts_els], axis=0)
        shifts_z = np.stack([np.reshape(_sh_, dims_grid, order='C').reshape(-1).astype(
                        np.float32) for _sh_ in self.z_shifts_els], axis=0)
        xyz_centers = np.array([it[3:6]+np.array(self.strides)/2 for it in sliding_window_3d(video[0,:,:,:], self.overlaps, self.strides)])
        shift_indices = distance.cdist(xyz_centers,points).argmin(0)
            
        for t in range(video.shape[0]):
            P_T[:,:,t] = points.copy()
            P_T[:,0,t] = P_T[:,0,t]-shifts_x[t,shift_indices]+shifts_x[0,shift_indices]
            P_T[:,1,t] = P_T[:,1,t]-shifts_y[t,shift_indices]+shifts_y[0,shift_indices]
            P_T[:,2,t] = P_T[:,2,t]-shifts_z[t,shift_indices]+shifts_z[0,shift_indices]
        
        return P_T

    def get_params(self):
        return {'max_shifts': self.max_shifts, 
                'niter_rig':self.niter_rig, 'splits_rig': self.splits_rig, 
                'strides': self.strides, 'overlaps': self.overlaps, 
                'splits_els':self.splits_els, 'upsample_factor_grid':self.upsample_factor_grid, 
                'max_deviation_rigid':self.max_deviation_rigid, 'min_mov': self.min_mov,
                'border_nan': self.border_nan, 'pw_rigid':self.pw_rigid, 
                'is3D':self.is3D, 'total_template_rig':self.total_template_rig, 
                'templates_rig':self.templates_rig,
                'shifts_rig':self.shifts_rig, 'total_template_els':self.total_template_els, 
                 'templates_els':self.templates_els, 
                'x_shifts_els':self.x_shifts_els, 'y_shifts_els':self.y_shifts_els, 
                'z_shifts_els':self.z_shifts_els}
#%%
def apply_shift_iteration(img, shift, border_nan:bool=False, border_type=cv2.BORDER_REFLECT):
    # todo todocument

    sh_x_n, sh_y_n = shift
    w_i, h_i = img.shape
    M = np.float32([[1, 0, sh_y_n], [0, 1, sh_x_n]])
    min_, max_ = np.nanmin(img), np.nanmax(img)
    img = np.clip(cv2.warpAffine(img, M, (h_i, w_i),
                                 flags=cv2.INTER_CUBIC, borderMode=border_type), min_, max_)
    if border_nan is not False:
        max_w, max_h, min_w, min_h = 0, 0, 0, 0
        max_h, max_w = np.ceil(np.maximum(
            (max_h, max_w), shift)).astype(np.int)
        min_h, min_w = np.floor(np.minimum(
            (min_h, min_w), shift)).astype(np.int)
        if border_nan is True:
            img[:max_h, :] = np.nan
            if min_h < 0:
                img[min_h:, :] = np.nan
            img[:, :max_w] = np.nan
            if min_w < 0:
                img[:, min_w:] = np.nan
        elif border_nan == 'min':
            img[:max_h, :] = min_
            if min_h < 0:
                img[min_h:, :] = min_
            img[:, :max_w] = min_
            if min_w < 0:
                img[:, min_w:] = min_
        elif border_nan == 'copy':
            if max_h > 0:
                img[:max_h] = img[max_h]
            if min_h < 0:
                img[min_h:] = img[min_h-1]
            if max_w > 0:
                img[:, :max_w] = img[:, max_w, np.newaxis]
            if min_w < 0:
                img[:, min_w:] = img[:, min_w-1, np.newaxis]

    return img



#%%

def bin_median(mat, window=10, exclude_nans=True):
    """ compute median of 3D array in along axis o by binning values

    Args:
        mat: ndarray
            input 3D matrix, time along first dimension

        window: int
            number of frames in a bin

    Returns:
        img:
            median image

    Raises:
        Exception 'Path to template does not exist:'+template
    """

    T, d1, d2 = np.shape(mat)
    if T < window:
        window = T
    num_windows = np.int(old_div(T, window))
    num_frames = num_windows * window
    if exclude_nans:
        img = np.nanmedian(np.nanmean(np.reshape(
            mat[:num_frames], (window, num_windows, d1, d2)), axis=0), axis=0)
    else:
        img = np.median(np.mean(np.reshape(
            mat[:num_frames], (window, num_windows, d1, d2)), axis=0), axis=0)

    return img

def bin_median_3d(mat, window=10, exclude_nans=True):
    """ compute median of 4D array in along axis o by binning values

    Args:
        mat: ndarray
            input 4D matrix, (T, h, w, z)

        window: int
            number of frames in a bin

    Returns:
        img:
            median image

    Raises:
        Exception 'Path to template does not exist:'+template
    """

    T, d1, d2, d3 = np.shape(mat)
    if T < window:
        window = T
    num_windows = np.int(old_div(T, window))
    num_frames = num_windows * window
    if exclude_nans:
        img = np.nanmedian(np.nanmean(np.reshape(
            mat[:num_frames], (window, num_windows, d1, d2, d3)), axis=0), axis=0)
    else:
        img = np.median(np.mean(np.reshape(
            mat[:num_frames], (window, num_windows, d1, d2, d3)), axis=0), axis=0)

    return img

#%%

def _upsampled_dft(data, upsampled_region_size,
                   upsample_factor=1, axis_offsets=None):
    """
    adapted from SIMA (https://github.com/losonczylab) and the scikit-image (http://scikit-image.org/) package.

    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

    Upsampled DFT by matrix multiplication.

    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.

    Args:
        data : 2D ndarray
            The input data array (DFT of original data) to upsample.

        upsampled_region_size : integer or tuple of integers, optional
            The size of the region to be sampled.  If one integer is provided, it
            is duplicated up to the dimensionality of ``data``.

        upsample_factor : integer, optional
            The upsampling factor.  Defaults to 1.

        axis_offsets : tuple of integers, optional
            The offsets of the region to be sampled.  Defaults to None (uses
            image center)

    Returns:
        output : 2D ndarray
                The upsampled DFT of the specified region.
    """
    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size, ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError("shape of upsampled region sizes must be equal "
                             "to input data's number of dimensions.")

    if axis_offsets is None:
        axis_offsets = [0, ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError("number of axis offsets must be equal to input "
                             "data's number of dimensions.")

    col_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[1] * upsample_factor)) *
        (ifftshift(np.arange(data.shape[1]))[:, None] -
         np.floor(old_div(data.shape[1], 2))).dot(
             np.arange(upsampled_region_size[1])[None, :] - axis_offsets[1])
    )
    row_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[0] * upsample_factor)) *
        (np.arange(upsampled_region_size[0])[:, None] - axis_offsets[0]).dot(
            ifftshift(np.arange(data.shape[0]))[None, :] -
            np.floor(old_div(data.shape[0], 2)))
    )

    if data.ndim > 2:
        pln_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[2] * upsample_factor)) *
        (np.arange(upsampled_region_size[2])[:, None] - axis_offsets[2]).dot(
                ifftshift(np.arange(data.shape[2]))[None, :] -
                np.floor(old_div(data.shape[2], 2))))

    # output = np.tensordot(np.tensordot(row_kernel,data,axes=[1,0]),col_kernel,axes=[1,0])
    output = np.tensordot(row_kernel, data, axes = [1,0])
    output = np.tensordot(output, col_kernel, axes = [1,0])

    if data.ndim > 2:
        #import pdb
        #pdb.set_trace()
        output = np.tensordot(output, pln_kernel, axes = [1,1])
    #output = row_kernel.dot(data).dot(col_kernel)
    return output


def _compute_phasediff(cross_correlation_max):
    """
    Compute global phase difference between the two images (should be zero if images are non-negative).

    Args:
        cross_correlation_max : complex
            The complex value of the cross correlation at its maximum point.
    """
    return np.arctan2(cross_correlation_max.imag, cross_correlation_max.real)


def _compute_error(cross_correlation_max, src_amp, target_amp):
    """
    Compute RMS error metric between ``src_image`` and ``target_image``.

    Args:
        cross_correlation_max : complex
            The complex value of the cross correlation at its maximum point.

        src_amp : float
            The normalized average image intensity of the source image

        target_amp : float
            The normalized average image intensity of the target image
    """
    error = 1.0 - cross_correlation_max * cross_correlation_max.conj() /\
        (src_amp * target_amp)
    return np.sqrt(np.abs(error))

#%%
    
def register_translation_3d(src_image, target_image, upsample_factor = 1,
                            space = "real", shifts_lb = None, shifts_ub = None,
                            max_shifts = [10,10,1]):

    """
    Simple script for registering translation in 3D using an FFT approach.
    
    Args:
        src_image : ndarray
            Reference image.

        target_image : ndarray
            Image to register.  Must be same dimensionality as ``src_image``.

        upsample_factor : int, optional
            Upsampling factor. Images will be registered to within
            ``1 / upsample_factor`` of a pixel. For example
            ``upsample_factor == 20`` means the images will be registered
            within 1/20th of a pixel.  Default is 1 (no upsampling)

        space : string, one of "real" or "fourier"
            Defines how the algorithm interprets input data.  "real" means data
            will be FFT'd to compute the correlation, while "fourier" data will
            bypass FFT of input data.  Case insensitive.

    Returns:
        shifts : ndarray
            Shift vector (in pixels) required to register ``target_image`` with
            ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)

        error : float
            Translation invariant normalized RMS error between ``src_image`` and
            ``target_image``.

        phasediff : float
            Global phase difference between the two images (should be
            zero if images are non-negative).

    Raises:
     NotImplementedError "Error: register_translation_3d only supports "
                                  "subpixel registration for 3D images"

     ValueError "Error: images must really be same size for "
                         "register_translation_3d"

     ValueError "Error: register_translation_3d only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument."

    """
    
    # images must be the same shape
    if src_image.shape != target_image.shape:
        raise ValueError("Error: images must really be same size for "
                         "register_translation_3d")

    # only 3D data makes sense right now
    if src_image.ndim != 3 and upsample_factor > 1:
        raise NotImplementedError("Error: register_translation_3d only supports "
                                  "subpixel registration for 3D images")

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        src_image_cpx = np.array(
            src_image, dtype=np.complex64, copy=False)
        target_image_cpx = np.array(
            target_image, dtype=np.complex64, copy=False)
        src_freq = np.fft.fftn(src_image_cpx)
        target_freq = np.fft.fftn(target_image_cpx)
    else:
        raise ValueError("Error: register_translation_3d only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument.")

    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    cross_correlation = np.fft.ifftn(image_product)
#    cross_correlation = ifftn(image_product) # TODO CHECK why this line is different
    new_cross_corr = np.abs(cross_correlation)

    CCmax = cross_correlation.max()

    del cross_correlation

    if (shifts_lb is not None) or (shifts_ub is not None):

        if (shifts_lb[0] < 0) and (shifts_ub[0] >= 0):
            new_cross_corr[shifts_ub[0]:shifts_lb[0], :, :] = 0
        else:
            new_cross_corr[:shifts_lb[0], :, :] = 0
            new_cross_corr[shifts_ub[0]:, :, :] = 0

        if (shifts_lb[1] < 0) and (shifts_ub[1] >= 0):
            new_cross_corr[:, shifts_ub[1]:shifts_lb[1], :] = 0
        else:
            new_cross_corr[:, :shifts_lb[1], :] = 0
            new_cross_corr[:, shifts_ub[1]:, :] = 0

        if (shifts_lb[2] < 0) and (shifts_ub[2] >= 0):
            new_cross_corr[:, :, shifts_ub[2]:shifts_lb[2]] = 0
        else:
            new_cross_corr[:, :, :shifts_lb[2]] = 0
            new_cross_corr[:, :, shifts_ub[2]:] = 0
    else:
        new_cross_corr[max_shifts[0]:-max_shifts[0], :, :] = 0
        new_cross_corr[:, max_shifts[1]:-max_shifts[1], :] = 0
        new_cross_corr[:, :, max_shifts[2]:-max_shifts[2]] = 0

    maxima = np.unravel_index(np.argmax(new_cross_corr), new_cross_corr.shape)
    midpoints = np.array([np.fix(axis_size//2) for axis_size in shape])

#    maxima = np.unravel_index(np.argmax(new_cross_corr),cross_correlation.shape)
#    midpoints = np.array([np.fix(old_div(axis_size, 2)) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float32)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]


    if upsample_factor > 1:

        shifts = old_div(np.round(shifts * upsample_factor), upsample_factor)
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(old_div(upsampled_region_size, 2.0))
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freq.size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor

        cross_correlation = _upsampled_dft(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(
            np.argmax(np.abs(cross_correlation)),
            cross_correlation.shape),
            dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + old_div(maxima, upsample_factor)
        CCmax = cross_correlation.max()

    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    return shifts, src_freq, _compute_phasediff(CCmax)

#%%

def register_translation(src_image, target_image, upsample_factor=1,
                         space="real", shifts_lb=None, shifts_ub=None, max_shifts=(10, 10),
                         use_cuda=False):
    """

    adapted from SIMA (https://github.com/losonczylab) and the
    scikit-image (http://scikit-image.org/) package.


    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    Efficient subpixel image translation registration by cross-correlation.

    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.

    Args:
        src_image : ndarray
            Reference image.

        target_image : ndarray
            Image to register.  Must be same dimensionality as ``src_image``.

        upsample_factor : int, optional
            Upsampling factor. Images will be registered to within
            ``1 / upsample_factor`` of a pixel. For example
            ``upsample_factor == 20`` means the images will be registered
            within 1/20th of a pixel.  Default is 1 (no upsampling)

        space : string, one of "real" or "fourier"
            Defines how the algorithm interprets input data.  "real" means data
            will be FFT'd to compute the correlation, while "fourier" data will
            bypass FFT of input data.  Case insensitive.

        use_cuda : bool, optional
            Use skcuda.fft (if available). Default: False

    Returns:
        shifts : ndarray
            Shift vector (in pixels) required to register ``target_image`` with
            ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)

        error : float
            Translation invariant normalized RMS error between ``src_image`` and
            ``target_image``.

        phasediff : float
            Global phase difference between the two images (should be
            zero if images are non-negative).

    Raises:
     NotImplementedError "Error: register_translation only supports "
                                  "subpixel registration for 2D images"

     ValueError "Error: images must really be same size for "
                         "register_translation"

     ValueError "Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument."

    References:
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008).
    """
    # images must be the same shape
    if src_image.shape != target_image.shape:
        raise ValueError("Error: images must really be same size for "
                         "register_translation")

    # only 2D data makes sense right now
    if src_image.ndim != 2 and upsample_factor > 1:
        raise NotImplementedError("Error: register_translation only supports "
                                  "subpixel registration for 2D images")

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        if opencv:
            src_freq_1 = fftn(
                src_image, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
            src_freq = src_freq_1[:, :, 0] + 1j * src_freq_1[:, :, 1]
            src_freq = np.array(src_freq, dtype=np.complex128, copy=False)
            target_freq_1 = fftn(
                target_image, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
            target_freq = target_freq_1[:, :, 0] + 1j * target_freq_1[:, :, 1]
            target_freq = np.array(
                target_freq, dtype=np.complex128, copy=False)
        else:
            src_image_cpx = np.array(
                src_image, dtype=np.complex128, copy=False)
            target_image_cpx = np.array(
                target_image, dtype=np.complex128, copy=False)
            src_freq = np.fft.fftn(src_image_cpx)
            target_freq = np.fft.fftn(target_image_cpx)

    else:
        raise ValueError("Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument.")

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    
    if opencv:
        image_product_cv = np.dstack(
            [np.real(image_product), np.imag(image_product)])
        cross_correlation = fftn(
            image_product_cv, flags=cv2.DFT_INVERSE + cv2.DFT_SCALE)
        cross_correlation = cross_correlation[:,
                                              :, 0] + 1j * cross_correlation[:, :, 1]
    else:
        cross_correlation = ifftn(image_product)

    # Locate maximum
    new_cross_corr = np.abs(cross_correlation)

    if (shifts_lb is not None) or (shifts_ub is not None):

        if (shifts_lb[0] < 0) and (shifts_ub[0] >= 0):
            new_cross_corr[shifts_ub[0]:shifts_lb[0], :] = 0
        else:
            new_cross_corr[:shifts_lb[0], :] = 0
            new_cross_corr[shifts_ub[0]:, :] = 0

        if (shifts_lb[1] < 0) and (shifts_ub[1] >= 0):
            new_cross_corr[:, shifts_ub[1]:shifts_lb[1]] = 0
        else:
            new_cross_corr[:, :shifts_lb[1]] = 0
            new_cross_corr[:, shifts_ub[1]:] = 0
    else:

        new_cross_corr[max_shifts[0]:-max_shifts[0], :] = 0

        new_cross_corr[:, max_shifts[1]:-max_shifts[1]] = 0

    maxima = np.unravel_index(np.argmax(new_cross_corr),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(old_div(axis_size, 2))
                          for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:

        src_amp = old_div(np.sum(np.abs(src_freq) ** 2), src_freq.size)
        target_amp = old_div(
            np.sum(np.abs(target_freq) ** 2), target_freq.size)
        CCmax = cross_correlation.max()
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        # Initial shift estimate in upsampled grid
        shifts = old_div(np.round(shifts * upsample_factor), upsample_factor)
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(old_div(upsampled_region_size, 2.0))
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freq.size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor

        cross_correlation = _upsampled_dft(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(
            np.argmax(np.abs(cross_correlation)),
            cross_correlation.shape),
            dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + old_div(maxima, upsample_factor)
        CCmax = cross_correlation.max()
        src_amp = _upsampled_dft(src_freq * src_freq.conj(),
                                 1, upsample_factor)[0, 0]
        src_amp /= normalization
        target_amp = _upsampled_dft(target_freq * target_freq.conj(),
                                    1, upsample_factor)[0, 0]
        target_amp /= normalization

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    return shifts, src_freq, _compute_phasediff(CCmax)

#%%

def apply_shifts_dft(src_freq, shifts, diffphase, is_freq=True, border_nan=True):
    """
    adapted from SIMA (https://github.com/losonczylab) and the
    scikit-image (http://scikit-image.org/) package.


    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    Args:
        apply shifts using inverse dft
        src_freq: ndarray
            if is_freq it is fourier transform image else original image
        shifts: shifts to apply
        diffphase: comes from the register_translation output
    """

    is3D = len(src_freq.shape) == 3
    if not is_freq:
        if is3D:
            src_freq = np.fft.fftn(src_freq)
        else:
            src_freq = np.dstack([np.real(src_freq), np.imag(src_freq)])
            src_freq = fftn(src_freq, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
            src_freq = src_freq[:, :, 0] + 1j * src_freq[:, :, 1]
            src_freq = np.array(src_freq, dtype=np.complex128, copy=False)

    if not is3D:
        shifts = shifts[::-1]
        nc, nr = np.shape(src_freq)
        Nr = ifftshift(np.arange(-np.fix(nr/2.), np.ceil(nr/2.)))
        Nc = ifftshift(np.arange(-np.fix(nc/2.), np.ceil(nc/2.)))
        Nr, Nc = np.meshgrid(Nr, Nc)
        Greg = src_freq * np.exp(1j * 2 * np.pi *
                                 (-shifts[0] * 1. * Nr / nr - shifts[1] * 1. * Nc / nc))
    else:
        #shifts = np.array([*shifts[:-1][::-1],shifts[-1]])
        shifts = np.array(list(shifts[:-1][::-1]) + [shifts[-1]])
        nc, nr, nd = np.array(np.shape(src_freq), dtype=float)
        Nr = ifftshift(np.arange(-np.fix(nr / 2.), np.ceil(nr / 2.)))
        Nc = ifftshift(np.arange(-np.fix(nc / 2.), np.ceil(nc / 2.)))
        Nd = ifftshift(np.arange(-np.fix(nd / 2.), np.ceil(nd / 2.)))
        Nr, Nc, Nd = np.meshgrid(Nr, Nc, Nd)
        Greg = src_freq * np.exp(-1j * 2 * np.pi *
                                 (-shifts[0] * Nr / nr - shifts[1] * Nc / nc -
                                  shifts[2] * Nd / nd))

    Greg = Greg.dot(np.exp(1j * diffphase))
    if is3D:
        new_img = np.real(np.fft.ifftn(Greg))
    else:
        Greg = np.dstack([np.real(Greg), np.imag(Greg)])
        new_img = ifftn(Greg)[:, :, 0]

    if border_nan is not False:
        max_w, max_h, min_w, min_h = 0, 0, 0, 0
        max_h, max_w = np.ceil(np.maximum(
            (max_h, max_w), shifts[:2])).astype(np.int)
        min_h, min_w = np.floor(np.minimum(
            (min_h, min_w), shifts[:2])).astype(np.int)
        if is3D:
            max_d = np.ceil(np.maximum(0, shifts[2])).astype(np.int)
            min_d = np.floor(np.minimum(0, shifts[2])).astype(np.int)
        if border_nan is True:
            new_img[:max_h, :] = np.nan
            if min_h < 0:
                new_img[min_h:, :] = np.nan
            new_img[:, :max_w] = np.nan
            if min_w < 0:
                new_img[:, min_w:] = np.nan
            if is3D:
                new_img[:, :, :max_d] = np.nan
                if min_d < 0:
                    new_img[:, :, min_d:] = np.nan
        elif border_nan == 'min':
            min_ = np.nanmin(new_img)
            new_img[:max_h, :] = min_
            if min_h < 0:
                new_img[min_h:, :] = min_
            new_img[:, :max_w] = min_
            if min_w < 0:
                new_img[:, min_w:] = min_
            if is3D:
                new_img[:, :, :max_d] = min_
                if min_d < 0:
                    new_img[:, :, min_d:] = min_
        elif border_nan == 'copy':
            new_img[:max_h] = new_img[max_h]
            if min_h < 0:
                new_img[min_h:] = new_img[min_h-1]
            if max_w > 0:
                new_img[:, :max_w] = new_img[:, max_w, np.newaxis]
            if min_w < 0:
                new_img[:, min_w:] = new_img[:, min_w-1, np.newaxis]
            if is3D:
                if max_d > 0:
                    new_img[:, :, :max_d] = new_img[:, :, max_d, np.newaxis]
                if min_d < 0:
                    new_img[:, :, min_d:] = new_img[:, :, min_d-1, np.newaxis]

    return new_img


#%%
def sliding_window(image, overlaps, strides):
    """ efficiently and lazily slides a window across the image

    Args: 
        img:ndarray 2D
            image that needs to be slices

        windowSize: tuple
            dimension of the patch

        strides: tuple
            stride in each dimension

     Returns:
         iterator containing five items
              dim_1, dim_2 coordinates in the patch grid
              x, y: bottom border of the patch in the original matrix

              patch: the patch
     """
    windowSize = np.add(overlaps, strides)
    range_1 = list(range(
        0, image.shape[0] - windowSize[0], strides[0])) + [image.shape[0] - windowSize[0]]
    range_2 = list(range(
        0, image.shape[1] - windowSize[1], strides[1])) + [image.shape[1] - windowSize[1]]
    for dim_1, x in enumerate(range_1):
        for dim_2, y in enumerate(range_2):
            # yield the current window
            yield (dim_1, dim_2, x, y, image[x:x + windowSize[0], y:y + windowSize[1]])

def sliding_window_3d(image, overlaps, strides):
    """ efficiently and lazily slides a window across the image

    Args: 
        img:ndarray 3D
            image that needs to be slices

        windowSize: tuple
            dimension of the patch

        strides: tuple
            stride in each dimension

     Returns:
         iterator containing seven items
              dim_1, dim_2, dim_3 coordinates in the patch grid
              x, y, z: bottom border of the patch in the original matrix

              patch: the patch
     """
    windowSize = np.add(overlaps, strides)
    range_1 = list(range(
        0, image.shape[0] - windowSize[0], strides[0])) + [image.shape[0] - windowSize[0]]
    range_2 = list(range(
        0, image.shape[1] - windowSize[1], strides[1])) + [image.shape[1] - windowSize[1]]
    range_3 = list(range(
        0, image.shape[2] - windowSize[2], strides[2])) + [image.shape[2] - windowSize[2]]
    for dim_1, x in enumerate(range_1):
        for dim_2, y in enumerate(range_2):
            for dim_3, z in enumerate(range_3):
                # yield the current window
                yield (dim_1, dim_2, dim_3, x, y, z, image[x:x + windowSize[0], y:y + windowSize[1], z:z + windowSize[2]])

def iqr(a):
    return np.percentile(a, 75) - np.percentile(a, 25)

def create_weight_matrix_for_blending(img, overlaps, strides):
    """ create a matrix that is used to normalize the intersection of the stiched patches

    Args:
        img: original image, ndarray

        shapes, overlaps, strides:  tuples
            shapes, overlaps and strides of the patches

    Returns:
        weight_mat: normalizing weight matrix
    """
    shapes = np.add(strides, overlaps)

    max_grid_1, max_grid_2 = np.max(
        np.array([it[:2] for it in sliding_window(img, overlaps, strides)]), 0)

    for grid_1, grid_2, _, _, _ in sliding_window(img, overlaps, strides):

        weight_mat = np.ones(shapes)

        if grid_1 > 0:
            weight_mat[:overlaps[0], :] = np.linspace(
                0, 1, overlaps[0])[:, None]
        if grid_1 < max_grid_1:
            weight_mat[-overlaps[0]:,
                       :] = np.linspace(1, 0, overlaps[0])[:, None]
        if grid_2 > 0:
            weight_mat[:, :overlaps[1]] = weight_mat[:, :overlaps[1]
                                                     ] * np.linspace(0, 1, overlaps[1])[None, :]
        if grid_2 < max_grid_2:
            weight_mat[:, -overlaps[1]:] = weight_mat[:, -
                                                      overlaps[1]:] * np.linspace(1, 0, overlaps[1])[None, :]

        yield weight_mat

def high_pass_filter_space(img_orig, gSig_filt):
    ksize = tuple([(3 * i) // 2 * 2 + 1 for i in gSig_filt])
    ker = cv2.getGaussianKernel(ksize[0], gSig_filt[0])
    ker2D = ker.dot(ker.T)
    nz = np.nonzero(ker2D >= ker2D[:, 0].max())
    zz = np.nonzero(ker2D < ker2D[:, 0].max())
    ker2D[nz] -= ker2D[nz].mean()
    ker2D[zz] = 0
    return cv2.filter2D(np.array(img_orig, dtype=np.float32), -1, ker2D, borderType=cv2.BORDER_REFLECT)

def tile_and_correct(img, template, strides, overlaps, max_shifts, newoverlaps=None, newstrides=None, upsample_factor_grid=4,
                     upsample_factor_fft=10, show_movie=False, max_deviation_rigid=2, add_to_movie=0, shifts_opencv=False, gSig_filt=None,
                     use_cuda=False, border_nan=True):
    """ perform piecewise rigid motion correction iteration, by
        1) dividing the FOV in patches
        2) motion correcting each patch separately
        3) upsampling the motion correction vector field
        4) stiching back together the corrected subpatches

    Args:
        img: ndaarray 2D
            image to correct

        template: ndarray
            reference image

        strides: tuple
            strides of the patches in which the FOV is subdivided

        overlaps: tuple
            amount of pixel overlaping between patches along each dimension

        max_shifts: tuple
            max shifts in x and y

        newstrides:tuple
            strides between patches along each dimension when upsampling the vector fields

        newoverlaps:tuple
            amount of pixel overlaping between patches along each dimension when upsampling the vector fields

        upsample_factor_grid: int
            if newshapes or newstrides are not specified this is inferred upsampling by a constant factor the cvector field

        upsample_factor_fft: int
            resolution of fractional shifts

        show_movie: boolean whether to visualize the original and corrected frame during motion correction

        max_deviation_rigid: int
            maximum deviation in shifts of each patch from the rigid shift (should not be large)

        add_to_movie: if movie is too negative the correction might have some issues. In this case it is good to add values so that it is non negative most of the times

        filt_sig_size: tuple
            standard deviation and size of gaussian filter to center filter data in case of one photon imaging data

        use_cuda : bool, optional
            Use skcuda.fft (if available). Default: False

        border_nan : bool or string, optional
            specifies how to deal with borders. (True, False, 'copy', 'min')

    Returns:
        (new_img, total_shifts, start_step, xy_grid)
            new_img: ndarray, corrected image


    """

    img = img.astype(np.float64).copy()
    template = template.astype(np.float64).copy()

    if gSig_filt is not None:

        img_orig = img.copy()
        img = high_pass_filter_space(img_orig, gSig_filt)

    img = img + add_to_movie
    template = template + add_to_movie

    # compute rigid shifts
    rigid_shts, sfr_freq, diffphase = register_translation(
        img, template, upsample_factor=upsample_factor_fft, max_shifts=max_shifts, use_cuda=use_cuda)

    if max_deviation_rigid == 0:

        if shifts_opencv:
            if gSig_filt is not None:
                img = img_orig

            new_img = apply_shift_iteration(
                img, (-rigid_shts[0], -rigid_shts[1]), border_nan=border_nan)

        else:

            if gSig_filt is not None:
                raise Exception(
                    'The use of FFT and filtering options have not been tested. Set opencv=True')

            new_img = apply_shifts_dft(
                sfr_freq, (-rigid_shts[0], -rigid_shts[1]), diffphase, border_nan=border_nan)

        return new_img - add_to_movie, (-rigid_shts[0], -rigid_shts[1]), None, None
    else:
        # extract patches
        templates = [
            it[-1] for it in sliding_window(template, overlaps=overlaps, strides=strides)]
        xy_grid = [(it[0], it[1]) for it in sliding_window(
            template, overlaps=overlaps, strides=strides)]
        num_tiles = np.prod(np.add(xy_grid[-1], 1))
        imgs = [it[-1]
                for it in sliding_window(img, overlaps=overlaps, strides=strides)]
        dim_grid = tuple(np.add(xy_grid[-1], 1))

        if max_deviation_rigid is not None:

            lb_shifts = np.ceil(np.subtract(
                rigid_shts, max_deviation_rigid)).astype(int)
            ub_shifts = np.floor(
                np.add(rigid_shts, max_deviation_rigid)).astype(int)

        else:

            lb_shifts = None
            ub_shifts = None

        # extract shifts for each patch
        shfts_et_all = [register_translation(
            a, b, c, shifts_lb=lb_shifts, shifts_ub=ub_shifts, max_shifts=max_shifts, use_cuda=use_cuda) for a, b, c in zip(
            imgs, templates, [upsample_factor_fft] * num_tiles)]
        shfts = [sshh[0] for sshh in shfts_et_all]
        diffs_phase = [sshh[2] for sshh in shfts_et_all]
        # create a vector field
        shift_img_x = np.reshape(np.array(shfts)[:, 0], dim_grid)
        shift_img_y = np.reshape(np.array(shfts)[:, 1], dim_grid)
        diffs_phase_grid = np.reshape(np.array(diffs_phase), dim_grid)

        if shifts_opencv:
            if gSig_filt is not None:
                img = img_orig

            dims = img.shape
            x_grid, y_grid = np.meshgrid(np.arange(0., dims[1]).astype(
                np.float32), np.arange(0., dims[0]).astype(np.float32))
            m_reg = cv2.remap(img, cv2.resize(shift_img_y.astype(np.float32), dims[::-1]) + x_grid,
                              cv2.resize(shift_img_x.astype(np.float32), dims[::-1]) + y_grid,
                              cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                             # borderValue=add_to_movie)
            total_shifts = [
                    (-x, -y) for x, y in zip(shift_img_x.reshape(num_tiles), shift_img_y.reshape(num_tiles))]
            return m_reg - add_to_movie, total_shifts, None, None

        # create automatically upsample parameters if not passed
        if newoverlaps is None:
            newoverlaps = overlaps
        if newstrides is None:
            newstrides = tuple(
                np.round(np.divide(strides, upsample_factor_grid)).astype(np.int))

        newshapes = np.add(newstrides, newoverlaps)

        imgs = [it[-1]
                for it in sliding_window(img, overlaps=newoverlaps, strides=newstrides)]

        xy_grid = [(it[0], it[1]) for it in sliding_window(
            img, overlaps=newoverlaps, strides=newstrides)]

        start_step = [(it[2], it[3]) for it in sliding_window(
            img, overlaps=newoverlaps, strides=newstrides)]

        dim_new_grid = tuple(np.add(xy_grid[-1], 1))

        shift_img_x = cv2.resize(
            shift_img_x, dim_new_grid[::-1], interpolation=cv2.INTER_CUBIC)
        shift_img_y = cv2.resize(
            shift_img_y, dim_new_grid[::-1], interpolation=cv2.INTER_CUBIC)
        diffs_phase_grid_us = cv2.resize(
            diffs_phase_grid, dim_new_grid[::-1], interpolation=cv2.INTER_CUBIC)

        num_tiles = np.prod(dim_new_grid)

        max_shear = np.percentile(
            [np.max(np.abs(np.diff(ssshh, axis=xxsss))) for ssshh, xxsss in itertools.product(
                [shift_img_x, shift_img_y], [0, 1])], 75)

        total_shifts = [
            (-x, -y) for x, y in zip(shift_img_x.reshape(num_tiles), shift_img_y.reshape(num_tiles))]
        total_diffs_phase = [
            dfs for dfs in diffs_phase_grid_us.reshape(num_tiles)]

        if gSig_filt is not None:
            raise Exception(
                'The use of FFT and filtering options have not been tested. Set opencv=True')

        imgs = [apply_shifts_dft(im, (
            sh[0], sh[1]), dffphs, is_freq=False, border_nan=border_nan) for im, sh, dffphs in zip(
            imgs, total_shifts, total_diffs_phase)]

        normalizer = np.zeros_like(img) * np.nan
        new_img = np.zeros_like(img) * np.nan

        weight_matrix = create_weight_matrix_for_blending(
            img, newoverlaps, newstrides)

        if max_shear < 0.5:
            for (x, y), (_, _), im, (_, _), weight_mat in zip(start_step, xy_grid, imgs, total_shifts, weight_matrix):

                prev_val_1 = normalizer[x:x + newshapes[0], y:y + newshapes[1]]

                normalizer[x:x + newshapes[0], y:y + newshapes[1]] = np.nansum(
                    np.dstack([~np.isnan(im) * 1 * weight_mat, prev_val_1]), -1)
                prev_val = new_img[x:x + newshapes[0], y:y + newshapes[1]]
                new_img[x:x + newshapes[0], y:y + newshapes[1]
                        ] = np.nansum(np.dstack([im * weight_mat, prev_val]), -1)

            new_img = old_div(new_img, normalizer)

        else:  # in case the difference in shift between neighboring patches is larger than 0.5 pixels we do not interpolate in the overlaping area
            half_overlap_x = np.int(newoverlaps[0] / 2)
            half_overlap_y = np.int(newoverlaps[1] / 2)
            for (x, y), (idx_0, idx_1), im, (_, _), weight_mat in zip(start_step, xy_grid, imgs, total_shifts, weight_matrix):

                if idx_0 == 0:
                    x_start = x
                else:
                    x_start = x + half_overlap_x

                if idx_1 == 0:
                    y_start = y
                else:
                    y_start = y + half_overlap_y

                x_end = x + newshapes[0]
                y_end = y + newshapes[1]
                new_img[x_start:x_end,
                        y_start:y_end] = im[x_start - x:, y_start - y:]

        if show_movie:
            img = apply_shifts_dft(
                sfr_freq, (-rigid_shts[0], -rigid_shts[1]), diffphase, border_nan=border_nan)
            img_show = np.vstack([new_img, img])

            img_show = cv2.resize(img_show, None, fx=1, fy=1)

            cv2.imshow('frame', old_div(img_show, np.percentile(template, 99)))
            cv2.waitKey(int(1. / 500 * 1000))

        else:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        return new_img - add_to_movie, total_shifts, start_step, xy_grid

#%%        
def tile_and_correct_3d(img, template, strides, overlaps, max_shifts, newoverlaps=None, newstrides=None, upsample_factor_grid=4,
                     upsample_factor_fft=10, show_movie=False, max_deviation_rigid=2, add_to_movie=0, shifts_opencv=False, gSig_filt=None,
                     use_cuda=False, border_nan=True):
    """ perform piecewise rigid motion correction iteration, by
        1) dividing the FOV in patches
        2) motion correcting each patch separately
        3) upsampling the motion correction vector field
        4) stiching back together the corrected subpatches

    Args:
        img: ndaarray 3D
            image to correct

        template: ndarray
            reference image

        strides: tuple
            strides of the patches in which the FOV is subdivided

        overlaps: tuple
            amount of pixel overlaping between patches along each dimension

        max_shifts: tuple
            max shifts in x, y, and z

        newstrides:tuple
            strides between patches along each dimension when upsampling the vector fields

        newoverlaps:tuple
            amount of pixel overlaping between patches along each dimension when upsampling the vector fields

        upsample_factor_grid: int
            if newshapes or newstrides are not specified this is inferred upsampling by a constant factor the cvector field

        upsample_factor_fft: int
            resolution of fractional shifts

        show_movie: boolean whether to visualize the original and corrected frame during motion correction

        max_deviation_rigid: int
            maximum deviation in shifts of each patch from the rigid shift (should not be large)

        add_to_movie: if movie is too negative the correction might have some issues. In this case it is good to add values so that it is non negative most of the times

        filt_sig_size: tuple
            standard deviation and size of gaussian filter to center filter data in case of one photon imaging data

        use_cuda : bool, optional
            Use skcuda.fft (if available). Default: False

        border_nan : bool or string, optional
            specifies how to deal with borders. (True, False, 'copy', 'min')

    Returns:
        (new_img, total_shifts, start_step, xyz_grid)
            new_img: ndarray, corrected image


    """

    img = img.astype(np.float64).copy()
    template = template.astype(np.float64).copy()

    if gSig_filt is not None:

        img_orig = img.copy()
        img = high_pass_filter_space(img_orig, gSig_filt)

    img = img + add_to_movie
    template = template + add_to_movie

    # compute rigid shifts
    rigid_shts, sfr_freq, diffphase = register_translation_3d(
        img, template, upsample_factor=upsample_factor_fft, max_shifts=max_shifts)
    
    if max_deviation_rigid == 0: # if rigid shifts only
#        if shifts_opencv:
            # NOTE: opencv does not support 3D operations - skimage is used instead
 #       else:

        if gSig_filt is not None:
            raise Exception(
                'The use of FFT and filtering options have not been tested. Set opencv=True')

        new_img = apply_shifts_dft( # TODO: check
            sfr_freq, (rigid_shts[0], rigid_shts[1], rigid_shts[2]), diffphase, border_nan=border_nan)
        return new_img - add_to_movie, (-rigid_shts[0], -rigid_shts[1], -rigid_shts[2]), None, None
    else:
        # extract patches
        templates = [
            it[-1] for it in sliding_window_3d(template, overlaps=overlaps, strides=strides)]
        xyz_grid = [(it[0], it[1], it[2]) for it in sliding_window_3d(
            template, overlaps=overlaps, strides=strides)]
        num_tiles = np.prod(np.add(xyz_grid[-1], 1))
        imgs = [it[-1]
                for it in sliding_window_3d(img, overlaps=overlaps, strides=strides)]
        dim_grid = tuple(np.add(xyz_grid[-1], 1))

        if max_deviation_rigid is not None:
            lb_shifts = np.ceil(np.subtract(
                rigid_shts, max_deviation_rigid)).astype(int)
            ub_shifts = np.floor(
                np.add(rigid_shts, max_deviation_rigid)).astype(int)

        else:
            lb_shifts = None
            ub_shifts = None

        # extract shifts for each patch
        shfts_et_all = [register_translation_3d(
            a, b, c, shifts_lb=lb_shifts, shifts_ub=ub_shifts, max_shifts=max_shifts) for a, b, c in zip(
            imgs, templates, [upsample_factor_fft] * num_tiles)]
        shfts = [sshh[0] for sshh in shfts_et_all]
        diffs_phase = [sshh[2] for sshh in shfts_et_all]
        # create a vector field
        shift_img_x = np.reshape(np.array(shfts)[:, 0], dim_grid)
        shift_img_y = np.reshape(np.array(shfts)[:, 1], dim_grid)
        shift_img_z = np.reshape(np.array(shfts)[:, 2], dim_grid)
        diffs_phase_grid = np.reshape(np.array(diffs_phase), dim_grid)

        #  shifts_opencv doesn't make sense here- replace with shifts_skimage
        if shifts_opencv:
            if gSig_filt is not None:
                img = img_orig

            dims = img.shape
            x_grid, y_grid, z_grid = np.meshgrid(np.arange(0., dims[1]).astype(
                np.float32), np.arange(0., dims[0]).astype(np.float32),
                np.arange(0., dims[2]).astype(np.float32))
            m_reg = warp_sk(img, np.stack((resize_sk(shift_img_x.astype(np.float32), dims) + y_grid,
                              resize_sk(shift_img_y.astype(np.float32), dims) + x_grid,
                              resize_sk(shift_img_z.astype(np.float32), dims) + z_grid),axis=0),
                              order=3, mode='constant')
                             # borderValue=add_to_movie)
            total_shifts = [
                    (-x, -y, z) for x, y, z in zip(shift_img_x.reshape(num_tiles), shift_img_y.reshape(num_tiles), shift_img_z.reshape(num_tiles))]
            return m_reg - add_to_movie, total_shifts, None, None

        # create automatically upsample parameters if not passed
        if newoverlaps is None:
            newoverlaps = overlaps
        if newstrides is None:
            newstrides = tuple(
                np.round(np.divide(strides, upsample_factor_grid)).astype(np.int))

        newshapes = np.add(newstrides, newoverlaps)

        imgs = [it[-1]
                for it in sliding_window_3d(img, overlaps=newoverlaps, strides=newstrides)]

        xyz_grid = [(it[0], it[1], it[2]) for it in sliding_window_3d(
            img, overlaps=newoverlaps, strides=newstrides)]

        start_step = [(it[3], it[4], it[5]) for it in sliding_window_3d(
            img, overlaps=newoverlaps, strides=newstrides)]

        dim_new_grid = tuple(np.add(xyz_grid[-1], 1))

        shift_img_x = resize_sk(
            shift_img_x, dim_new_grid[::-1], order=3)
        shift_img_y = resize_sk(
            shift_img_y, dim_new_grid[::-1], order=3)
        shift_img_z = resize_sk(
            shift_img_z, dim_new_grid[::-1], order=3)
        diffs_phase_grid_us = resize_sk(
            diffs_phase_grid, dim_new_grid[::-1], order=3)

        num_tiles = np.prod(dim_new_grid)

        # what dimension shear should be looked at? shearing for 3d point scanning happens in y and z but no for plane-scanning
        max_shear = np.percentile(
            [np.max(np.abs(np.diff(ssshh, axis=xxsss))) for ssshh, xxsss in itertools.product(
                [shift_img_x, shift_img_y], [0, 1])], 75)

        total_shifts = [
            (-x, -y, -z) for x, y, z in zip(shift_img_x.reshape(num_tiles), shift_img_y.reshape(num_tiles), shift_img_z.reshape(num_tiles))]
        total_diffs_phase = [
            dfs for dfs in diffs_phase_grid_us.reshape(num_tiles)]

        if gSig_filt is not None:
            raise Exception(
                'The use of FFT and filtering options have not been tested. Set opencv=True')

        imgs = [apply_shifts_dft(im, (
            sh[0], sh[1], sh[2]), dffphs, is_freq=False, border_nan=border_nan) for im, sh, dffphs in zip(
            imgs, total_shifts, total_diffs_phase)]

        normalizer = np.zeros_like(img) * np.nan
        new_img = np.zeros_like(img) * np.nan

        weight_matrix = create_weight_matrix_for_blending(
            img, newoverlaps, newstrides)

        if max_shear < 0.5:
            for (x, y, z), (_, _, _), im, (_, _, _), weight_mat in zip(start_step, xyz_grid, imgs, total_shifts, weight_matrix):

                prev_val_1 = normalizer[x:x + newshapes[0], y:y + newshapes[1], z:z + newshapes[2]]

                normalizer[x:x + newshapes[0], y:y + newshapes[1], z:z + newshapes[2]] = np.nansum(
                    np.dstack([~np.isnan(im) * 1 * weight_mat, prev_val_1]), -1)
                prev_val = new_img[x:x + newshapes[0], y:y + newshapes[1], z:z + newshapes[2]]
                new_img[x:x + newshapes[0], y:y + newshapes[1], z:z + newshapes[2]
                        ] = np.nansum(np.dstack([im * weight_mat, prev_val]), -1)
                
            new_img = old_div(new_img, normalizer)

        else:  # in case the difference in shift between neighboring patches is larger than 0.5 pixels we do not interpolate in the overlaping area
            half_overlap_x = np.int(newoverlaps[0] / 2)
            half_overlap_y = np.int(newoverlaps[1] / 2)
            half_overlap_z = np.int(newoverlaps[2] / 2)
            
            for (x, y, z), (idx_0, idx_1, idx_2), im, (_, _, _), weight_mat in zip(start_step, xyz_grid, imgs, total_shifts, weight_matrix):

                if idx_0 == 0:
                    x_start = x
                else:
                    x_start = x + half_overlap_x

                if idx_1 == 0:
                    y_start = y
                else:
                    y_start = y + half_overlap_y

                if idx_2 == 0:
                    z_start = z
                else:
                    z_start = z + half_overlap_z
                    
                x_end = x + newshapes[0]
                y_end = y + newshapes[1]
                z_end = z + newshapes[2]
                new_img[x_start:x_end,y_start:y_end,
                        z_start:z_end] = im[x_start - x:, y_start - y:, z_start -z:]

        if show_movie:
            img = apply_shifts_dft(
                sfr_freq, (-rigid_shts[0], -rigid_shts[1], -rigid_shts[2]), diffphase, border_nan=border_nan)
            img_show = np.vstack([new_img, img])

            img_show = resize_sk(img_show, None, fx=1, fy=1, fz=1)

            cv2.imshow('frame', old_div(img_show, np.percentile(template, 99)))
            cv2.waitKey(int(1. / 500 * 1000))

        else:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        return new_img - add_to_movie, total_shifts, start_step, xyz_grid

#%%
def motion_correct_batch_rigid(video, max_shifts, dview=None, splits=56, num_splits_to_process=None, num_iter=1,
                               template=None, shifts_opencv=False, add_to_movie=None,
                               nonneg_movie=False, gSig_filt=None, subidx=slice(None, None, 1), use_cuda=False,
                               border_nan=True, var_name_hdf5='mov', is3D=False, indices=(slice(None), slice(None))):
    """
    Function that perform memory efficient hyper parallelized rigid motion corrections while also saving a memory mappable file

    Args:
        max_shifts: tuple
            x and y (and z if 3D) maximum allowed shifts

        dview: ipyparallel view
            used to perform parallel computing

        splits: int
            number of batches in which the movies is subdivided

        num_splits_to_process: int
            number of batches to process. when not None, the movie is not saved since only a random subset of batches will be processed

        num_iter: int
            number of iterations to perform. The more iteration the better will be the template.

        template: ndarray
            if a good approximation of the template to register is available, it can be used

        shifts_opencv: boolean
             toggle the shifts applied with opencv, if yes faster but induces some smoothing

        subidx: slice
            Indices to slice

        use_cuda : bool, optional
            Use skcuda.fft (if available). Default: False

        indices: tuple(slice), default: (slice(None), slice(None))
           Use that to apply motion correction only on a part of the FOV

    Returns:
         total_template:ndarray

         templates:list
              list of produced templates, one per batch

         shifts: list
              inferred rigid shifts to correct the movie

    Raises:
        Exception 'The movie contains nans. Nans are not allowed!'

    """
    

    m = video
    if template is None:
        if gSig_filt is not None:
            m = np.array([high_pass_filter_space(m_, gSig_filt) for m_ in m])
        if is3D:
            # TODO - motion_correct_3d needs to be implemented in movies.py
            template = bin_median_3d(m) # motion_correct_3d has not been implemented yet - instead initialize to just median image
#            template = caiman.motion_correction.bin_median_3d(
#                    m.motion_correct_3d(max_shifts[2], max_shifts[1], max_shifts[0], template=None)[0])
        else:
            template = bin_median(m.motion_correct(max_shifts[1], max_shifts[0], template=None)[0])
        
    new_templ = template
    if add_to_movie is None:
        add_to_movie = -np.min(template)

    if np.isnan(add_to_movie):
        logging.error('The movie contains NaNs. NaNs are not allowed!')
        raise Exception('The movie contains NaNs. NaNs are not allowed!')
    else:
        logging.debug('Adding to movie ' + str(add_to_movie))

    res_rig:List = []
    for iter_ in range(num_iter):
        logging.debug(iter_)
        old_templ = new_templ.copy()
        if iter_ == num_iter - 1:
            logging.debug('saving!')

        
        res_rig = motion_correction_piecewise(video, splits, strides=None, overlaps=None,
                                             add_to_movie=add_to_movie, template=old_templ, max_shifts=max_shifts, max_deviation_rigid=0,
                                             dview=dview, subidx = subidx,
                                             num_splits=num_splits_to_process, shifts_opencv=shifts_opencv, nonneg_movie=nonneg_movie, gSig_filt=gSig_filt,
                                             use_cuda=use_cuda, border_nan=border_nan, var_name_hdf5=var_name_hdf5, is3D=is3D,
                                             indices=indices)
        if is3D:
            new_templ = np.nanmedian(np.stack([r[-2] for r in res_rig]), 0)           
        else:
            new_templ = np.nanmedian(np.dstack([r[-2] for r in res_rig]), -1)
        if gSig_filt is not None:
            new_templ = high_pass_filter_space(new_templ, gSig_filt)

#        logging.debug((old_div(np.linalg.norm(new_templ - old_templ), np.linalg.norm(old_templ))))

    total_template = new_templ
    templates = []
    shifts:List = []
    mc = []
    for rr in res_rig:
        shift_info, idxs, tmpl, mc_cur = rr
        templates.append(tmpl)
        shifts += [sh[0] for sh in shift_info[:len(idxs)]]
        mc.append(mc_cur)

    return total_template, templates, shifts, mc

def motion_correct_batch_pwrigid(video, max_shifts, strides, overlaps, add_to_movie, newoverlaps=None, newstrides=None,
                                 dview=None, upsample_factor_grid=4, max_deviation_rigid=3,
                                 splits=56, num_splits_to_process=None, num_iter=1,
                                 template=None, shifts_opencv=False, nonneg_movie=False, gSig_filt=None,
                                 use_cuda=False, border_nan=True, var_name_hdf5='mov', is3D=False,
                                 indices=(slice(None), slice(None))):
    """
    Function that perform memory efficient hyper parallelized rigid motion corrections while also saving a memory mappable file

    Args:
        strides: tuple
            strides of patches along x and y (and z if 3D)

        overlaps:
            overlaps of patches along x and y (and z if 3D). example: If strides = (64,64) and overlaps (32,32) patches will be (96,96)

        newstrides: tuple
            overlaps after upsampling

        newoverlaps: tuple
            strides after upsampling

        max_shifts: tuple
            x and y maximum allowed shifts (and z if 3D)

        dview: ipyparallel view
            used to perform parallel computing

        splits: int
            number of batches in which the movies is subdivided

        num_splits_to_process: int
            number of batches to process. when not None, the movie is not saved since only a random subset of batches will be processed

        num_iter: int
            number of iterations to perform. The more iteration the better will be the template.

        template: ndarray
            if a good approximation of the template to register is available, it can be used

        shifts_opencv: boolean
             toggle the shifts applied with opencv, if yes faster but induces some smoothing

        use_cuda : bool, optional
            Use skcuda.fft (if available). Default: False

        indices: tuple(slice), default: (slice(None), slice(None))
           Use that to apply motion correction only on a part of the FOV

    Returns:
        total_template:ndarray

        templates:list
            list of produced templates, one per batch

        shifts: list
            inferred rigid shifts to corrrect the movie

    Raises:
        Exception 'You need to initialize the template with a good estimate. See the motion'
                        '_correct_batch_rigid function'
    """
    
    
    if template is None:
        raise Exception('You need to initialize the template with a good estimate. See the motion'
                        '_correct_batch_rigid function')
    else:
        new_templ = template

    if np.isnan(add_to_movie):
        logging.error('The template contains NaNs. NaNs are not allowed!')
        raise Exception('The template contains NaNs. NaNs are not allowed!')
    else:
        logging.debug('Adding to movie ' + str(add_to_movie))

    for iter_ in range(num_iter):
        logging.debug(iter_)
        old_templ = new_templ.copy()
        
        res_el = motion_correction_piecewise(video, splits, strides, overlaps,
                                            add_to_movie=add_to_movie, template=old_templ, max_shifts=max_shifts,
                                            max_deviation_rigid=max_deviation_rigid,
                                            newoverlaps=newoverlaps, newstrides=newstrides,
                                            upsample_factor_grid=upsample_factor_grid, order='F', dview=dview,
                                            num_splits=num_splits_to_process, shifts_opencv=shifts_opencv, 
                                            nonneg_movie=nonneg_movie, gSig_filt=gSig_filt,
                                            use_cuda=use_cuda, border_nan=border_nan, var_name_hdf5=var_name_hdf5, is3D=is3D,
                                            indices=indices)

        new_templ = np.nanmedian(np.dstack([r[-2] for r in res_el]), -1)
        if gSig_filt is not None:
            new_templ = high_pass_filter_space(new_templ, gSig_filt)

    total_template = new_templ
    templates = []
    x_shifts = []
    y_shifts = []
    z_shifts = []
    coord_shifts = []
    mc = []
    for rr in res_el:
        shift_info_chunk, idxs_chunk, tmpl_chunk, mc_cur = rr
        templates.append(tmpl_chunk)
        for i in range(len(shift_info_chunk)):
            shift_info = shift_info_chunk[i]
            mc.append(mc_cur)
            if is3D:
                total_shift, _, xyz_grid = shift_info
                
                x_shifts.append(np.array([sh[0] for sh in total_shift]))
                y_shifts.append(np.array([sh[1] for sh in total_shift]))
                z_shifts.append(np.array([sh[2] for sh in total_shift]))
                coord_shifts.append(xyz_grid)
            else:
                total_shift, _, xy_grid = shift_info
                x_shifts.append(np.array([sh[0] for sh in total_shift]))
                y_shifts.append(np.array([sh[1] for sh in total_shift]))
                coord_shifts.append(xy_grid)
        
    return total_template, templates, x_shifts, y_shifts, z_shifts, coord_shifts, mc


#%% in parallel
def tile_and_correct_wrapper(params):
    """Does motion correction on specified image frames

    Returns:
    shift_info:
    idxs:
    mean_img: mean over all frames of corrected image

    """
    # todo todocument


    try:
        cv2.setNumThreads(0)
    except:
        pass  # 'Open CV is naturally single threaded'

    imgs, idxs, shape_mov, template, strides, overlaps, max_shifts,\
        add_to_movie, max_deviation_rigid, upsample_factor_grid, newoverlaps, newstrides, \
        shifts_opencv, nonneg_movie, gSig_filt, is_fiji, use_cuda, border_nan, var_name_hdf5, \
        is3D, indices = params


    shift_info = []
    
    mc = np.zeros(imgs.shape, dtype=np.float32)
    if not imgs[0].shape == template.shape:
        template = template[indices]
    for count, img in enumerate(imgs):
        if count % 10 == 0:
            logging.debug(count)
        if is3D:
            mc[count], total_shift, start_step, xyz_grid = tile_and_correct_3d(img, template, strides, overlaps, max_shifts,
                                                                       add_to_movie=add_to_movie, newoverlaps=newoverlaps,
                                                                       newstrides=newstrides,
                                                                       upsample_factor_grid=upsample_factor_grid,
                                                                       upsample_factor_fft=10, show_movie=False,
                                                                       max_deviation_rigid=max_deviation_rigid,
                                                                       shifts_opencv=shifts_opencv, gSig_filt=gSig_filt,
                                                                       use_cuda=use_cuda, border_nan=border_nan)
            shift_info.append([total_shift, start_step, xyz_grid])
            
        else:
            mc[count], total_shift, start_step, xy_grid = tile_and_correct(img, template, strides, overlaps, max_shifts,
                                                                       add_to_movie=add_to_movie, newoverlaps=newoverlaps,
                                                                       newstrides=newstrides,
                                                                       upsample_factor_grid=upsample_factor_grid,
                                                                       upsample_factor_fft=10, show_movie=False,
                                                                       max_deviation_rigid=max_deviation_rigid,
                                                                       shifts_opencv=shifts_opencv, gSig_filt=gSig_filt,
                                                                       use_cuda=use_cuda, border_nan=border_nan)
            shift_info.append([total_shift, start_step, xy_grid])
        
    new_temp = np.nanmean(mc, 0)
    new_temp[np.isnan(new_temp)] = np.nanmin(new_temp)
    
    return shift_info, idxs, new_temp, np.array(mc).transpose([1,2,3,0])

def motion_correction_piecewise(video, splits, strides, overlaps, add_to_movie=0, template=None,
                                max_shifts=(12, 12), max_deviation_rigid=3, newoverlaps=None, newstrides=None,
                                upsample_factor_grid=4, order='F', dview=None, 
                                subidx = None, num_splits=None, shifts_opencv=False, nonneg_movie=False, gSig_filt=None,
                                use_cuda=False, border_nan=True, var_name_hdf5='mov', is3D=False,
                                indices=(slice(None), slice(None))):
    """

    """
    # todo todocument
    is_fiji = False
    
    
    dims = video.shape[:3]
    T = video.shape[-1]

    
    z = np.zeros(dims)
    dims = z[indices].shape
    logging.debug('Number of Splits: {}'.format(splits))
    if type(splits) is int:
        if subidx is None:
            rng = range(T)
        else:
            rng = range(T)[subidx]

        idxs = np.array_split(list(rng), splits)

    else:
        idxs = splits
        
    if template is None:
        raise Exception('Not implemented')
    
    shape_mov = (np.prod(dims), T)
    if num_splits is not None:
        idxs = np.array(idxs)[np.random.randint(0, len(idxs), num_splits)]
        #logging.warning('**** MOVIE NOT SAVED BECAUSE num_splits is not None ****')

    pars = []
    for idx in idxs:
        logging.debug('Processing: frames: {}'.format(idx))
        pars.append([video, idx, shape_mov, template, strides, overlaps, max_shifts, np.array(
            add_to_movie, dtype=np.float32), max_deviation_rigid, upsample_factor_grid,
            newoverlaps, newstrides, shifts_opencv, nonneg_movie, gSig_filt, is_fiji,
            use_cuda, border_nan, var_name_hdf5, is3D, indices])
    
    if dview is not None:
        logging.info('** Starting parallel motion correction **')
        if 'multiprocessing' in str(type(dview)):
            res = dview.map_async(tile_and_correct_wrapper, pars).get(4294967)
        else:
            res = dview.map_sync(tile_and_correct_wrapper, pars)
        logging.info('** Finished parallel motion correction **')
    else:
        res = list(map(tile_and_correct_wrapper, pars))
    
    
    return res