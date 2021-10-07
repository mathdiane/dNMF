# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 14:03:18 2021

@author: Amin
"""

import numpy as np
from scipy.optimize import nnls

@staticmethod
def histogram_match(a,b,nbins,type):
    """Takes as input two traces a (moving), b (reference) and outputs 
        normalized trace atransform that has a similar histogram as b
        Args:
            a (numpy.array): moving time series
            b (numpy.array): reference time series
            nbins (integer): number of bins to discretize both time series into
        Returns: 
            atransform (numpy.array): moved time series
            distance (double): histogram distance between atransform and b
    """
    
    #  discarding nans from time series
    a_nan_idx = ~np.isnan(a)
    b_nan_idx = ~np.isnan(b)
    a = a[a_nan_idx]
    b = b[b_nan_idx]
    
    Y = np.linspace(0,1,nbins)
    
    #  discretizing time series using quantiles
    abins = np.quantile(a,Y).T
    bbins = np.quantile(b,Y).T
    
    #  weighted linear regression of the matching quantiles
    if type == 'non-negative':
        beta = nnls(np.concatenate([abins,np.ones((abins.shape[0],1))],0),bbins)
    elif type == 'regular':
        beta = np.linalg.solve(np.concatenate([abins,np.ones((abins.shape[0],1))],0),bbins)
    
    # transformed time series with nan's put back in
    atransform = np.zeros(a_nan_idx.shape)*np.nan
    atransform[a_nan_idx] = a*beta[0] + beta[1]
    
    distance = np.nan
    
    return atransform,distance



def cleanTraces(traces, fps, sigma_threshold=10, detrend_mode=2, interp_method=[],
                smooth_method=[], smooth_window=[],):
    """
    CLEANTRACES Clean up the neural traces by removing outliers & bleaching,
                then scaling the traces to [0,1].
    
       Args:
           traces   - the neural traces to clean
           fps      - frames/second
           sigma    - remove outliers wherein a single frame is sigma standard
                      deviations from the mean ([] = don't remove outliers).
                      default: 10
           debleach - detrend the traces to remove bleaching
                      default: 2
                      0 = leave the data as is (don't detrend)
                      1 = detrend the data by computing a global bleach curve
                      2 = detrend the data by computing individual bleach curves
                      3 = detrend the data by computing individual bleach curves
                          + compute dF/F0 using F0 = 5th percentile.
           interp   - interpolate missing data with the specified method
                      (see interp1)
                      default: [] (no interpolation)
           smooth   - smooth the traces with the specified method
                      (see smoothdata - you MUST specify the window size as well)
                      causal = causal bandpass
                      high = high pass
                      low = low pass
                      default: [] (no smoothing)
           window   - the window size to use for filtering
                      default: [] (no smoothing)
        
       Returns:
           traces   - the scaled neural traces, cleaned up
           scales   - the scales used to normalize to unit traces
           offsets  - the offsets used to normalize to unit traces
   """
    
    # Clean up the data by removing the first & last frames (outliers).
    traces[:,1:round(fps/2)] = np.nan
    traces[:,end] = np.nan
    traces[traces <= 0.01] = np.nan
    
    # Dampen extreme outliers: a single frame wherein the signal change is
    # larger than sigma_threshold standard deviations from the mean.
    if len(sigma_threshold) > 0 and abs(sigma_threshold) > 0:
        
        # Compute the thresholds for ouliers.
        extreme_threshold = sigma_threshold * np.nanstd(traces, 0, 2) + \
            np.nanmean(traces, 2)
        diff_traces = np.diff(traces, 1, 2)
        
        # Find the outliers.
        extreme_max_traces = diff_traces > extreme_threshold
        extreme_min_traces = diff_traces < -extreme_threshold
        [iExtreme_neurons, iExtreme_frames] = np.where(\
            (extreme_max_traces(:,1:(end-1)) & extreme_min_traces(:,2:end)) ...
            | (extreme_min_traces(:,1:(end-1)) & extreme_max_traces(:,2:end)));
        iExtreme_frames = iExtreme_frames + 1;
        
         Remove the outliers.
        for i = 1:length(iExtreme_neurons)
            traces(iExtreme_neurons(i),iExtreme_frames(i)) = nan;
        end
        
         Median filter nearest neighbors..
        traces = medfilt1(traces, 3, [], 2);
    end
    
     Initialize the trace offsets & scaling.
    offsets = zeros(size(traces,1),1);
    detrend_offsets = zeros(size(traces,1),1);
    scales = ones(size(traces,1),1);
    
     Detrend the traces to remove bleaching.
    x = 1:size(traces,2);
    if detrend_mode > 0
    
         Determine F0 for the traces.
         Note: this must go here, after de-bleaching you may invert traces
         if F0 < 0;
        traces_nan = traces;
        traces_nan(traces_nan <= 0.1) = nan;
        F0 = prctile(traces_nan, 5, 2);
        
         Determine the filter order for median filtering (10s).
        filt_order = round(10 * fps);
        
         Do we have enough data to compute the bleach curve?
        detrend_threshold = 0.1 * size(traces,2);  10 of the data
    
         Assume a uniform bleach curve & detrend.
        if detrend_mode == 1
            
             Scale the traces.
            offsets = nanmin(traces, [], 2);
            traces = traces - offsets;
            scales = nanmax(traces, [], 2);
            traces = traces ./ scales;
            
             Compute the global bleach curve.
            y = nanmean(traces, 1);
            y_filt = medfilt1(y, filt_order, 'omitnan');
            y_filt_data = ~isnan(y_filt);
            y_data = ~isnan(y);
            if sum(y_data) > detrend_threshold
                [p, ~, mu] = polyfit(x(y_data), y(y_data), 2);
                f_y = polyval(p, x, [] ,mu);
                f = fit(x(y_filt_data)', y_filt(y_filt_data)', 'exp1');
    
                 Detrend the traces to remove bleaching.
                if f.b < 0  bleach curves must decay
                    f_y = f.a * exp(f.b * x);
                    detrend_offsets(:) = f.a;
                    traces = traces - f_y;
                end
            end
            
         Compute individual bleach curves, per neuron, & detrend.
        else
            
             Compute individual bleach curves, per neuron, & detrend.
            for i = 1:size(traces,1)
                yi = traces(i,:);
                yi_filt = medfilt1(yi, filt_order, 'omitnan');
                yi_filt_data = ~isnan(yi_filt);
                
                 Do we have enough data to use a local bleach curve?
                if sum(yi_filt_data) > detrend_threshold
                    fi = fit(x(yi_filt_data)', yi_filt(yi_filt_data)', 'exp1');
                    
                     Detrend the traces to remove bleaching.
                    if fi.b < 0  bleach curves must decay
                        fi_yi = fi.a * exp(fi.b * x);
                        detrend_offsets(i) = fi.a;
                        traces(i,:) = (yi - fi_yi);
                    end
                end
            end
        end
        
         Compute dF/F0 using F0 = 5th percentile.
        if detrend_mode == 3
            F0 = zeros(size(traces,1),1) + nanmedian(F0);  use the median F0
            scales = F0;
            scales(scales < 1) = 1;  dNMF ~ [0,1]
            offsets = zeros(size(F0));
            traces = (traces - offsets) ./ scales;
        end
    end
    
     Interpolate missing data.
    if ~isempty(interp_method)
        for i = 1:size(traces,1)
            
             Find the missing data.
            nan_data = isnan(traces(i,:));
            
             Is there any data to work with?
            if sum(nan_data) < size(traces,2)
                
                 Interpolate the missing data.
                y_data = ~nan_data;
                traces(i,nan_data) = interp1(x(y_data), traces(i,y_data), ...
                    x(nan_data), interp_method);
            end
        end
    end
    
     Smooth the data.
    if ~isempty(smooth_method) && ~isempty(smooth_window)
        
         Causal bandpass filter.
        if strcmpi(smooth_method, 'causal')
            traces = causalBandpassFilter(traces, ...
                smooth_window(1), smooth_window(2), smooth_window(3));
    
         Highpass filter.
        elseif strcmpi(smooth_method, 'high')
            traces = highpassFilter(traces, smooth_window(1), smooth_window(2));
            
         Lowpass filter.
        elseif strcmpi(smooth_method, 'low')
            traces = lowpassFilter(traces, smooth_window(1), smooth_window(2));
            
         Standard filter.
        else
            traces = smoothdata(traces, 2, smooth_method, smooth_window, ...
                'includenan');
        end
    end
    
     Re-scale the traces.
    if detrend_mode < 3
        new_offsets = nanmin(traces, [], 2);
        traces = traces - new_offsets;
        new_scales = nanmax(traces, [], 2);
        traces = traces ./ new_scales;
        
         Compute the compounded scales & offsets.
        offsets = offsets + (detrend_offsets + new_offsets) .* scales;
        scales = scales .* new_scales;
        
         Rescale the traces to [0.05, 0.95].
        traces = traces * 0.9 + 0.05;
    end
    end
