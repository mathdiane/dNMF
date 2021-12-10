# -*- coding: utf-8 -*-
"""
Created on Thu May  6 14:38:02 2021

@author: Amin
"""
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
#suppress the warning: *c* argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with *x* & *y*.  Please use the *color* keyword-argument or provide a 2-D array with a single row if you intend to specify the same RGB or RGBA value for all points.
#when running visualize_trajectory fx

# %%

def visualize_image(img):
    plt.imshow(img)
    plt.show()

def visualize_images(ims,titles,save=False,file=None):
    plt.figure(figsize=(10,5))
    for i in range(len(ims)):
        plt.subplot(1,len(ims),i+1)    
    
        plt.imshow(ims[i])
        plt.title(titles[i])
        
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()

def visualize_video(video=None,tracks=None,u_colors=True,labels=None,scale=1,fontsize=20,save=False,file=None):
    if tracks is not None and u_colors:
        colors = plt.cm.hsv(np.linspace(0,1,tracks.shape[0]+1)[0:-1])[:,0:3]
    else:
        colors = 'k'
            
    fig, ax = plt.subplots(figsize=(10,10))
    
    if video is not None:
        im = ax.imshow(video[:,:,:,0].max(2).squeeze())
    if tracks is not None:
        sc = ax.scatter(tracks[:,1,0], tracks[:,0,0],marker='x',color=colors)
        ax.set_aspect('equal', adjustable='box')
    
    time_text = fig.text(0.5, 0.03,'Frame = 0',horizontalalignment='center',
                         verticalalignment='top',fontsize=fontsize)
    
    ax.axis('off')
    scalebar = ScaleBar(scale,'um')
    ax.add_artist(scalebar)
    
    ax.set_title('Neural Centers', fontsize=fontsize)
    
    if labels is not None:
        annot = []
        for i, txt in enumerate(labels):
            annot.append(ax.text(tracks[i,1,0], tracks[i,0,0],txt,color=colors[i,:],fontsize=8))

    def init():
        if tracks is not None:
            sc.set_offsets(tracks[:,[1,0],0])
            ret = sc
        if video is not None:
            im.set_data(video[:,:,:,0].max(2).squeeze())
            ret = im
        
        return (ret,)
    
    
    def animate(t):
        if tracks is not None:
            P_t = tracks[:,:,t]
            sc.set_offsets(P_t[:,[1,0]])
            ret = sc
        
        if video is not None:
            data_slice = video[:,:,:,t].max(2).squeeze()
            im.set_data(data_slice)
            ret = im
        
        if labels is not None:
            for i in range(len(labels)):
                annot[i].set_x(P_t[i,1])
                annot[i].set_y(P_t[i,0])
                
        time_text.set_text('Frame = ' + str(t))
        
        return (ret,)
    
    if video is not None:
        T = video.shape[3]
    elif tracks is not None:
        T = tracks.shape[2]
        
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                   frames=T, interval=500, blit=True)
    if save is True:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(file, writer=writer)
        plt.close('all')
    else:
        plt.show()
    
    
def visualize_temporal(x,titlestr='',fontsize=12,linewidth=2,save=False,file=None):
    plt.figure(figsize=(5,x.shape[0]))
    colors = plt.cm.hsv(np.linspace(0,1,len(x)+1)[0:-1])[:,0:3]

    offset = np.append(0.0, np.nanmax(x[0:-1,:],1)-np.nanmin(x[0:-1,:],1))
    s = (x-np.nanmin(x,1)[:,None]+np.cumsum(offset)[:,None])
    for i in range(len(s)):
        plt.plot(s[i],linewidth=linewidth,color=colors[i])
        
    plt.yticks(s[:,0],[str(signal_idx) for signal_idx in range(s.shape[0])])
        
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()
        
def visualize_spatial(A,save=False,file=None,RGB=True):
    colors = plt.cm.hsv(np.linspace(0,1,A.shape[2]+1)[0:-1])[:,0:3]
    
    if RGB:
        plt.figure(figsize=(5,5))
        colored = np.einsum('mnk,ks->mns',A,colors)
        plt.imshow(2*colored/colored.max())
    else:
        m = int(np.sqrt(A.shape[2]))
        n = np.ceil(A.shape[2]/m)
        plt.figure(figsize=(3*n,3*m))
        for i in range(A.shape[2]):
            plt.subplot(m,n,i+1)
            colored = np.einsum('mnk,ks->mns',A[:,:,i][:,:,None],colors[i,:][None,:])
            plt.imshow(2*colored/colored.max())    
            plt.axis('off')
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()
        
def visualize_trajectory(P1,P2,save=False,file=None,flip_axis=True):
  #Input:
  #flip_axis: to flip the axis for plotting such that the dimension matches the video dimensions
    print("flip_axis=",flip_axis)
    plt.figure(figsize=(10,10))
    
    colors = plt.cm.hsv(np.linspace(0,1,P1.shape[0]+1)[0:-1])[:,0:3]
    for k in range(P1.shape[0]):
      color_k = colors[k,:]
      if flip_axis==True:
        pos = P1[k,:,:].squeeze()
        plt.scatter(pos[1,0],pos[0,0],c=color_k)
        plt.plot(pos[1,:], pos[0,:],c=color_k)

        pos = P2[k,:,:].squeeze()
        plt.scatter(pos[1,0],pos[0,0],c=color_k,marker='x')
        plt.plot(pos[1,:], pos[0,:],c=color_k,linestyle='--')
      else:
        pos = P1[k,:,:].squeeze()
        plt.scatter(pos[0,0],pos[1,0],c=color_k)
        plt.plot(pos[0,:], pos[1,:],c=color_k)

        pos = P2[k,:,:].squeeze()
        plt.scatter(pos[0,0],pos[1,0],c=color_k,marker='x')
        plt.plot(pos[0,:], pos[1,:],c=color_k,linestyle='--')

    plt.grid()
    if flip_axis==True:
        plt.gca().invert_yaxis() #flip y axis

    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()
    
