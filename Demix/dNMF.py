# -*- coding: utf-8 -*-

from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.animation as animation
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.io import savemat
import torch.optim as optim
from ..WUtils import Utils
import numpy as np
import torch

# %%

class dNMF:
    """Deformable Non-negative Matrix Factorization for joint tracking and 
        demixing of microscopy videos
        https://www.biorxiv.org/content/10.1101/2020.07.07.192120v1.abstract
    """
    eps = 1e-32
    
    default_params = {'positions':None,
                      'scale':torch.tensor([1,1,1]).float(),
                      'radius':3,
                      'step_S':0.,
                      'gamma':0.,
                      'use_gpu':False,
                      'initial_p':None,
                      'sigma_inv':2,
                      'method':'1->t',
                      'verbose':True}
    
    def online_optimization(self, video, times, params={}):
        default_params = dNMF.default_params.copy()
        default_params.update(self.params.copy())
        
        if not hasattr(self, 'times'):
            self.times = np.arange(0,self.B.shape[2])
            
        positions = torch.zeros((self.A.shape[0], 3, times.max()+1))
        if hasattr(self, 'positions'):
            positions[:,:,:self.positions.shape[2]] = self.positions.clone()
            positions[:,:,self.times] = self.get_positions()
        else:
            positions[:,:,:self.B.shape[2]] = self.get_positions()
            
        if 'initial_p' not in params:
            default_params['initial_p'] = self.get_positions()[:,:,self.times == times.min()].squeeze()
        
        if 'C' not in params:
            default_params['C'] = self.C[:,self.times == times.min()].repeat((1,times.max()-times.min()+1)).clone()
                    
        default_params.update(params.copy())
        
        self.A,self.B,self.C,self.Y,_,_,_, \
        _,_,_,_,_,self.BG,self.P, \
        self.RE,self.verbose,self.LL = dNMF.initialize(video,default_params)
        
        self.positions = positions.clone()
        self.times = times.copy()
        
        
    @staticmethod
    def initialize(video, params):
        if params['use_gpu'] and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            
        if 'C' in params.keys():
            C = torch.tensor(params['C']).to(device=device)
        else:
            K = params['initial_p'].shape[0]
            T = video.shape[3]
            C = torch.rand(K,T)
            C = C.to(device=device)
            
        if 'B' in params.keys():
            B = torch.tensor(params['B']).detach().requires_grad_(True).to(device=device)
        else:
            T = video.shape[3]
            B_init = torch.zeros((10,3,T))
            for t in range(T):
                B_init[:,:,t] = torch.cat((torch.zeros(1,3), torch.eye(3), torch.zeros(6,3)),0)
        
            B = B_init.clone()
            B = B.detach().requires_grad_(True).to(device=device)
            
        if 'A' in params.keys():
            A = torch.tensor(params['A']).to(device=device)
        else:
            A = params['initial_p'].clone()
            A = A.detach().to(device=device)
            
        radius = torch.tensor(params['radius']).to(device=device)
        step_S = torch.tensor(params['step_S']).to(device=device)
        gamma = torch.tensor(params['gamma']).to(device=device)
            
        
        scale = torch.tensor(params['scale']).float().to(device=device).squeeze() # microns per pixel
            
        if 'LL' in params.keys():
            LL = torch.tensor(params['LL']).float().to(device=device)
        else:
            K = params['initial_p'].shape[0]
            LL_init = torch.zeros((K))
            for k in range(K):
                LL_init[k] = params['sigma_inv']
            LL = LL_init.clone().detach().requires_grad_(True).to(device=device)
            
        if 'method' in params.keys():
            method = params['method']
        else:
            method = method
        
        if 'BG' in params.keys():
            BG = torch.tensor(params['BG']).to(device=device)
        else:
            BG = torch.tensor(np.percentile(video,60)).to(device=device)
        
        if 'cost' in params.keys():
            cost = params['cost']
        else:
            cost = []
            
        if 'pseudo_colors' in params.keys():
            pseudo_colors = torch.tensor(params['pseudo_colors']).float().to(device=device)
        else:
            R = np.linspace(0,1,C.shape[0]+1)[0:-1]
            color = plt.cm.hsv(R)[:,0:3]
            np.random.shuffle(color)
            
            pseudo_colors = torch.tensor(color).float().to(device=device)
        
        if params['positions'] is None:
            positions = A[:,:,np.newaxis]
        else:
            positions = params['positions']
            
        if len(positions.shape) == 2:
            positions = positions.copy()[:,:,np.newaxis]
            
        sz = video.shape
        pos = torch.tensor(np.array(np.where(np.ones(sz[0:3]))).T)
        a = positions.permute([1,0,2]).reshape(3,int(positions.numel()/3)).t()
        a = a[~(a > torch.tensor(video.shape[:3])[np.newaxis,:]).any(1),:]
        a = a[~(a < torch.tensor([0,0,0])[np.newaxis,:]).any(1),:]
        ind = np.ravel_multi_index(a.numpy().T.astype(int),np.array(video.shape[0:3]).astype(int))
        mask = np.zeros(video.shape[0:3])
        mask.ravel()[ind] = 1
        
        filt,_ = dNMF.create_spherical_filter([0,0,0],radius,scale.squeeze())
        maskc = convolve(mask,filt)
        maskc[maskc>0] = 1
        
        keep_indices = np.where(maskc.flatten()>0)
        P = pos[keep_indices,:].squeeze().float()
        Y = video.reshape([int(float(video.numel())/video.shape[3]), video.shape[3]]).squeeze()
        Y = Y[keep_indices,:].squeeze()
        
        
        P = P.to(device=device)
        Y = Y.to(device=device)
        RE = ((Y.clone())*0).to(device=device)
        verbose = params['verbose']
        
        return A,B,C,Y,radius,scale,pseudo_colors,method,cost,device,gamma,step_S,BG,P,RE,verbose,LL
        
    def __init__(self,video,params={}):
        default_params = dNMF.default_params.copy()
        default_params.update(params.copy())
        
        self.params = params.copy()
        
        self.A,self.B,self.C,self.Y,self.radius,self.scale,self.pseudo_colors, \
        self.method,self.cost,self.device,self.gamma,self.step_S,self.BG,self.P, \
        self.RE,self.verbose,self.LL = dNMF.initialize(video,default_params)
    
    def optimize(self,lr=.1,n_iter=100,n_iter_c=20,sample_size=None):
        optimizer = optim.Adam({self.B,self.LL}, lr=lr)
        
        if sample_size is None:
            y_ind = np.arange(self.Y.shape[0])
            t_ind = np.arange(self.Y.shape[1])
                
        for iter in range(n_iter):
            if self.verbose:
                print(self.LL.mean())
                print('Update C:')
            
            
            if sample_size is not None:
                y_ind = np.random.randint(0,self.Y.shape[0],sample_size[0])
                t_ind = np.random.randint(0,self.Y.shape[1],sample_size[1])
                t_ind.sort()
            
            self.update_C(t_ind,y_ind)
            
            
            if self.verbose:
                print(self.C.mean())
                print('Update A:')
                
            L = self.cost_A(t_ind,y_ind)
            
            if self.verbose:
                print(self.B.mean())
                print(L)
            
            
            if self.verbose:
                print('Backward:')
            
            self.cost.append(L.cpu().detach().numpy())
            def closure():
                optimizer.zero_grad()
                L.backward(retain_graph=True)
                return L
            
            
            optimizer.step(closure)
            
            if self.verbose:
                print(self.B.grad.mean())
                print('Iter: ' + str(iter) + ' - Cost: ' + str(L))
        
        for inner_iter in range(n_iter_c):
            if sample_size is not None:
                y_ind = np.random.randint(0,self.Y.shape[0],sample_size[0])
                t_ind = np.random.randint(0,self.Y.shape[1],sample_size[1])
                t_ind.sort()
            
            if self.verbose:
                print('Inner Iter: ' + str(inner_iter)) 
                
            self.update_C(t_ind,y_ind)
    

    def update_C(self,times,indices):
        with torch.no_grad():
            R = self.Y[indices,:].clone()
            
            A = self.A
            for t in times:
                if self.method == '1->t':
                    A_p = torch.cat((torch.ones((len(self.A),1)),self.A,self.A*self.A,(self.A[:,0]*self.A[:,1]).unsqueeze(1),\
                             (self.A[:,0]*self.A[:,2]).unsqueeze(1),(self.A[:,1]*self.A[:,2]).unsqueeze(1)), 1).to(device=self.device)
                elif self.method == 't-1->t':
                    A_p = torch.cat((torch.ones((len(A),1)),A,A*A,(A[:,0]*A[:,1]).unsqueeze(1),\
                             (A[:,0]*A[:,2]).unsqueeze(1),(A[:,1]*A[:,2]).unsqueeze(1)), 1).to(device=self.device)
            
                
                A = A_p@self.B[:,:,t]
                
                if t==0 and self.Y.shape[1] > 1:
                    S = self.C[:,t+1]; factor=1
                elif t==self.Y.shape[1]-1:
                    S = self.C[:,t-1]; factor=1
                else:
                    S = self.C[:,t-1]+self.C[:,t+1]; factor=2
                    
                a_ks = torch.exp(-0.5*dNMF.pairwise_distances(self.P[indices,:]*self.scale, A*self.scale)*self.LL)
                R[:,t] = R[:,t] - a_ks@self.C[:,t]
                data_term = self.Y[indices,t]-self.BG # Y
                data_term[data_term < 0] = 0
                self.C[:,t] = self.C[:,t].clone()*(a_ks.t()@(data_term)+self.step_S*S+dNMF.eps)/(a_ks.t()@a_ks@self.C[:,t].clone()+self.step_S*factor*self.C[:,t].clone()+dNMF.eps)
                
            R[R < 0] = 0
            self.BG = R.mean()
            
    def cost_A(self,times,indices):
        cost = 0
        
        A = self.A
        for t in times:
            if self.method == '1->t':
                A_p = torch.cat((torch.ones((len(self.A),1)),self.A,self.A*self.A,(self.A[:,0]*self.A[:,1]).unsqueeze(1),\
                         (self.A[:,0]*self.A[:,2]).unsqueeze(1),(self.A[:,1]*self.A[:,2]).unsqueeze(1)), 1).to(device=self.device)
            elif self.method == 't-1->t':
                A_p = torch.cat((torch.ones((len(A),1)),A,A*A,(A[:,0]*A[:,1]).unsqueeze(1),\
                         (A[:,0]*A[:,2]).unsqueeze(1),(A[:,1]*A[:,2]).unsqueeze(1)), 1).to(device=self.device)
            
            A = A_p@self.B[:,:,t]
            
            cost = cost + self.gamma*(dNMF.det_jac(self.B[:,:,t],self.P.min(0)[0])-1)**2 \
                        + self.gamma*(dNMF.det_jac(self.B[:,:,t],self.P.min(0)[0])-1)**2
            
            if t > 1:
                cost = cost + self.gamma*((A_p@self.B[:,:,t] - A_p@self.B[:,:,t-1])**2).mean()
            
            a_ks = torch.exp(-0.5*dNMF.pairwise_distances(self.P[indices,:]*self.scale, A*self.scale)*self.LL)
            
            x = a_ks@self.C[:,t]+self.BG
            y = self.Y[indices,t]
            
            vx = x - torch.mean(x)
            vy = y - torch.mean(y)
            
            corrcoeff = torch.sum(vx*vy)/(torch.sqrt(torch.sum(vx**2))*torch.sqrt(torch.sum(vy**2)))
            cost = cost + (1-corrcoeff)

#            cost = cost + ((a_ks@self.C[:,t]-self.Y[indices,t]+self.BG)**2).mean()
                    
        
        return cost
    
    
    def get_positions(self):
        positions = torch.zeros((self.A.shape[0],3,self.Y.shape[1]))
        positions[:,:,0] = self.A
        for t in range(1,self.Y.shape[1]):
            if self.method == '1->t':
                A_p = torch.cat((torch.ones((len(self.A),1)),self.A,self.A*self.A,(self.A[:,0]*self.A[:,1]).unsqueeze(1),\
                         (self.A[:,0]*self.A[:,2]).unsqueeze(1),(self.A[:,1]*self.A[:,2]).unsqueeze(1)), 1).to(device=self.device)
            elif self.method == 't-1->t':
                A = positions[:,:,t-1]
                A_p = torch.cat((torch.ones((len(A),1)),A,A*A,(A[:,0]*A[:,1]).unsqueeze(1),\
                         (A[:,0]*A[:,2]).unsqueeze(1),(A[:,1]*A[:,2]).unsqueeze(1)), 1).to(device=self.device)
            
            positions[:,:,t] = A_p@self.B[:,:,t]
        
        return positions.detach()
    
    # %% Helper functions
    @staticmethod
    def pairwise_distances(x,y):
        x_norm = (x**2).sum(1).view(-1, 1)
        y_norm = (y**2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
        return dist
    
    @staticmethod
    def decimate_video(video,positions,dfactor,beta=None):
        dvideo = zoom(video,dfactor,order=1)
        
        if len(positions.shape) == 3:
            dpositions = positions.numpy()*np.array(dfactor[0:3])[np.newaxis,:,np.newaxis]
        elif len(positions.shape) == 2:
            dpositions = positions.numpy()*np.array(dfactor[0:3])[np.newaxis,:]
            
        if beta is not None:
            dscale = np.array(dfactor[0:3])
            bscale = dscale*np.repeat(np.hstack((1, 1/dscale, 1/(dscale*dscale), 1/(dscale[0]*dscale[1]), 1/(dscale[1]*dscale[2]), 1/(dscale[2]*dscale[0])))[:,np.newaxis],3,1)
            dbeta = beta*torch.tensor(bscale[:,:,np.newaxis]).float()
        
            return dvideo,dpositions,dbeta
        else:
            return dvideo,dpositions
    
    @staticmethod
    def det_jac(B,P):
        x,y,z = P[0],P[1],P[2]
        
#        jac = torch.tensor([[B[1,0]+2*B[4,0]*x+B[8,0]*y+B[9,0]*z, B[2,0]+2*B[5,0]*y+B[7,0]*x+B[8,0]*y, B[3,0]+2*B[6,0]*z+B[8,0]*y+B[9,0]*x ], 
#                            [B[1,1]+2*B[4,1]*x+B[8,1]*y+B[9,1]*z, B[2,1]+2*B[5,1]*y+B[7,1]*x+B[8,1]*y, B[3,1]+2*B[6,1]*z+B[8,1]*y+B[9,1]*x ],
#                            [B[1,2]+2*B[4,2]*x+B[8,2]*y+B[9,2]*z, B[2,2]+2*B[5,2]*y+B[7,2]*x+B[8,2]*y, B[3,2]+2*B[6,2]*z+B[8,2]*y+B[9,2]*x ]])
    
        a = B[1,0]+2*B[4,0]*x+B[7,0]*y+B[9,0]*z
        b = B[2,0]+2*B[5,0]*y+B[7,0]*x+B[8,0]*z
        c = B[3,0]+2*B[6,0]*z+B[8,0]*y+B[9,0]*x
        d = B[1,1]+2*B[4,1]*x+B[7,1]*y+B[9,1]*z
        e = B[2,1]+2*B[5,1]*y+B[7,1]*x+B[8,1]*z
        f = B[3,1]+2*B[6,1]*z+B[8,1]*y+B[9,1]*x
        g = B[1,2]+2*B[4,2]*x+B[7,2]*y+B[9,2]*z
        h = B[2,2]+2*B[5,2]*y+B[7,2]*x+B[8,2]*z
        i = B[3,2]+2*B[6,2]*z+B[8,2]*y+B[9,2]*x
        
        det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
        return det
    
    @staticmethod
    def create_spherical_filter(position,radius,scale=[1,1,1]):
        coords = [[x,y,z] for x in range(position[0]-int(radius//scale[0]),int(position[0]+radius//scale[0])) 
                          for y in range(position[1]-int(radius//scale[1]),int(position[1]+radius//scale[1]))
                          for z in range(position[2]-int(radius//scale[2]),int(position[2]+radius//scale[2]))
                          if x**2*scale[0]+y**2*scale[1]+z**2*scale[2]<radius**2]
        coords = np.array(coords)
        coords = coords-coords.min(0)
        filt = np.zeros(coords.max(0)+1)
        ind = np.ravel_multi_index(coords.T,coords.max(0)+1)
        filt.ravel()[ind] = 1
        return filt,np.array(coords)
    
    # %% Visualization
    def visualize_stats(self,file,save=True,fontsize=20):
        
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(self.cost)
        ax.set_xlabel('Iteration', fontsize=fontsize)
        ax.set_ylabel('Cost Value', fontsize=fontsize)
        
        if save:
            plt.savefig(file+'-cost.png')
            plt.close('all')
        else:
            plt.show()
            plt.close('all')
        
        
        fig, ax = plt.subplots(figsize=(10,10))
        c = self.C.cpu().detach()
        ax.imshow(c/c.max(1)[0].unsqueeze(1))
        ax.set_xlabel('Time Points', fontsize=fontsize)
        ax.set_ylabel('Neuron Index', fontsize=fontsize)
        
        if save:
            plt.savefig(file+'-C.png')
            plt.close('all')
        else:
            plt.show()
            plt.close('all')
        
    
    @staticmethod
    def visualize_traces(file,all_traces,neuron_names=None,save=True,colors=None,labels=None,fontsize=20):

        for trace_idx in range(len(all_traces)):
            c = all_traces[trace_idx]
            
            fig, ax = plt.subplots(figsize=(5,c.shape[0]/4))
            
            offset = np.append(0.0, c[0:-1,:].max(1))
            offset[offset < dNMF.eps] = dNMF.eps
            signals = (c+np.expand_dims(np.cumsum(offset,0),1)).T
            if colors is None:
                ax.plot(signals)
            else:
                for i in range(signals.shape[1]):
                    ax.plot(signals[:,i], color=colors[i,:])
                
            offset2 = c[:,1]
            c[offset2 < dNMF.eps] = dNMF.eps
            if neuron_names is not None:
                plt.yticks(np.cumsum(offset)+offset2,neuron_names)
                
            ax.set_xlabel('Time Points',fontsize=fontsize)
            ax.set_ylabel('Neurons',fontsize=fontsize)
            
            if save:
                if labels is None:
                    plt.savefig(file+'-method('+str(trace_idx)+').png')
                else:
                    plt.savefig(file+'-method('+labels[trace_idx]+').png')
                plt.close('all')

            else:
                plt.show()
                plt.close('all')

    
    
    def visualize_raw(self, file, video, fontsize=20):
        fig, ax = plt.subplots(figsize=(10,10))
        im = ax.imshow(video[:,:,:,0].max(2)[0].squeeze())
        
        time_text = fig.text(0.5, 0.03,'Frame = 0',horizontalalignment='center',verticalalignment='top',fontsize=fontsize)
        
        ax.axis('off')
        scalebar = ScaleBar(self.scale[0],'um')
        ax.add_artist(scalebar)
        
        def init():
            im.set_data(video[:,:,:,0].max(2)[0].squeeze())
            return (im,)
    
        def animate(t):
            data_slice = video[:,:,:,t].max(2)[0].squeeze()
            im.set_data(data_slice)
            
            time_text.set_text('Frame = ' + str(t))
            
            return (im,)
    
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=video.shape[3], interval=200, blit=True)
    
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(file+'-raw.mp4', writer=writer)
        
        plt.close('all')
        
        return anim
    
    def visualize_tracks(self, file, video, labels=None, positions=None, fontsize=20):
        fig, ax = plt.subplots(figsize=(10,10))
        im = ax.imshow(video[:,:,:,0].max(2)[0].squeeze())
        sc = ax.scatter(self.A[:,1], self.A[:,0],marker='x',color=self.pseudo_colors)
        
        time_text = fig.text(0.5, 0.03,'Frame = 0',horizontalalignment='center',verticalalignment='top',fontsize=fontsize)
        
        ax.axis('off')
        scalebar = ScaleBar(self.scale[0],'um')
        ax.add_artist(scalebar)
        
        ax.set_title('Neural Centers', fontsize=fontsize)
        
        if labels is not None:
            annot = []
            for i, txt in enumerate(labels):
                annot.append(ax.text(self.A[i,1], self.A[i,0],txt,color=self.pseudo_colors[i,:],fontsize=8))

        def init():
            im.set_data(video[:,:,:,0].max(2)[0].squeeze())
            return (im,)
        
        if positions is None:
            positions = self.get_positions()
        
        
        def animate(t):
            P_t = positions[:,:,t]
            data_slice = video[:,:,:,t].max(2)[0].squeeze()
            im.set_data(data_slice)
            sc.set_offsets(P_t[:,[1,0]])
            
            if labels is not None:
                for i in range(len(labels)):
                    annot[i].set_x(P_t[i,1])
                    annot[i].set_y(P_t[i,0])
            time_text.set_text('Frame = ' + str(t))
            
            return (im,)
    
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=video.shape[3], interval=200, blit=True)
    
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(file+'-tracks.mp4', writer=writer)
        
        plt.close('all')
        
        return anim
    
    def visualize_average_neuron_shapes(self, file, neurons_plot, neuron_names, rvideo, window=np.array([10,10,1])):
        
        neurons = [neurons_plot[i] for i in range(len(neurons_plot)) if neurons_plot[i] in neuron_names]
        indices = [neuron_names.index(neuron) for neuron in neurons]
        positions = self.get_positions().detach().numpy()
        fact = 5/rvideo.max()
        video = fact*rvideo.clone()
        for i, idx in enumerate(indices):
            avg = torch.zeros((2*window[0:2]+1).tolist())
            for t in range(positions.shape[2]):
                patch = Utils.subcube(video[:,:,:,t][:,:,:,np.newaxis], positions[idx,:,t], window).max(2)[0].squeeze()
                avg = avg + patch
            avg = avg/positions.shape[2]
            plt.imshow(avg,vmin=0,vmax=1)
            plt.axis('off')
            plt.scatter(window[1],window[0],color=self.pseudo_colors[idx])
            scalebar = ScaleBar(self.scale[0],'um')
            plt.gca().add_artist(scalebar)
            plt.title(neurons[i])
            plt.savefig(file+'-'+neurons[i]+'.png')
            plt.close('all')
            
            
            
    def visualize_neurons(self, file, neurons_plot, neuron_names, video, window=np.array([10,10,1]), fontsize=20, gt={}, mfact=5):
        neurons = [[neurons_plot[i][j] for j in range(len(neurons_plot[0])) if neurons_plot[i][j] in neuron_names] for i in range(len(neurons_plot))]
        indices = [[neuron_names.index(neurons[i][j]) for j in range(len(neurons[0]))] for i in range(len(neurons))]
            
        fact = mfact/video.max()
        video = fact*video
        
        min_val = -.2
        
        def clean_ax(ax,add_scalebar=True,clear_x=True,clear_y=True):
            
            if clear_x:
                ax.set_xticklabels([])
                ax.set_xticks([])
            if clear_y:
                ax.set_yticklabels([])    
                ax.set_yticks([])
            
            ax.autoscale(False)
            
            if add_scalebar:
                scalebar = ScaleBar(self.scale[0],'um')
                ax.add_artist(scalebar)
            
        
        colors = self.pseudo_colors
        
        positions = self.get_positions()
        
        if 'C' not in gt.keys() and 'positions' not in gt.keys():
            rows = 3
        elif 'positions' not in gt.keys():
            rows = 4
        else:
            rows = 5
        
        fig = plt.figure(figsize=(10,(10)*(2+rows*len(neurons))/len(neurons[0])),constrained_layout=True)
        gs = fig.add_gridspec(4+2*rows*len(neurons), 3*len(neurons[0]))
        
        ax_main = fig.add_subplot(gs[0:4, 0:int(len(neurons[0]))])
        ax_reco = fig.add_subplot(gs[0:4, int(len(neurons[0])):2*int(len(neurons[0])) ])
        ax_resi = fig.add_subplot(gs[0:4, 2*int(len(neurons[0])): ])
        
        time_text = fig.text(0.5, 1.0,'Frame = 0',horizontalalignment='center',verticalalignment='top',fontsize=fontsize)
    
        P = np.array([positions[ind,:,:].detach().numpy() for ind in indices])
        
        im = [[None for j in range(len(neurons[0]))] for i in range(rows*len(neurons))]
        ax = [[None for j in range(len(neurons[0]))] for i in range(rows*len(neurons))]
        
        data_slice = video[:,:,:,0].max(2)[0]
        im_main = ax_main.imshow(data_slice,vmin=min_val,vmax=1)
        
        s_main = ax_main.scatter(P[:,:,1,0],P[:,:,0,0],s=5,color='r',marker='x',linewidths=3)
        r_main = [[ax_main.text(P[i,j,1,0].flatten(), P[i,j,0,0].flatten(), neurons[i][j], color=self.pseudo_colors[indices[i][j]], fontsize=10) for j in range(len(neurons[0]))] for i in range(len(neurons))]
        
        
        im_reco = ax_reco.imshow(data_slice,vmin=min_val,vmax=1)
        im_resi = ax_resi.imshow(data_slice,vmin=min_val,vmax=1)
        
        
        ax_reco.set_title('',fontsize=fontsize)
        ax_main.set_title('Data',fontsize=fontsize)
        ax_resi.set_title('Residual',fontsize=fontsize)
        
        clean_ax(ax_reco,add_scalebar=True)
        clean_ax(ax_main,add_scalebar=False)
        clean_ax(ax_resi,add_scalebar=False)
        
        for i in range(rows*len(neurons)):
            for j in range(len(neurons[0])):
                ax[i][j] = fig.add_subplot(gs[2*i+4:2*i+6,3*j:3*j+3])
                if i%rows == rows-1 and 'C' in gt.keys():
                    k = indices[i//rows][j]
                    ax[i][j].plot(self.C[k,:],linewidth=2,c=self.pseudo_colors[k,:])
                    ax[i][j].plot(gt['C'][k,:],linewidth=2,c='k')
                    
#                    ax[i][j].xaxis.set_tick_params(labelsize=5)

                    ax[i][j].grid('on')
#                    if j == 0:
#                        ax[i][j].legend(['Inferred','GT'])
                    ax[i][j].set_xlabel('Frames $\pm$ 10')
                    im[i][j] = ax[i][j].axvline(x=0,color='r')
                    clean_ax(ax[i][j],add_scalebar=False)
                else:
                    im[i][j] = ax[i][j].imshow(np.zeros((window[0]*2+1,window[0]*2+1)),vmin=min_val,vmax=1)
                    clean_ax(ax[i][j],add_scalebar=(i==0 and j==0))
                
                if i%rows==0:
                    ax[i][j].set_title(neurons[i//rows][j],fontsize=fontsize)
        
        Ps = torch.tensor(np.array(np.where(np.ones(video.shape[0:3]))).T).float()
        pos = torch.tensor(np.array(np.where(np.ones(2*window+1))).T).float()
        
        def animate(t):
            if self.verbose:
                print('Time: '+ str(t))
            
#            data_slice = video[:,:,:,t].max(2)[0]
#            im_main.set_data(data_slice)
#            s_main.set_offsets(P[:,:,[1,0],t].reshape(-1,2))
#            
#            reco_mat = self.C[:,t]*torch.exp(-0.5*self.LL.detach()*dNMF.pairwise_distances(Ps*self.scale,positions[:,:,t]*self.scale))
#            a_k = (reco_mat@colors)
#            reco_slice = (fact*a_k.reshape((video.shape[0],video.shape[1],video.shape[2],3))).max(2)[0]
#            reco_slice[reco_slice < 0] = 0
#            reco_slice[reco_slice > 1] = 1
#            im_reco.set_data(reco_slice)
#            
#            
#            resi_slice = (video[:,:,:,t]-fact*reco_mat.sum(1).reshape(video.shape[0],video.shape[1],video.shape[2])).max(2)[0]
#            im_resi.set_data(resi_slice)
            
            for i in range(len(neurons)):
                for j in range(len(neurons[0])):
                    r_main[i][j].set_x(P[i,j,[1],t].flatten())
                    r_main[i][j].set_y(P[i,j,[0],t].flatten())
                    
            
            
            for i in range(rows*len(neurons)):
                for j in range(len(neurons[0])):
                    if i%rows==0:
                        patch = Utils.subcube(video[:,:,:,t][:,:,:,np.newaxis], P[i//rows,j,:,t], window).max(2)[0].squeeze()
                    elif i%rows==1:
                        k = indices[i//rows][j]
                        a_k = (self.C[:,t]*torch.exp(-0.5*self.LL.detach()*dNMF.pairwise_distances(pos*self.scale,(positions[:,:,t] - positions[k,:,t] + torch.tensor(window).float())*self.scale))@colors)
                        patch = (fact*a_k.reshape((2*window[0]+1,2*window[1]+1,2*window[2]+1,3))).max(2)[0]
                        
                        patch[patch < 0] = 0
                        patch[patch > 1] = 1
                    elif i%rows==2 and 'positions' in gt.keys():
                        k = indices[i//rows][j]
                        a_k = (gt['C'][:,t]*torch.exp(-0.5*gt['LL']*dNMF.pairwise_distances(pos*self.scale,(gt['positions'][:,:,t] - gt['positions'][k,:,t] + torch.tensor(window).float())*self.scale))@colors)
                        patch = (fact*a_k.reshape((2*window[0]+1,2*window[1]+1,2*window[2]+1,3))).max(2)[0]
                        
                        patch[patch < 0] = 0
                        patch[patch > 1] = 1
                    elif (rows==3 and i%rows==rows-1) or (rows>3 and i%rows==rows-2):
                        k = indices[i//rows][j]
                        a_k = (self.C[:,t]*torch.exp(-0.5*self.LL.detach()*dNMF.pairwise_distances(pos*self.scale,(positions[:,:,t] - positions[k,:,t] + torch.tensor(window).float())*self.scale))).sum(1)
                        patch_r = fact*a_k.reshape((2*window[0]+1,2*window[1]+1,2*window[2]+1,1))
                        patch_d = Utils.subcube(video[:,:,:,t][:,:,:,np.newaxis], P[i//rows,j,:,t], window)
                        patch = (patch_d - patch_r).max(2)[0].squeeze()
                        patch[patch < 0] = 0
                        patch[patch > 1] = 1
                        
                    elif i%rows==rows-1 and 'C' in gt.keys():
                        im[i][j].set_data([t,t],[0,1])
                        ax[i][j].set_xlim([max(t-10,0), min(t+10,video.shape[3])])
                    if (rows==3) or (rows>3 and i%rows!=rows-1):
                        im[i][j].set_data(patch)
                    
                    
            time_text.set_text('Frame = ' + str(t))
            return (im_main,)
    
        anim = animation.FuncAnimation(fig, animate, frames=video.shape[3], interval=200, blit=False)
        
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(file+'-neurons.mp4', writer=writer)
        plt.close('all')

    # %% Saving 
    def save_results(self, file, trace_array, neuron_names, sort_indices=[]):
        savemat(file + '-matlab.mat', \
                mdict={'all_traces': trace_array, 'C': self.C.cpu().detach().numpy(), \
                     'B': self.B.cpu().detach().numpy(), 'A': self.A.cpu().detach().numpy(),
                     'P': self.P.cpu().detach().numpy(), 'scale': self.scale.cpu().detach().numpy(),\
                     'cost': self.cost, 'radius': self.radius.cpu().detach().numpy(), \
                     'step_S': self.step_S.cpu().detach().numpy(), 'gamma': self.gamma.cpu().detach().numpy(), \
                     'neuron_names': neuron_names, 'sort_indices': sort_indices, 'BG':self.BG.cpu().detach().numpy(), \
                     'LL':self.LL.cpu().detach().numpy(), 'pseudo_colors':self.pseudo_colors.cpu().detach().numpy()})