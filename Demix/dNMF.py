# -*- coding: utf-8 -*-

from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.animation as animation
from scipy.spatial.distance import cdist
from Methods.Utils import Utils
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.io import savemat
import torch.optim as optim
import numpy as np
import torch

class dNMF:
    eps = 1e-32
    
    @staticmethod
    def optimize_sequential(video,initial_p,scale=torch.tensor([1,1,1]).float(),radius=3,step_S=.0,gamma=.0,use_gpu=False,\
                            sigma_inv=.1,lr=.1,b_iter=10,c_iter=2,n_iter=2,dscale=(1,1,1,1),batchsize=12,overlap=2,method='1->t'):
        T = video.shape[3]
        K = initial_p.shape[0]
        
        dvideo,dpositions = dNMF.decimate_video(video,initial_p,dscale)
        dvideo = torch.tensor(dvideo).float()
        dpositions = torch.tensor(dpositions).float()
        
        B = torch.cat((torch.zeros(1,3), torch.eye(3), torch.zeros(6,3)),0)[:,:,np.newaxis].repeat(1,1,T).float()
        C = torch.zeros((K,T)).float()
        A = dpositions
        P = torch.zeros((K,3,T)).float()
        P[:,:,0] = initial_p
#        B0 = torch.zeros((1,3,T))
        
        
        
        for t in range(0,T,batchsize-overlap):
            print('t = ' + str(t))
#            with torch.no_grad():
#                b,p = ot_registration.initialize_beta(P[:,:,t-1],1+0*C[:,t-1].double().detach().numpy(),dvideo[:,:,:,t],n_iter=20,mp_var=[5,5,5],mp_k=int(1.5*K),alpha=1)
#                B[:,:,t] = torch.tensor(b)
            
            s_t = t
            e_t = t+batchsize+1
            
            if e_t > T:
                e_t = T
                
            print('start: ' + str(s_t) + ', end: ' + str(e_t))
            dnmf = dNMF(dvideo[:,:,:,s_t:e_t],positions=P[:,:,s_t:e_t],radius=radius,step_S=step_S,\
                        gamma=gamma,use_gpu=use_gpu,initial_p=P[:,:,s_t],sigma_inv=sigma_inv,method=method)
            dnmf.B = B[:,:,s_t:e_t].requires_grad_(True)
            
            dnmf.update_C()
#            dnmf.C = C[:,s_t].repeat((dnmf.C.shape[1],1)).T
            
            dnmf.optimize(lr,n_iter=n_iter,n_iter_c=c_iter)
            
            C[:,  s_t:e_t] = dnmf.C.requires_grad_(False)
            B[:,:,s_t:e_t] = dnmf.B.detach().requires_grad_(False)
            P[:,:,s_t:e_t] = dnmf.get_positions().detach().requires_grad_(False)
#            B0[:,:,s_t:e_t] = dnmf.B0.detach().requires_grad_(False)
            
        dnmf = dNMF(dvideo,positions=P[:,:,0][:,:,np.newaxis],radius=radius,step_S=step_S,gamma=gamma,use_gpu=use_gpu,initial_p=A,sigma_inv=sigma_inv,method=method)
        dnmf.B = B
        dnmf.C = C
#        dnmf.B0 = B0
        
        return dnmf,dvideo
        
    
    def __init__(self,video,positions=None,scale=torch.tensor([1,1,1]).float(),radius=3,\
                 step_S=0.,gamma=0.,use_gpu=False,initial_p=None,sigma_inv=2,method='1->t',
                 params=None,verbose=True):
        
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        if params is not None:
            if 'C' in params.keys():
                self.C = torch.tensor(params['C']).to(device=self.device)
            if 'B' in params.keys():
                self.B = torch.tensor(params['B']).detach().requires_grad_(True).to(device=self.device)
            if 'A' in params.keys():
                self.A = torch.tensor(params['A']).to(device=self.device)
            if 'cost' in params.keys():
                self.cost = torch.tensor(params['cost']).to(device=self.device)
            if 'radius' in params.keys():
                self.radius = torch.tensor(params['radius']).to(device=self.device)
            if 'step_S' in params.keys():
                self.step_S = torch.tensor(params['step_S']).to(device=self.device)
            if 'gamma' in params.keys():
                self.gamma = torch.tensor(params['gamma']).to(device=self.device)
            if 'P' in params.keys():
                self.P = torch.tensor(params['P']).to(device=self.device)
            if 'scale' in params.keys():
                self.scale = torch.tensor(params['scale']).float().to(device=self.device) # microns per pixel
        
        
        
        if not hasattr(self, 'radius'):
            self.radius = torch.tensor(radius).float().to(device=self.device)
            
        if not hasattr(self, 'step_S'):
            self.step_S = torch.tensor(step_S).float().to(device=self.device)
        
        if not hasattr(self, 'gamma'):
            self.gamma = torch.tensor(gamma).float().to(device=self.device)
        
        if not hasattr(self, 'A'):
            A = initial_p.clone()
            self.A = A.detach().to(device=self.device)
        
        if positions is None:
            positions = self.A[:,:,np.newaxis]
            
        sz = video.shape
        pos = torch.tensor(np.array(np.where(np.ones(sz[0:3]))).T)
        a = positions.permute([1,0,2]).reshape(3,int(positions.numel()/3)).t()
        keep_indices = np.where(cdist((a.float()*scale.reshape(1,3)).half(), (pos.float()*scale.reshape(1,3)).half()).min(0)<self.radius.cpu().numpy())
        
        if type(keep_indices) is tuple and len(keep_indices) > 1:
            keep_indices = keep_indices[1]
        P = pos[torch.tensor(np.array(keep_indices).T),:].squeeze().float()
        self.P = P.to(device=self.device)
            
        Y = video.reshape([int(float(video.numel())/video.shape[3]), video.shape[3]]).squeeze()
        Y = Y[keep_indices,:].squeeze()
    
        self.Y = Y.to(device=self.device)
        
        if not hasattr(self, 'method'):
            self.method = method
        
        
        if not hasattr(self, 'LL'):
            K = positions.shape[0]
            T = video.shape[3]
            
            LL_init = torch.zeros((K))
            for k in range(K):
                LL_init[k] = sigma_inv
            self.LL = LL_init.clone().detach().requires_grad_(False).to(device=self.device)

        
            
        if not hasattr(self, 'C'):
            C = torch.rand(K,T)
            self.C = C.to(device=self.device)
        
        if not hasattr(self, 'pseudo_colors'):
            R = np.linspace(0,1,self.C.shape[0]+1)[0:-1]
            color = plt.cm.hsv(R)[:,0:3]
            np.random.shuffle(color)
            
            self.pseudo_colors = torch.tensor(color).float().to(device=self.device)
        
        if not hasattr(self, 'B'):
            B_init = torch.zeros((10,3,T))
            for t in range(T):
                B_init[:,:,t] = torch.cat((torch.zeros(1,3), torch.eye(3), torch.zeros(6,3)),0)
        
            B = B_init.clone()
            self.B = B.detach().requires_grad_(True).to(device=self.device)
            
        if not hasattr(self, 'BG'):
            self.BG = torch.tensor(np.percentile(video,60)).to(device=self.device)
        
        if not hasattr(self, 'scale'):
            self.scale = scale.reshape(1,3).to(device=self.device)
            
        self.RE = ((self.Y.clone())*0).to(device=self.device)
        
        if not hasattr(self, 'cost'):
            self.cost = []
            
        self.verbose = verbose
    
    def optimize(self,lr=.1,n_iter=100,n_iter_c=20,sample_size=None):
#        optimizer = optim.SGD({self.B}, lr=lr)
        optimizer = optim.Adam({self.B}, lr=lr)
        
        if sample_size is None:
            y_ind = np.arange(self.Y.shape[0])
            t_ind = np.arange(self.Y.shape[1])
                
        for iter in range(n_iter):
            if self.verbose:
                print(self.LL.mean())
                print('Update C:')
            
            
            if sample_size is not None:
                y_ind = np.random.randint(0,self.Y.shape[0]-1,sample_size[0])
                t_ind = np.random.randint(0,self.Y.shape[1]-1,sample_size[1])
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
                y_ind = np.random.randint(0,self.Y.shape[0]-1,sample_size[0])
                t_ind = np.random.randint(0,self.Y.shape[1]-1,sample_size[1])
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
#                A = A_p@self.B[:,:,t] + self.B0[:,:,t]
                
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
#            + self.B0[:,:,t]
            
            cost = cost + self.gamma*(dNMF.det_jac(self.B[:,:,t],self.P.min(0)[0])-1)**2 \
                        + self.gamma*(dNMF.det_jac(self.B[:,:,t],self.P.min(0)[0])-1)**2
            
            if t > 1:
#                cost = cost + self.gamma*((self.B[:,:,t] - self.B[:,:,t-1])**2).mean()
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
#            +self.B0[:,:,t]
        
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

    
    
    def visualize_raw(self, file, video):
        fig, ax = plt.subplots(figsize=(10,10))
        im = ax.imshow(video[:,:,:,0].max(2)[0].squeeze())

        def init():
            im.set_data(video[:,:,:,0].max(2)[0].squeeze())
            return (im,)
    
        def animate(t):
            data_slice = video[:,:,:,t].max(2)[0].squeeze()
            im.set_data(data_slice)
            
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
        
        ax.axis('off')
        scalebar = ScaleBar(self.scale[0,0],'um')
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
        # animation function. This is called sequentially
        def animate(t):
            P_t = positions[:,:,t]
            data_slice = video[:,:,:,t].max(2)[0].squeeze()
            im.set_data(data_slice)
            sc.set_offsets(P_t[:,[1,0]])
            
            if labels is not None:
                for i in range(len(labels)):
                    annot[i].set_x(P_t[i,1])
                    annot[i].set_y(P_t[i,0])
            
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
            scalebar = ScaleBar(self.scale[0,0],'um')
            plt.gca().add_artist(scalebar)
            plt.title(neurons[i])
            plt.savefig(file+'-'+neurons[i]+'.png')
            plt.close('all')
            
            
            
    def visualize_neurons(self, file, neurons_plot, neuron_names, video, window=np.array([10,10,1]), fontsize=20):
        neurons = [neurons_plot[i] for i in range(len(neurons_plot)) if neurons_plot[i]+'L' in neuron_names and neurons_plot[i]+'R' in neuron_names]
        
        indices_L = [neuron_names.index(neurons[i]+'L') for i in range(len(neurons))]
        indices_R = [neuron_names.index(neurons[i]+'R') for i in range(len(neurons))]
        
        if not neurons:
            neurons = [neurons_plot[i] for i in range(len(neurons_plot)) if neurons_plot[i] in neuron_names]
            
            indices_L = [neuron_names.index(neurons[i]) for i in range(len(neurons))]
            indices_R = [neuron_names.index(neurons[i]) for i in range(len(neurons))]
            
            
        fact = 5/video.max()
        video = fact*video
        def clean_ax(ax):
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_xticks([])
            ax.autoscale(False)
            scalebar = ScaleBar(self.scale[0,0],'um')
            ax.add_artist(scalebar)
            
        colors = self.pseudo_colors
        
        positions = self.get_positions()
        
        fig = plt.figure(figsize=(len(neurons)*4,20),constrained_layout=True)
        gs = fig.add_gridspec(6, len(neurons))
        ax_main = fig.add_subplot(gs[0:2, 0:int(len(neurons)/2)])
        ax_reco = fig.add_subplot(gs[0:2, int(len(neurons)/2):])
        
        
    
    
        P_L = positions[indices_L,:,:].detach().numpy()
        P_R = positions[indices_R,:,:].detach().numpy()
        
        
        im = [[None for j in range(len(neurons))] for i in range(2,6)]
        ax = [[None for j in range(len(neurons))] for i in range(2,6)]
        
        data_slice = video[:,:,:,0].max(2)[0]
        im_main = ax_main.imshow(data_slice,vmin=0,vmax=1)
        sr_main = ax_main.scatter(P_R[:,1,0],P_R[:,0,0],s=1,color='r',marker='x')
        sl_main = ax_main.scatter(P_L[:,1,0],P_L[:,0,0],s=1,color='g',marker='x')
        ar_main = [ax_main.text(P_R[i,1,0], P_R[i,0,0],neurons[i]+'R',color=self.pseudo_colors[indices_R[i]]) for i in range(len(neurons))]
        al_main = [ax_main.text(P_L[i,1,0], P_L[i,0,0],neurons[i]+'L',color=self.pseudo_colors[indices_L[i]]) for i in range(len(neurons))]
        
        im_reco = ax_reco.imshow(data_slice,vmin=0,vmax=1)
        
        
        ax_reco.set_title('Reconstruction',fontsize=fontsize)
        ax_main.set_title('Data',fontsize=fontsize)
        
        clean_ax(ax_reco)
        clean_ax(ax_main)
        
        
        for i in range(2,6):
            for j in range(len(neurons)):
                ax[i-2][j] = fig.add_subplot(gs[i,j])
                im[i-2][j] = ax[i-2][j].imshow(np.zeros((window[0]*2+1,window[0]*2+1)),vmin=0,vmax=1)
                
                if i == 2:
                    ax[i-2][j].set_title(neurons[j],fontsize=fontsize)
                if j == 0:
                    if i == 2:
                        ax[i-2][j].set_ylabel('L',fontsize=fontsize)
                    elif i == 3:
                        ax[i-2][j].set_ylabel('R',fontsize=fontsize)
                    elif i == 4:
                        ax[i-2][j].set_ylabel('L',fontsize=fontsize)
                    elif i == 5:
                        ax[i-2][j].set_ylabel('R',fontsize=fontsize)
                        
                clean_ax(ax[i-2][j])
        
        
        P = torch.tensor(np.array(np.where(np.ones(video.shape[0:3]))).T).float()
        pos = torch.tensor(np.array(np.where(np.ones(2*window+1))).T).float()
        
        def animate(t):
            if self.verbose:
                print('Time: '+ str(t))
            
            data_slice = video[:,:,:,t].max(2)[0]
            im_main.set_data(data_slice)
            sr_main.set_offsets(P_R[:,[1,0],t])
            sl_main.set_offsets(P_L[:,[1,0],t])
            
            a_k = (self.C[:,t]*torch.exp(-0.5*self.LL.detach()*dNMF.pairwise_distances(P*self.scale,positions[:,:,t]*self.scale))@colors)
            data_slice = (fact*a_k.reshape((video.shape[0],video.shape[1],video.shape[2],3))).max(2)[0]
            data_slice[data_slice < 0] = 0
            data_slice[data_slice > 1] = 1
            im_reco.set_data(data_slice)
            
            for j in range(len(neurons)):
                ar_main[j].set_x(P_R[j,[1],t])
                ar_main[j].set_y(P_R[j,[0],t])
                
                al_main[j].set_x(P_L[j,[1],t])
                al_main[j].set_y(P_L[j,[0],t])
            
            
            for i in range(2,6):
                for j in range(len(neurons)):
                    if i == 2:
                        patch = Utils.subcube(video[:,:,:,t][:,:,:,np.newaxis], P_L[j,:,t], window).max(2)[0].squeeze()
                    elif i == 3:
                        patch = Utils.subcube(video[:,:,:,t][:,:,:,np.newaxis], P_R[j,:,t], window).max(2)[0].squeeze()
                    else:
                        if i == 4:
                            k = indices_L[j]
                        elif i == 5:
                            k = indices_R[j]
                            
                        a_k = (self.C[:,t]*torch.exp(-0.5*self.LL.detach()*dNMF.pairwise_distances(pos*self.scale,(positions[:,:,t] - positions[k,:,t] + torch.tensor(window).float())*self.scale))@colors)
                        patch = (fact*a_k.reshape((2*window[0]+1,2*window[1]+1,2*window[2]+1,3))).max(2)[0]
                        
                        patch[patch < 0] = 0
                        patch[patch > 1] = 1
                    im[i-2][j].set_data(patch)
                    
            return (im_main,)
    
        anim = animation.FuncAnimation(fig, animate, frames=video.shape[3], interval=200, blit=False)
        
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(file+'-neurons.mp4', writer=writer)

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