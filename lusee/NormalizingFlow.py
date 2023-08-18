#30 Aug 2022, Rugved Pund

import torch
from sinf import GIS
import matplotlib.pyplot as plt
import numpy as np
import lusee
import matplotlib
import healpy as hp
import fitsio

class NormalizingFlow:
    def __init__(self,nocuda,loadPath=''):
        self.device=torch.device('cpu') if nocuda else torch.device('cuda')
        try: 
            self.model=torch.load(loadPath).to(self.device)
            self.nlayer=len(self.model.layer)
            print('model loaded from ',loadPath)
        except FileNotFoundError:
            print('no file found, need to train')
        self.precompute_data_after=dict()
    def train(self,train_data,validate_data,nocuda,savePath=None):
        """
        data must be of shape (nsamples, ndim)
        """
        self.nsamples,self.ndim=train_data.shape
        print('now training model with nsamples', self.nsamples,' of ndim',self.ndim)
        self.train_data=self._toTensor(train_data)
        self.validate_data=self._toTensor(validate_data)
        self.model=GIS.GIS(self.train_data.clone(),self.validate_data.clone(),nocuda=nocuda)
        self.nlayer=len(self.model.layer)
        print('Total number of iterations: ', len(self.model.layer))
        if savePath is not None: 
            torch.save(self.model,savePath)
            print('model saved at', savePath)
    def _toTensor(self,nparray):
        return torch.from_numpy(nparray).float().to(self.device)
    def plotDataToGaussian(self,data,label=None, niterationSteps=8):
        data=self._toTensor(data)
        dn=(self.nlayer-1)//niterationSteps
        plt.figure(figsize=(15,12))
        for ii,i in enumerate(range(0,self.nlayer,dn)):
            plt.subplot(niterationSteps//4+1,4,ii+1)
            data_after=np.array(self.model.forward(data, end=i)[0])
            plt.title(f'after {i} iterations')
            plt.hist2d(data_after[:,0],data_after[:,1],bins=50)
        plt.suptitle(('Data' if label is None else label)+' to Gaussian Flow')
        plt.show()
    def plotGaussianToData(self,gaussdata, label=None, niterationSteps=8):
        gaussdata=self._toTensor(gaussdata)
        dn=(self.nlayer-1)//niterationSteps
        plt.figure(figsize=(15,12))
        for ii,i in enumerate(range(0,self.nlayer,dn)):
            plt.subplot(niterationSteps//4+1,4,ii+1)
            gaussdata_after=np.array(self.model.inverse(gaussdata, start=i)[0]) #weird convention
            plt.title(f'after {i} iterations')
            plt.hist2d(gaussdata_after[:,0],gaussdata_after[:,1],bins=50)
        plt.suptitle('Gaussian to ' + ('Data' if label is None else label))
        plt.show()
    def plotLikelihood(self, testgrid,logscale=False, niterationSteps=8):
        testgrid=self._toTensor(testgrid)
        dn=(self.nlayer-1)//niterationSteps
        plt.figure(figsize=(15,12))
        for ii,i in enumerate(range(0,self.nlayer,dn)):
            plt.subplot(niterationSteps//4+1,4,ii+1)
            logp=np.array(self.model.evaluate_density(testgrid, end=i)).reshape(100,100)
            plt.title(f'after {i} iterations')
            plt.imshow(np.exp(logp),extent=[-3,2,-3,2],
                      norm=matplotlib.colors.LogNorm() if logscale else None)
        plt.colorbar()
        plt.suptitle('Likelihood')
        plt.show()
    def plotHistograms(self,data,modeidxToPlot,other=None,means=None, t21=None,logscale=False,niterationSteps=1, fname=None):
        self._precomputeIterations(data,niterationSteps)
        data=self._toTensor(data)
        other=self._toTensor(other) if other is not None else None
        means=self._toTensor(means) if means is not None else None
        t21=self._toTensor(t21) if t21 is not None else None
        
        dn=(self.nlayer-1)//niterationSteps
        kwargs={'bins':50,'density':True,'norm':matplotlib.colors.LogNorm() if logscale else None}
        for ii,i in enumerate(range(0,self.nlayer,dn)):
            plt.figure(figsize=(15,15))
            data_after=self.precompute_data_after.get(i)
            other_after=(self.model.forward(other,end=i)[0]).numpy() if other is not None else None
            means_after=(self.model.forward(means,end=i)[0]).cpu().numpy() if means is not None else None
            t21_after=(self.model.forward(t21,end=i)[0]).cpu().numpy() if t21 is not None else None
            for jj,j in enumerate(modeidxToPlot):
                for kk,k in enumerate(modeidxToPlot):
                    plt.subplot(len(modeidxToPlot),len(modeidxToPlot),len(modeidxToPlot)*jj+kk+1)
                    if j==k: 
                        plt.hist(data_after[:,k])
                        if means is not None: plt.axvline(means_after[:,k],c='r')
                        if t21 is not None: plt.axvline(t21_after[:,k],c='g')
                    else:
                        plt.hist2d(data_after[:,k],data_after[:,j],**kwargs,cmap='Blues')
                    if other is not None: plt.hist2d(other_after[:,k],other_after[:,j],**kwargs,
                                                    cmap='cividis')
                    if means is not None and j!=k: 
                        plt.scatter(means_after[:,k],means_after[:,j],marker='o',c='r',s=12**2)
                    if t21 is not None and j!=k: 
                        plt.scatter(t21_after[:,k],t21_after[:,j],marker='D',c='green',s=12**2)
                    plt.xlabel(f'mode {k+1}')
                    plt.ylabel(f'mode {j+1}')
            # plt.suptitle(f'iteration {i}')
            plt.tight_layout()
            plt.suptitle(fname)
            plt.savefig(f"{fname}_{i}_"+"_".join([str(m) for m in modeidxToPlot])+'.png')
            plt.show()
    def plotCovMat(self,data,logscale=False, niterationSteps=1):
        self._precomputeIterations(data,niterationSteps)
        data=self._toTensor(data)
        dn=(self.nlayer-1)//niterationSteps
        plt.figure(figsize=(15,15))
        for ii,i in enumerate(range(0,self.nlayer,dn)):
            plt.subplot(niterationSteps//3+1,3,ii+1)
            data_after=self.precompute_data_after.get(i)
            plt.imshow(np.cov(data_after.T),cmap='RdBu',
                       norm=matplotlib.colors.SymLogNorm(linthresh=1e-3, vmax=1,vmin=-1) if logscale else None)
            plt.title(f'after {i} iterations')
            plt.colorbar(shrink=0.5)
        plt.show()
    def _precomputeIterations(self,data,niterationSteps=1):
        data=self._toTensor(data)
        assert type(data)==type(self._toTensor(np.array([]))) #why?
        dn=(self.nlayer-1)//niterationSteps
        for ii,i in enumerate(range(0,self.nlayer,dn)):
            check=self.precompute_data_after.get(i,None)
            if check is None:
                print(f'precomputing iteration {i} of', list(range(0,self.nlayer,dn)))        
                data_after,_=self.model.forward(data,end=i)
                self.precompute_data_after[i]=data_after.detach().cpu().numpy()
    def _distanceFromIdentity(self,cov):
        #norm(A_ij)=sqrt(sum(A_ij**2))
        return np.linalg.norm(cov-np.eye(*cov.shape))
    def plotDistanceFromIdentity(self,data,niterationSteps=4):
        self._precomputeIterations(data,niterationSteps)
        data=self._toTensor(data)
        dn=(self.nlayer-1)//niterationSteps
        distFromId=[]
        for ii,i in enumerate(range(0,self.nlayer,dn)):
            data_after=self.precompute_data_after.get(i)
            distFromId.append(self._distanceFromIdentity(np.cov(data_after.T)))
        plt.plot(range(0,self.nlayer,dn),distFromId,'o-')
        plt.title('Frobenius Distance of Covmat from Identity')
        plt.xlabel('iteration')
        plt.show()
    def _minmaxfinder(self,data):
        nsamples,ndim=data.shape
        self.minmax=np.zeros((ndim,2))
        self.minmax[:,0]=data.min(axis=0)
        self.minmax[:,1]=data.max(axis=0)
    def _likelihood(self,data):
        data=self._toTensor(data)
        return self.model.evaluate_density(data)
        
class FlowAnalyzer(NormalizingFlow):
    def __init__(self, nocuda, loadPath):
        super().__init__(nocuda=nocuda,loadPath=loadPath)
    def precomputeIterations(self,data,niterationSteps):
        self._precomputeIterations(data,niterationSteps)
    def set_fg(self, fg, sigma, chromatic, galcut, noPCA, subsample, noise_K, noiseSeed, combineSigma):
        self.fg=fg
        self.sigma=sigma
        self.chromatic=chromatic
        self.galcut=galcut
        self.noPCA=noPCA
        self.subsample=subsample
        self.noise_K=noise_K
        self.noiseSeed=noiseSeed
        self.combineSigma=combineSigma
        
        self.noise=self.generate_noise(self.fg,self.noise_K,self.subsample,self.noiseSeed)
        self.fg_noisy=self.fg+self.noise
        self.fgsmooth=self.smooth(self.fg_noisy,self.sigma,self.chromatic)
        
        if self.galcut is not None:
            self.fgsmooth_cut=self.galaxy_cut(self.fgsmooth,self.galcut)
        else:
            self.fgsmooth_cut=self.fgsmooth
            
        if self.noPCA: 
            self.data,self.fgmeansdata=self.get_data_noPCA(self.fgsmooth_cut, vector=self.fgsmooth_cut.mean(axis=1))
        else:
            self.data,self.fgmeansdata=self.get_data_PCA(self.fgsmooth_cut, vector=self.fgsmooth_cut.mean(axis=1))
        
        if self.combineSigma is not None:
            self.fgsmooth2=self.smooth(self.fg,self.combineSigma,self.chromatic)
            if self.galcut is not None:
                self.fgsmooth_cut2=self.galaxy_cut(self.fgsmooth2,self.galcut)
            else:
                self.fgsmooth_cut2=self.fgsmooth2
            self.fgsmoothcut_combineSigma=np.vstack([self.fgsmooth_cut,self.fgsmooth_cut2])
            if self.noPCA: 
                self.data,self.fgmeansdata=self.get_data_noPCA(self.fgsmoothcut_combineSigma, vector=self.fgsmoothcut_combineSigma.mean(axis=1))
            else:
                self.data,self.fgmeansdata=self.get_data_PCA(self.fgsmoothcut_combineSigma, vector=self.fgsmoothcut_combineSigma.mean(axis=1))

            
        assert self.data.shape[0]>self.data.shape[1]
        assert self.subsample>=3
        print(f'subsampling {self.subsample}')
        self.train_data=self.data[::self.subsample,:]
        self.validate_data=self.data[1::self.subsample,:]
        self.test_data=self.data[2::self.subsample,:]
        print('done! self.train_data,validate_data and test_data ready')

    def generate_noise(self,fg,noise_K,subsample,seed=0):
        nfreq,ndata=fg.shape
        np.random.seed(seed)
        noise_sigma=noise_K*np.sqrt(ndata/subsample)
        noise=np.random.normal(0,noise_sigma,(nfreq,ndata))
        return noise
    
    def smooth(self,fg,sigma,chromatic):
        print(f'smoothing with {sigma} deg, and chromatic {chromatic}')
        fgsmooth=np.zeros_like(fg)
        nfreq,ndata=fg.shape
        for f in range(nfreq):
            sigma_f=sigma*(10.0/(f+1)) if chromatic else sigma
            fgsmooth[f,:]=hp.sphtfunc.smoothing(fg[f,:],sigma=np.deg2rad(sigma_f))
        return fgsmooth

    def galaxy_cut(self,fg,b_min):
        print(f'doing galcut for b_min={b_min} deg')
        _,npix=fg.shape
        nside=np.sqrt(npix/12).astype(int)
        col_min = np.pi/2 - np.deg2rad(b_min)
        col_max = np.pi/2 + np.deg2rad(b_min)
        mask_pix = hp.query_strip(nside, col_min, col_max, inclusive=False)
        cutfg = np.delete(fg, mask_pix, axis=1)
        return cutfg

    def get_data_PCA(self, data, vector=None):
        """
        data should be of shape (nfreqs,ndata)
        vector should be of shape (nfreqs)
        out is of shape (ndata,nfreqs) - because torch needs it that way
        """
        print('doing PCA')
        assert data.shape[0]<data.shape[1]
        # if vector is not None: assert vector.shape==data.shape[0]
        cov=np.cov(data)
        eva,eve=np.linalg.eig(cov)
        s=np.argsort(np.abs(eva))[::-1] 
        eva,eve=np.real(eva[s]),np.real(eve[:,s])
        proj_data=eve.T@(data-data.mean(axis=1)[:,None]) #subtract mean and project
        rms=np.sqrt(proj_data.var(axis=1))
        out=(proj_data/rms[:,None]).T #divide by rms
        out=np.random.permutation(out)
        if vector is None: 
            return out
        else:
            return out, (eve.T@vector)/rms

    def get_data_noPCA(self, data, vector=None):
        print('not doing PCA')
        assert data.shape[0]<data.shape[1]
        rms=np.sqrt(data.var(axis=1))
        out=((data - data.mean(axis=1)[:,None])/rms[:,None]).T
        out=np.random.permutation(out)
        if vector is None:
            return out
        else:
            return out, vector/rms
    
    def set_t21(self, t21, include_noise):
        if self.combineSigma is not None:
            print(f'combining sigma {self.sigma} with combineSigma {self.combineSigma}')
            if include_noise: 
                t21+=self.noise.mean(axis=1)
                print('including noise in t21')
            self.t21=np.hstack([t21,t21])
            print(self.t21.shape, self.fgsmoothcut_combineSigma.shape)
            print(f'projecting t21 according to noPCA={self.noPCA}')
            if self.noPCA:
                _,self.t21data=self.get_data_noPCA(self.fgsmoothcut_combineSigma, self.t21)
            else:
                _,self.t21data=self.get_data_PCA(self.fgsmoothcut_combineSigma, self.t21)
            print('done! self.t21data ready')
        else:
            if include_noise: 
                t21+=self.noise.mean(axis=1)
                print('including noise in t21')
            self.t21=t21
            print(f'projecting t21 according to noPCA={self.noPCA}')
            if self.noPCA:
                _,self.t21data=self.get_data_noPCA(self.fgsmooth_cut, self.t21)
            else:
                _,self.t21data=self.get_data_PCA(self.fgsmooth_cut, self.t21)
            print('done! self.t21data ready')
            
        
def load_ulsa_t21_tcmb_freqs():
    root='/home/rugved/Files/LuSEE/luseepy/Drive/'
    fg=fitsio.read(root+'Simulations/SkyModels/ULSA_32_ddi_smooth.fits')
    freqs=np.arange(1,51)
    t21=lusee.mono_sky_models.T_DarkAges(freqs)
    tcmb=np.ones_like(t21)*2.73
    return fg, t21, tcmb, freqs

def smooth(fg,sigma,chromatic):
    fgsmooth=np.zeros_like(fg)
    nfreq,_=fg.shape
    for f in range(nfreq):
        sigma_f=sigma*(10.0/(f+1)) if chromatic else sigma
        fgsmooth[f,:]=hp.sphtfunc.smoothing(fg[f,:],sigma=np.deg2rad(sigma_f))
    return fgsmooth

def get_projected_data(data, vector=None):
    """
    data should be of shape (nfreqs,ndata)
    vector should be of shape (nfreqs)
    out is of shape (ndata,nfreqs) - because torch needs it that way
    """
    assert data.shape[0]<data.shape[1]
    # if vector is not None: assert vector.shape==data.shape[0]
    cov=np.cov(data)
    eva,eve=np.linalg.eig(cov)
    s=np.argsort(np.abs(eva))[::-1] 
    eva,eve=np.real(eva[s]),np.real(eve[:,s])
    proj_data=eve.T@(data-data.mean(axis=1)[:,None]) #subtract mean and project
    rms=np.sqrt(proj_data.var(axis=1))
    out=(proj_data/rms[:,None]).T #divide by rms
    out=np.random.permutation(out)
    if vector is None: 
        return out
    else:
        return out, (eve.T@vector)/rms

