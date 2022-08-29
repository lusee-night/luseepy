import numpy as np
import scipy.linalg as la


class PCA_Analyzer:

    def __init__ (self, data, noise, weight = None):
        self.Nfreq = data.shape[1]
        if weight is None:
            weight = np.ones(self.Nfreq)
        self.weight = weight
        mdata = data*self.weight[None,:]
        C = np.cov(mdata, rowvar = False)
        eva0,eve0 = la.eig(C)
        s = np.argsort(eva0)[::-1]
        self.eva0,self.eve0 = np.real(eva0[s]), np.real(eve0[:,s])
        if noise is not None:
            wnoise = noise*self.weight
            C += np.diag(wnoise**2)
        self.noise = noise
            
        eva,eve = la.eig(C)
        s = np.argsort(eva)[::-1]
        self.eva,self.eve = np.real(eva[s]), np.real(eve[:,s])
        # recalculate eva
        pdata = np.einsum('ij,ki->kj',self.eve,mdata)
        self.evax = np.maximum((pdata**2).mean(axis=0),self.eva)
        self.mean_data = mdata.mean(axis=0)

    def get_template_power (self,template):
        rottemp_sq = (self.eve.T@(template*self.weight))**2
        return rottemp_sq
    
    def SNR (self, template, plot=None):
        rottemp_sq = self.get_template_power(template)
        SNR = np.sqrt(np.sum(rottemp_sq/self.eva))
        if plot is not None:
            x = np.arange(len(self.eva))+1
            plot.plot (x, self.eva, label = 'FG + noise')
            plot.plot (x, self.eva0, label = 'FG')
            plot.plot (x, rottemp_sq, label = 'signal')
            plot.semilogy()
            plot.legend()
        return SNR
    
    def get_chi2 (self, template,data, istart=2):
        rottemp = self.eve.T@(template*self.weight)
        rotdata = self.eve.T@(data*self.weight)
        chi2 = ((rottemp-rotdata)**2/self.evax)[istart:].sum()
        return chi2
    
    def get_chi2_table (self, templates, data, istart=2):
        chi2 = [[self.get_chi2(t,data) for t in x] for x in templates]
        return np.array(chi2)
    
    def sim_data (self, template):
        return self.mean_data/self.weight + template + np.random.normal(0,self.noise,self.Nfreq)



class Composite_PCA_Analyzer:
    def __init__ (self, alist):
        self.alist = alist

    def SNR(self, template_list, plot=None, verbose=False):
        SNR2 = 0
        for a,t in zip(self.alist, template_list):
            cSNR = a.SNR(t,plot)
            if verbose:
                print (f"sub SNR = {cSNR}")
            SNR2 += (cSNR)**2
        return np.sqrt(SNR2)

    def sim_data(self,template_list):
        return [a.sim_data(t) for a,t in zip(self.alist, template_list)]

    def get_chi2 (self,template_list, data, istart=2):
        return np.sum([a.get_chi2(t,d) for a,t,d in zip(self.alist, template_list, data)])
    
        
    def get_chi2_table (self, templates, data, istart=2):
        chi2 = [[self.get_chi2(t,data) for t in x] for x in templates]
        return np.array(chi2)
    
