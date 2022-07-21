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

    def SNR (self, template, plot=None):
        rottemp_sq = (self.eve.T@(template*self.weight))**2
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
    

    
    # def get_subtracted_data(self):
    #     mdata = np.copy(self.mean_data)
    #     out = [np.copy(mdata)]
    #     for mode in self.eve.T:
    #         mdata -= np.dot(mdata,mode)/np.dot(mode,mode)*mode
    #         out.append(np.copy(mdata))
    #     return np.array(out)

    
    
    # def SNR (self, template, noise, verbose = False, i_max=None, min_rms_frac = 0 ):
    #     if type(noise) == float:
    #         noise = noise*np.ones(self.Nfreq)
    #     template = template*self.weight
    #     noise = noise*self.weight
    #     i = 0
    #     mdata = np.copy(self.mean_data)
    #     SNR_best = np.sqrt( (template**2/(mdata**2+noise**2)).sum() )
    #     if (verbose):
    #         print (f"Eigen value 0, SNR = {SNR_best}")

    #     initial_template_var = template.var()
    #     for (i,mode), evalue in zip(enumerate(self.eve.T),self.eva):
    #         if evalue<0:
    #             break
    #         if (i_max is not None) and (i>i_max):
    #             break
    #         mdata -= np.dot(mdata,mode)/np.dot(mode,mode)*mode
    #         template -= np.dot(template,mode)/np.dot(mode,mode)*mode
    #         SNR = np.sqrt( np.minimum(10,(template**2/(mdata**2+noise**2))).sum() )
    #         rms_frac = np.sqrt(template.var()/initial_template_var)
    #         if (verbose):
    #             print (f"Eigen value {i+1}, SNR = {SNR}, template_frac = {rms_frac}")
    #         if rms_frac<min_rms_frac:
    #             break
    #         if SNR>SNR_best:
    #             SNR_best = SNR
    #             mdata_best = np.copy(mdata)
    #             template_best = np.copy(template)
    #             i_best = i
    #     return SNR_best, i_best+1, mdata_best, template_best, noise
    
        
            
    
            
        
        
