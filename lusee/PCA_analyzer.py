import numpy as np
import scipy.linalg as la


class PCA_Analyzer:

    def __init__ (self, data, weight = None, noise = None):
        self.Nfreq = data.shape[1]
        if weight is None:
            weight = np.ones(self.Nfreq)
        self.weight = weight
        mdata = data*self.weight[None,:]
        C = np.cov(mdata, rowvar = False)
        if noise is not None:
            wnoise = noise*self.weight
            C += np.outer(noise,noise)
        eva,eve = la.eig(C)
        s = np.argsort(eva)[::-1]
        self.eva,self.eve = np.real(eva[s]), np.real(eve[:,s])
        
        self.mean_data = mdata.mean(axis=0)
        
    def SNR (self, template, noise, verbose = False, i_max=None, min_rms_frac = 0 ):
        if type(noise) == float:
            noise = noise*np.ones(self.Nfreq)
        template = template*self.weight
        noise = noise*self.weight
        i = 0
        mdata = np.copy(self.mean_data)
        SNR_best = np.sqrt( (template**2/(mdata**2+noise**2)).sum() )
        if (verbose):
            print (f"Eigen value 0, SNR = {SNR_best}")

        initial_template_var = template.var()
        for (i,mode), evalue in zip(enumerate(self.eve.T),self.eva):
            if evalue<0:
                break
            if (i_max is not None) and (i>i_max):
                break
            mdata -= np.dot(mdata,mode)/np.dot(mode,mode)*mode
            template -= np.dot(template,mode)/np.dot(mode,mode)*mode
            SNR = np.sqrt( (template**2/(mdata**2+noise**2)).sum() )
            rms_frac = np.sqrt(template.var()/initial_template_var)
            if (verbose):
                print (f"Eigen value {i+1}, SNR = {SNR}, template_frac = {rms_frac}")
            if rms_frac<min_rms_frac:
                break
            if SNR>SNR_best:
                SNR_best = SNR
                mdata_best = np.copy(mdata)
                template_best = np.copy(template)
                i_best = i
        return SNR_best, i_best+1, mdata_best, template_best, noise
    
        
            
    
            
        
        
