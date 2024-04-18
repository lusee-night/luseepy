import numpy as np
import scipy.linalg as la


class PCAanalyzer:
    """
    Class that performs Principle Component Analysis on sky data and sky signal templates

    :param data: Sky data for analysis, two dimensional array where second dimension is frequency
    :type data: class
    :param noise: Noise array for data at N frequencies
    :type noise: array
    :param weight: Weight array for data at N frequencies
    :type weight: array
    """

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
        """
        Function that calculates power from weighted sky signal template

        :param template: Sky signal template
        :type template: array

        :returns: Power in sky signal
        :rtype: array
        """
        rottemp_sq = (self.eve.T@(template*self.weight))**2
        return rottemp_sq
    
    def SNR (self, template, plot=None):
        """
        Function that calculates the signal-to-noise ratio of input sky signal template against sky noise

        :param template: Sky signal template
        :type template: array
        :param plot: Whether to plot results
        :type plot: bool

        :returns: SNR
        :rtype: float
        """
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
    
    def get_chi2 (self, template, data, istart=2):
        """
        Function that calculates the Chi-squared value of the input sky signal template against sky data

        :param template: Sky signal template
        :type template: array
        :param data: Sky data
        :type data: array
        :param istart: Frequency index at which to start
        :type istart: int

        :returns: Chi-squared
        :rtype: float
        """
        rottemp = self.eve.T@(template*self.weight)
        rotdata = self.eve.T@(data*self.weight)
        chi2 = ((rottemp-rotdata)**2/self.evax)[istart:].sum()
        return chi2
    
    def get_chi2_table (self, templates, data, istart=2):
        """
        Function that calculates the Chi-squared value of a list of input sky signal templates against sky data

        :param templates: List of sky signal templates
        :type templates: list(array)
        :param data: Sky data
        :type data: array
        :param istart: Frequency index at which to start
        :type istart: int

        :returns: Chi-squared array
        :rtype: array
        """
        chi2 = [[self.get_chi2(t,data) for t in x] for x in templates]
        return np.array(chi2)
    
    def sim_data (self, template):
        """
        Function that calculates a simulated sky spectrum by summing 1) The mean un-weighted sky data in each frequency bin (from the sky data input to the PCAanalyzer class) as a background DC level, 2) the input sky signal template, and 3) a randomly generated instance of sky noise, with mean zero and standard deviation self.noise, at each of the N frequencies in the input sky data.

        :param template: Sky signal template
        :type template: array

        :returns: Simulated sky spectrum
        :rtype: array
        """
        return self.mean_data/self.weight + template + np.random.normal(0,self.noise,self.Nfreq)



class CompositePCAanalyzer:
    """
    Class that performs Principle Component Analysis on the composite of several sets of sky data

    :param alist: list of PCAanalyzer classes for the sky data sets
    :type alist: list(class)
    """
    def __init__ (self, alist):
        self.alist = alist

    def SNR(self, template_list, plot=None, verbose=False):
        """
        Function that calculates the signal-to-noise ratio of input sky signal template against sky noise

        :param template_list: List of sky signal templates
        :type template_list: list(array)
        :param plot: Whether to plot results
        :type plot: bool
        :param verbose: Whether to print SNR for each analyzer in list
        :type verbose: bool

        :returns: SNR
        :rtype: float
        """
        SNR2 = 0
        for a,t in zip(self.alist, template_list):
            cSNR = a.SNR(t,plot)
            if verbose:
                print (f"sub SNR = {cSNR}")
            SNR2 += (cSNR)**2
        return np.sqrt(SNR2)

    def sim_data(self,template_list):
        """
        Function that calculates simulated sky spectra for each set of sky data in self.alist, and each sky signal template in template_list

        :param template_list: List of sky signal templates
        :type template_list: list(array)

        :returns: Simulated sky spectra
        :rtype: list(array)
        """
        return [a.sim_data(t) for a,t in zip(self.alist, template_list)]

    def get_chi2 (self,template_list, data, istart=2):
        """
        Function that calculates the cumulative Chi-squared value for the list of sky data and sky signal templates in self.alist and template_list

        :param template_list: List of sky signal templates
        :type template_list: list(array)
        :param data: List of sky data sets
        :type data: list(array)
        :param istart: Frequency index at which to start
        :type istart: int

        :returns: Chi-squared
        :rtype: 
        """
        return np.sum([a.get_chi2(t,d) for a,t,d in zip(self.alist, template_list, data)])
    
        
    def get_chi2_table (self, templates, data, istart=2):
        """
        Function that calculates an array of Chi-squared values for a list of different templates, with each template applied to every analyzer in self.alist and set of sky data in data

        :param templates: List of sky signal templates
        :type templates: list(array)
        :param data: List of sky data sets
        :type data: list(array)
        :param istart: Frequency index at which to start
        :type istart: int

        :returns: Chi-squared array
        :rtype: array
        """
        chi2 = [[self.get_chi2(t,data) for t in x] for x in templates]
        return np.array(chi2)

