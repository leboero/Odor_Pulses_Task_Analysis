import numpy as np
import scipy
from scipy import signal
from scipy.stats import norm,entropy,linregress
from scipy.optimize import minimize, curve_fit
from scipy.io import savemat
import matplotlib.pyplot as plt


def calc_plam(num_pulses,lam1,lam2):
    pow_lam1 = np.power(lam1,num_pulses)
    pow_lam2 = np.power(lam2,num_pulses)
    exp_lam1 = np.exp(-lam1)
    exp_lam2 = np.exp(-lam2)
    return pow_lam1*exp_lam1/(pow_lam1*exp_lam1+pow_lam2*exp_lam2)

def compute_pcorrect(choices,Nps,Nb,sig_dN,sig_Np,Nmin=1):
    sig_Nps = sig_Np[Nps-Nmin]
    sig = np.sqrt(sig_Nps**2+sig_dN**2) #There's a 2 missing
    phigh = norm.cdf((Nps-Nb)/sig)
    pcorrect = (1-choices)+(choices*2-1)*phigh
    return pcorrect

def compute_logp_correct(choices,Nps,Nb,sig_dN,sig_Np,Nmin=1):
    pcorrect = compute_pcorrect(choices,Nps,Nb,sig_dN,sig_Np,Nmin=1)
    log_pcorrect = np.log(pcorrect)
    return log_pcorrect.sum()

def compute_pcorrect_weighted(choices,weighted_odor,Nps,Nb,sig_dN,sig_Np,Nmin=1):
    sig_Nps = sig_Np[Nps-Nmin]
    sig = np.sqrt(sig_Nps**2+sig_dN**2)
    phigh = norm.cdf((weighted_odor-Nb)/sig)
    pcorrect = (1-choices)+(choices*2-1)*phigh
    return pcorrect

def compute_phigh(Np,Nb,sig_dN,sig_Np):
    sig = np.sqrt(sig_Np**2+sig_dN**2)
    phigh = norm.cdf((Np-Nb)/sig)
    return phigh

def compute_phigh_sq(Np,Nb,sig_dN,ksig,Nmin=1,Nmax=18):
    sig_Np = np.sqrt(ksig*(np.arange(Nmin,Nmax+1)))
    sig = np.sqrt(sig_Np**2+sig_dN**2)
    phigh = norm.cdf((Np-Nb)/sig)
    return phigh

def compute_phigh_lin(Np,Nb,sig_dN,ksig,sig_0,Nmin=1,Nmax=18):
    sig_Np = ksig*(np.arange(Nmin,Nmax+1)) + sig_0
    sig = np.sqrt(sig_Np**2+sig_dN**2)
    phigh = norm.cdf((Np-Nb)/sig)
    return phigh

def compute_phigh_lin_sub(Np,Nb,sig_dN,ksig,sig_0,Nmin=1,Nmax=18): #ksig,sig_0,sub
    #sig_Np = ksig*(np.arange(Nmin,Nmax+1)) + sig_0
    sub_Np = np.zeros(4) #sig_0*np.ones(4)
    sig_Np = np.concatenate([sub_Np,ksig*(np.arange(5,Nmax+1,1))+sig_0]) # + sig_0
    print(sig_Np)
    sig = np.sqrt(sig_Np**2+sig_dN**2)   #Why this is not multiplied by 2
    phigh = norm.cdf((Np-Nb)/sig)
    return phigh

def compute_phigh_pow(Np,Nb,sig_dN,ksig,ind,Nmin=1,Nmax=18):
    sig_Np = ksig*np.power((np.arange(Nmin,Nmax+1)),ind)
    sig = np.sqrt(sig_Np**2+sig_dN**2)
    phigh = norm.cdf((Np-Nb)/sig)
    return phigh

def compute_phigh_const(Np,Nb,sig_dN,Nmin=1,Nmax=18):
    sig = sig_dN
    phigh = norm.cdf((Np-Nb)/sig)
    return phigh

def compute_logp_correct_weighted(choices,weighted_odor,Nps,Nb,sig_dN,sig_Np):
    pcorrect = compute_pcorrect_weighted(choices,weighted_odor,Nps,Nb,sig_dN,sig_Np)
    log_pcorrect = np.log(pcorrect)
    return log_pcorrect.sum()

def compute_logp_correct_sq(choices,Nps,Nb,sig_dN,ksig,Nmin=1,Nmax=18):
    sig_Np = np.sqrt(ksig*(np.arange(Nmin,Nmax+1)))
    return compute_logp_correct(choices,Nps,Nb,sig_dN,sig_Np)

def compute_logp_correct_sq_weighted(choices,weighted_odor,Nps,Nb,sig_dN,ksig,Nmin=1,Nmax=18):
    sig_Np = np.sqrt(ksig*(np.arange(Nmin,Nmax+1)))
    return compute_logp_correct_weighted(choices,weighted_odor,Nps,Nb,sig_dN,sig_Np)
    
def compute_logp_correct_lin(choices,Nps,Nb,sig_dN,ksig,sig_0,Nmin=1,Nmax=18):
    sig_Np = ksig*(np.arange(Nmin,Nmax+1)) + sig_0
    pcorrect = compute_pcorrect(choices,Nps,Nb,sig_dN,sig_Np,Nmin)
    log_pcorrect = np.log(pcorrect)
    return log_pcorrect.sum()

def compute_logp_correct_lin_weighted(choices,weighted_odor,Nps,Nb,sig_dN,ksig,sig_0,Nmin=1,Nmax=18):
    sig_Np = ksig*(np.arange(Nmin,Nmax+1)) + sig_0
    pcorrect = compute_pcorrect_weighted(choices,weighted_odor,Nps,Nb,sig_dN,sig_Np)
    log_pcorrect = np.log(pcorrect)
    return log_pcorrect.sum()

def compute_logp_correct_lin_sub(choices,Nps,Nb,sig_dN,ksig,sig_0,Nmin=1,Nmax=18): #ksig,sig_0,sub
    #sig_Np = ksig*(np.arange(Nmin,Nmax+1)) + sig_0
    sub_NP = np.zeros(4)
    sig_Np = np.concatenate([sub_NP,ksig*(np.arange(5,Nmax+1,1))+sig_0]) # + sig_0
    #sig_Np[0:4] = sub
    pcorrect = compute_pcorrect(choices,Nps,Nb,sig_dN,sig_Np,Nmin)
    log_pcorrect = np.log(pcorrect)
    return log_pcorrect.sum()

def compute_logp_correct_pow(choices,Nps,Nb,sig_dN,ksig,ind,Nmin=1,Nmax=18):
    sig_Np = ksig*np.power((np.arange(Nmin,Nmax+1)),ind)
    pcorrect = compute_pcorrect(choices,Nps,Nb,sig_dN,sig_Np,Nmin)
    log_pcorrect = np.log(pcorrect)
    return log_pcorrect.sum()

def compute_logp_correct_const(choices,Nps,Nb,sig_dN,Nmin=1,Nmax=18):
    sig_Np = np.zeros((Nmax-Nmin+1,))
    pcorrect = compute_pcorrect(choices,Nps,Nb,sig_dN,sig_Np,Nmin)
    log_pcorrect = np.log(pcorrect)
    return log_pcorrect.sum()

