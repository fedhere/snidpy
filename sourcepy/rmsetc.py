import numpy as np

def rrvector(buf1): return buf1.copy()
def rvscale(buf1, v): buf1 *= v
def cvscale(x1, v): x1 *= v
def ccvector(x1): return x1.copy()
def rcvector(buf): return buf.astype(complex)
def crvector(x): return x.real
def correlate(ft1, ft2): return ft1 * ft2.conj()
def getrms(x): return np.sqrt(np.abs(x)**2).sum() / len(x)
def aspart(n, k1, k2, k3, k4, shift, x):
    k = np.arange(n)
    angle = -2*np.pi*k*shift/n
    phase = np.exp(1j*angle)
    f = np.where((k==0)|(k==n/2), 1, 2)
    window = np.ones(n)
    mask = (k>=k1)&(k<=k4)
    window[mask&(k<k2)] = 0.25*(1-np.cos(np.pi*(k[mask&(k<k2)]-k1)/(k2-k1)))**2
    window[mask&(k>k3)] = 0.25*(1+np.cos(np.pi*(k[mask&(k>k3)]-k3)/(k4-k3)))**2
    window[~mask] = 0
    arms = np.sqrt((f*window*np.imag(phase*x)**2).sum())/n
    srms = np.sqrt((f*window*np.real(phase*x)**2).sum())/n
    return arms, srms
def shiftit(n, shift, x):
    k = np.arange(n); k[k>n/2] -= n
    x *= np.exp(-2j*np.pi*k*shift/n)
def rmsfilter(n, k1, k2, k3, k4, x):
    k = np.arange(n); k[k>n/2] -= n; k = np.abs(k)
    mask = (k>=k1)&(k<=k4)
    window = np.zeros(n)
    window[mask&(k<k2)] = 0.5*(1-np.cos(np.pi*(k[mask&(k<k2)]-k1)/(k2-k1)))
    window[mask&(k>k3)] = 0.5*(1+np.cos(np.pi*(k[mask&(k>k3)]-k3)/(k4-k3)))
    window[mask&(k>=k2)&(k<=k3)] = 1
    f = np.where((k==0)|(k==n/2), 1, 2)
    return np.sqrt((f*window*np.abs(x)**2).sum())/n
def filter(n, k1, k2, k3, k4, x):
    k = np.arange(n); k[k>n/2] -= n; k = np.abs(k)
    mask = (k>=k1)&(k<=k4)
    window = np.zeros(n, dtype=complex)
    window[mask&(k<k2)] = 0.5*(1-np.cos(np.pi*(k[mask&(k<k2)]-k1)/(k2-k1)))
    window[mask&(k>k3)] = 0.5*(1+np.cos(np.pi*(k[mask&(k>k3)]-k3)/(k4-k3)))
    window[mask&(k>=k2)&(k<=k3)] = 1
    x *= window
def phaseband(n, shift, x):
    phase = np.angle(x[:n//2])
    k = np.arange(n//2)
    angle = 2*np.pi*k*shift/n
    phase -= 2*np.pi*np.round((phase-angle)/np.pi/2)*np.pi
    full = np.zeros(n)
    full[:n//2] = phase
    full[n//2:] = phase[::-1] + angle[::-1]
    return full
