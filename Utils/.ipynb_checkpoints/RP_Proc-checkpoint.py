import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
from scipy import signal, stats
from scipy.fftpack import fft,ifft
from copy import deepcopy


def fRicker(f,dt):
   nw = 6/f/dt
   nw = 2*math.floor(nw/2)+1
   nc = math.floor(nw/2)
   i = np.arange(1,nw+1,dtype='float')
   alpha = (nc - i + 1)*f*dt*math.pi
   beta = np.power(alpha,2)
   return (1 - np.power(beta,2))*np.exp(-beta)


def fRingdownRemoval(R):
    mR = np.median(R,1)
    Rf = np.transpose(np.transpose(R) - np.transpose(mR))
    return Rf


def TimeShiftParInterp(trace, shift):
    """
    Performs time shift using parabolic interpolation.

    Parameters
    ----------
    trace : numpy.ndarray (1D)
        Input trace.
    shift : float
        Shift in samples.

    Returns
    -------
    out_trace : numpy.ndarray (1D)
        Shifted trace.

    Notes
    -----
    Converted from Matlab script: TimeShiftParInterp.m

    """

    n_samples = np.max(trace.shape)

    if trace.ndim > 1: # check input shape
        if np.min(trace.shape) != 1:
            out_trace = np.nan
            print('Trace must be a 1D vector')
            return out_trace
        else: #reshape array to 1D
            trace = trace.reshape(trace.size)
    else:
        dt = np.mod(shift,1)
        n_eq = np.floor(shift).astype('int')

        op = np.array([0.5*(dt**2 - dt),(1-dt**2),0.5*(dt**2+dt)])

        tmp = np.zeros(trace.shape)
        if n_eq > 0:
            if n_eq > (n_samples-1):
                out_trace = tmp
                return out_trace
            else:
                tmp[n_eq:n_samples] = trace[0:n_samples-n_eq]
        else:
            if n_eq < -n_samples:
                out_trace = tmp
                return out_trace
            else:
                tmp[0:n_samples+n_eq] = trace[-n_eq:n_samples]
        tmp2 = np.convolve(tmp, op)
        out_trace = tmp2[1:n_samples+1]
        return out_trace


def fDASIntegrate(X,Fs,Fc):
    """
    Integrate DAS data to convert strain rate to strain.

    Parameters
    ----------
    X : numpy.ndarray
        Differential data.
    Fs : float
        Sampling frequency (Hz).
    Fc : float
        Integration cutoff (Hz) in user settings.

    Returns
    -------
    Y : numpy.ndarray
        Integrated data.

    """
    m = 1-Fc*2*np.pi/Fs

    Y = deepcopy(X)

    nt = X.shape[0]
    for t in range(1,nt):
        Y[t,:] = m*Y[t-1,:]+X[t,:]

    return Y


#%% Sweep Correlation Utils

def fMakeSweep(sweep_length, dt, f1, f2, tap_length):
    """[Generates sweep signal with given parameters]

    Args:
        sweep_length ([float]): sweep length (s)
        dt ([float]): [sampling interval (s)]
        f1 ([float]): [starting frequency (Hz)]
        f2 ([float]): finishing frequency (Hz)
        tap_length (float): taper length (s)

    Returns:
        [T]: [time vector (np.float)]
        [S]: [sweep signal (np.float)]

    Example:
        T,S = fMakeSweep(5,0.001,10,100,0.5)
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        ax.plot(T, S)  # Plot some data on the axes.
        plt.show()

    """

    T=np.arange(0,sweep_length,dt)
    taper=0.5*(1-np.cos(np.pi*np.arange(0,tap_length,dt)/tap_length))
    dfdt = (f2-f1)/T[-1]
    S = np.cos(np.multiply(2*np.pi*T,f1+T*dfdt/2))
    S[0:len(taper)] = np.multiply(S[0:len(taper)],taper)
    S[(len(S)-(len(taper))):len(S)] = np.multiply(S[(len(S)-(len(taper))):len(S)],np.flipud(taper))
    return T,S

def fCor(R,sweep):
    """
    Perform sweep correlation.

    Parameters
    ----------
    R : numpy.ndarray
        Input data.
    sweep : numpy.ndarray (1D)
        Sweep (can be generated using fMakeSweep).

    Returns
    -------
    Rd : numpy.ndarray
        Data after correlations with sweep.

    Notes
    -----
    Converted from Matlab by GuXH
    """

    nnp=np.size(R,0)
    ntr=np.size(R,1)
    #[nnp,ntr] = size(R);

    sweep_spec = np.fft.fft(sweep, nnp)#sweep_spec = np.fft.fft(sweep[:,0],n=nnp)
    #sweep_spec = fft(single(sweep),nnp);

    sweep_spec[np.arange(math.floor(nnp/2),nnp-1,1)]=0#np.zeros[nnp-math.floor(nnp/2)]
    #sweep_spec(floor(nnp/2):nnp)=0;

    amp_sw_spec=abs(sweep_spec)
    #amp_sw_spec = abs(sweep_spec);

    pha_sw_spec = np.angle(sweep_spec)
    #pha_sw_spec = (angle(sweep_spec));


    S2 = np.fft.fft(R,n=None,axis=0)#S2 = np.fft.fft(R,n=None,axis=0)
    #S = fft(single(R),[],1);

    S2[np.arange(math.floor(nnp/2),nnp,1),:]=0
    #S(floor(nnp/2):nnp,:) = 0;

    for k in range(0,ntr): #ntr??
    #for k = 1:ntr
        Phase = np.angle(S2[:,k])-pha_sw_spec.T
        #Phase = angle(S(:,k)) - (pha_sw_spec)';

        S2[:,k]=abs(S2[:,k])*amp_sw_spec.T*np.exp(1j*Phase)
        #S(:,k) = abs(S(:,k)).*amp_sw_spec'.*exp(1i*Phase);

    Rd = np.fft.irfft(S2[0:math.floor(nnp/2)],n=nnp,axis=0)
     #Rd = ifft(S,[],1,'symmetric');
    return Rd

#%% Signal to Noise Ratio Utils

def fCalcFrameSNR(D,percent,maxlag_np):

    ntr = np.size(D[0,:])
    nsmp = np.size(D[:,0])

    g_max = np.zeros(ntr-1,dtype='float')
    ccf_lags = signal.correlation_lags(nsmp,nsmp)
    gv=(np.abs(ccf_lags)<maxlag_np)
    for n in range(ntr-1):
        ccf=signal.correlate((D[:,n]-D[:,n].mean())/(D[:,n].std()*nsmp),(D[:,n+1]-D[:,n+1].mean())/D[:,n+1].std(),'full')
        g_max[n] = np.max(ccf[(gv)])

    SNR_vec = np.sqrt(g_max/(1-g_max))

    return stats.trim_mean(SNR_vec,percent/100)

def fCalcSN42Traces(tr1, tr2, window):
    # computes S/N ratio for a pair of traces
    # window - in samples
    # maxlag - maximum lag for CCF computation
    # interp - 1: interpolate when looking for maximum, 0: not

    trh = 0.005  # i.e. if energy in the window is less than 0.001 of predicted,
                 # the window will be rejected

    nsmp = len(tr1)
    if tr1.ndim != 1 or tr2.ndim != 1:
        SN = np.nan
        print('must be single traces')
        return SN

    w2 = window // 2



    median_val = np.mean([np.median(np.abs(tr1)), np.median(np.abs(tr2))])
    tr_lev = (median_val**2) * trh

    # tr1 = tr1 + median_val * (0.5 - np.random.rand(len(tr1))) * 0.001
    # tr2 = tr2 + median_val * (0.5 - np.random.rand(len(tr1))) * 0.001

    SN = np.zeros_like(tr1)

    for n in range(nsmp):
        s1 = max(n - w2, 0)
        s2 = min(n + w2, nsmp-1)
        nn = s2-s1+1;
        tt1 = np.sum(tr1[s1:s2+1]**2)
        tt2 = np.sum(tr2[s1:s2+1]**2)

        if tt1 < tr_lev * (s2-s1+1) or tt2 < tr_lev * (s2-s1+1):
            continue

        #ccf = signal.correlate(tr1[s1:s2+1], tr2[s1:s2+1], mode='valid') / (window * tt1 * tt2)
        ccf=signal.correlate((tr1[s1:s2+1]-tr1[s1:s2+1].mean())/(tr1[s1:s2+1].std()*nn),(tr2[s1:s2+1]-tr2[s1:s2+1].mean())/tr2[s1:s2+1].std(),'same',method='fft')

        g_max = np.max(ccf)

        if 0 < g_max < 1:
            SN[n] = np.sqrt(g_max / (1 - g_max))


    return SN, g_max

def fCalcSNR2DFrame(D,window):
    nsmp,ntr = np.shape(D)
    SNR2D = np.zeros((nsmp,ntr-1))

    w2 = window // 2
    for n in range(nsmp):
        s1 = max(n - w2, 0)
        s2 = min(n + w2, nsmp-1)
        nn = s2-s1+1
        L = nn*2-1
        Dd = D[s1:s2+1,:]
        Dd = Dd - np.mean(Dd,axis=0)
        D0 = Dd[:,0:ntr-1]
        D1 = Dd[:,1:ntr]

        R = np.real(ifft(fft(D0,L,axis=0)*fft(np.flipud(D1.conj()),L,axis=0),axis=0))
        R/=np.std(D0,axis=0)*np.std(D1,axis=0)*nn
        g_max_v = np.max(R,axis=0)
        #print(np.shape(R))
        SNR2D[n,:] = np.sqrt(g_max_v/(1-g_max_v))

    return SNR2D

def fOrmsbASDesign(dt, nnp, fOrmsb):
    df = (1000.0 / (nnp * dt))  # for ms
    Fr = np.arange(0, (nnp - 1) * df + df, df)
    flt = np.zeros(nnp)

    flt[np.logical_and(Fr >= fOrmsb[1], Fr <= fOrmsb[2])] = 1
    nf = np.sum(np.logical_and(Fr >= fOrmsb[0], Fr < fOrmsb[1]))
    o = (np.cos(np.arange(0, nf) / nf * np.pi - np.pi) + 1) / 2
    flt[np.logical_and(Fr >= fOrmsb[0], Fr < fOrmsb[1])] = o

    nf = np.sum(np.logical_and(Fr > fOrmsb[2], Fr <= fOrmsb[3]))
    o = (np.cos(np.arange(0, nf) / nf * np.pi - np.pi) + 1) / 2
    flt[np.logical_and(Fr > fOrmsb[2], Fr <= fOrmsb[3])] = np.flip(o)

    return Fr, flt

def fOrmsbyBPFilter(R, dt, f1, f2, f3, f4, filt_mode):
    nnp, ntr = R.shape

    _, amp_fl_spec = fOrmsbASDesign(dt, nnp, [f1, f2, f3, f4])

    amp_fl_spec = np.array(amp_fl_spec, dtype=np.float32)
    amp_fl_spec = amp_fl_spec.reshape(1, nnp)

    if filt_mode < 0:#to make a notch filter set filt_mode to anything less than 0
        amp_fl_spec = 1 - amp_fl_spec

    S = np.fft.fft(np.array(R, dtype=np.float32), axis=0)
    S[nnp // 2:nnp, :] = 0

    for k in range(ntr):
        S[:, k] = S[:, k] * amp_fl_spec

    Rd = np.fft.ifft(S, axis=0)
    Rd = 2*np.real(Rd)


    return Rd
