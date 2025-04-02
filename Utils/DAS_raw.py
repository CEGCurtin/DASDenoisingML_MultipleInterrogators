import numpy as np
import datetime as dtime
import h5py
from struct import unpack

import logging

logging.basicConfig(level=logging.INFO)


#%% Silixa iDAS
def fLoadSilixaTDMS(fname,scan_size=2000):
    '''Function loads Silixa TDMS file
    v.0.2, Roman Pevzner
    
    scan_size - number of byts to scan for aquisition parameters
    
    D       - data
    mlength - measure length (m)
    dx      - chanel spacing (m)
    dt      - sampling interval (s)
    
    '''
    D,dx,mlength,dt=np.zeros((0,0),dtype='int16'),0,0,0
    SPAT_RES = b'SpatialResolution[m]'
    MES_LENGTH = b'MeasureLength[m]'
    SAMPL_FREQ = b'SamplingFrequency[Hz]'
    GPS_TIME = b'GPSTimeStamp'

    with open(fname,'rb') as f:
        f.seek(20,0)
        #unpack()
        hh=unpack('i',f.read(4))
        header_size=hh[0] + 28
        f.seek(0,0)
        sbuf = f.read(scan_size)
        f.seek(0,2)
        nbytes = f.tell()
        
        ss = sbuf.find(SAMPL_FREQ)+len(SAMPL_FREQ)+4
        col = sbuf[ss:ss+8]
        ss = sbuf.find(SPAT_RES)+len(SPAT_RES)+4
        col = col+sbuf[ss:ss+8]
        ss = sbuf.find(GPS_TIME)+len(GPS_TIME)+4
        col = col+sbuf[ss:ss+16]
        ss = sbuf.find(MES_LENGTH)+len(MES_LENGTH)+4
        col = col+sbuf[ss:ss+4]
        sfreq,dx,gt1,gt2,mlength = unpack('ddQqI',col) #(float64,float64,uint32,uint64,int64)
        dt = 1.0/sfreq
        
        ntr = int(mlength/dx)
        nsmp = int((nbytes - header_size)/(2*ntr))
        f.seek(header_size,0)
        #for n in range(nsmp):
        #    D[:,n] = np.fromfile(f,dtype='int16',count=ntr)
        D = np.fromfile(f,dtype='int16',count=int(nbytes/2))

    
    D = D.reshape(nsmp,ntr)
    np.transpose(D)
    return D,dx,mlength,dt

#%% Terra15 Treble
def fLoadTerra15HDF5(fname):
    """
    Import Terra15 Treble+ HDF5 files.

    Parameters
    ----------
    fname : string
        File name.

    Returns
    -------
    velocity : np.array
        Raw data array (n_timesamp x n_trace).
    dx : float
        Channel spacing [m].
    mlength : float
        Measured length [m].
    dt : float
        Sampling interval [s].
    """
    with h5py.File(fname, "r") as f:
        # Stored as 2D array with time increasing down rows
        velocity = f['data_product']['data'][:, :]
        metadata = dict(f.attrs)
        dt = metadata['dt_computer']
        dx = metadata['dx']
        nx = metadata['nx']
        mlength = dx*nx
    
    return velocity, dx, mlength, dt

#%% ASN OptoDAS
def fLoadOptoDASHDF5(fname):
    """
    Import ASN OptoDAS HDF5 raw files.

    Parameters
    ----------
    fname : string
        File name.

    Returns
    -------
    D : np.array
        Raw data array (n_timesamp x n_trace).
    dx : float
        Channel spacing [m] (does not account for decimation).
    mlength : float
        Measured length [m].
    dt : float
        Sampling interval [s].
    Tstart_UTC : datime.datetime
        Time of the first sample (UTC).
    LG :  float
        Gauge length [m].
    """
    with h5py.File(fname) as f:
        # Get data
        D = f['data'][:,:]
    
        # Store some header values
        dt = f['header']['dt'][()]
        dx = f['header']['dx'][()]
        LG = f['header']['gaugeLength'][()]
        dims = f['header']['dimensionSizes'][()]
        ntr = dims[1]
        Tstart_UTC = dtime.datetime.fromtimestamp(f['header']['time'][()], dtime.timezone.utc) #no conversion to GPS time
        start_chan = f['header']['dimensionRanges']['dimension1']['min'][()]
        end_chan = f['header']['dimensionRanges']['dimension1']['max'][()]
        mlength = dx*(ntr-1)*(end_chan[0]-start_chan[0])/(ntr-1) #different from Matlab, account for decimation - need to check

    return D, dx, mlength, dt, Tstart_UTC, LG  