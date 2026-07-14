# --------------------------------------------------
# Name: clutter_removal.py
# Author: J.W. Thiesing and Robby M. Frost
# Advanced Radar Research Center
# University of Oklahoma
# Created: 23 Oct 2025
# Purpose: Adapted code from Vitor Goede, 
# --------------------------------------------------
from grf_functions import *
import numpy as np
import glob, os
import warnings
warnings.filterwarnings("ignore")
# --------------------------------------------------
# parameters

# project directory
dproj = "/data/arrcwx/raxpol/20250518/"
# directory holding IQ data
drad = f"{dproj}iq/"
# directory to output GRF moment data
dgrf = f"{dproj}grf/"
# directory to output non-GRF moment data
dnogrf = f"{dproj}no_grf/"

# desired azimuthal resolution (degrees)
des_az = 0.25
# set b_factor
if des_az < 1.0:
    b_factor = 2.0
else:
    b_factor = 1.0

# flag to perform global regression clutter filtering
grf = True
# --------------------------------------------------
# read iq

# list of all files in drad
files = os.listdir(drad)

# loop over files
for jt in range(1,len(files),1):
    # file name
    file = files[jt]
    # time string
    tstring = file[7:-8]
    print(f"\nStarting on {tstring}...\n")
    # full path to rkc
    frkc = drad + file
    # read in rkc file
    rkid = rkcfile(frkc, verbose=False)
    # --------------------------------------------------
    # extract data

    # iq
    pulses = rkid.pulses['iq'] # shape (pulse,iq,gate,hv)
    # reshape
    pulses = pulses.transpose((1,2,3,0)) # shape (iq,gate,hv,pulse)
    # combine signals
    pulses = pulses[0,:,:,:] + 1.j*pulses[1,:,:,:] # shape (gate,hv,pulse)
    # reshape again
    pulses = pulses.transpose((0,2,1)).astype(np.complex128) # shape (gate, pulse, hv)

    # wavelength
    lamb = rkid.header['desc']['wavelength']
    # pulse repitition time
    prt = rkid.header['config']['prt']
    # lore accurate speed of light
    c = 299,792,458
    # sample
    sample_freq = int(rkid.header['waveform']['fs'])
    sample_time = 1/sample_freq
    # nyquist velocity
    va = lamb/(4*prt)

    # number of pulses
    npulse = pulses.shape[1]
    # rays per sweeps
    nray = int(360/des_az)
    # number of range gates
    ngate = pulses.shape[0]
    # pulses per ray
    ppr = npulse // nray
    if ppr == 0:
        print(f"Skipping {tstring}, not enough pulses\n")
        continue
    # total number of 
    n = nray * ppr

    # azimuths
    az = rkid.pulses['azimuthDegrees']
    # elevation angle
    el = rkid.pulses['elevationDegrees']

    # horizontal iq
    X_h = pulses[:,:n,0] # shape (gate, pulse)
    X_h = np.reshape(X_h, (ngate, nray, ppr)) # shape (gate, ray, pulse)
    # vertical iq
    X_v = pulses[:,:n,1]
    X_v = np.reshape(X_v, (ngate, nray, ppr)) # shape (gate, ray, pulse)

    # reshape azimuths
    az = np.reshape(az[:n], (nray, ppr))
    az = az[:,0]
    # reshape elevations
    el = np.reshape(el[:n], (nray, ppr))
    el = np.nanmean(el, axis=1)
    # --------------------------------------------------
    # perform GRF if desired

    if grf:
        print("Performing GRF...")
        # set some radar params
        wavelength_scalar = lamb
        num_samples_actual = X_h.shape[2]

        # filter the I/Q time series for the horizontal and vertical polarizations
        filter_inst = GroundClutterFilter(
            wavelength = lamb,
            scan_rate = 1/(ppr * prt),
            prt = prt,
            num_samples = ppr
        )

        # print("\n--- Filtering X_h (Horizontal Polarization) ---")
        # run function
        filtered_i_h, filtered_q_h, _, _, _, _ = filter_inst.filter_iq_data(
            X_h.real, 
            X_h.imag, 
            cnr_db_map=None, 
            apply_interpolation=True,
            b_factor=b_factor)
        # complex signal
        filtered_X_h_optimized = filtered_i_h + 1j * filtered_q_h

        # print("\n--- Filtering X_v (Vertical Polarization) ---")
        # run function
        filtered_i_v, filtered_q_v, _, _, _, _ = filter_inst.filter_iq_data(
            X_v.real, 
            X_v.imag,
            cnr_db_map=None, 
            apply_interpolation=True,
            b_factor=b_factor)
        # complex signal
        filtered_X_v_optimized = filtered_i_v + 1j * filtered_q_v

        print("Done with GRF!\n")

    # --------------------------------------------------
    # output filtered I/Q if desired

    # range array
    dr = 30.
    R = np.arange(0, ngate) * dr

    # make dataset
    ds = xr.Dataset(
        data_vars={
            'Xh': (('range', 'azimuth', 'pulse'), filtered_X_h_optimized),
            'Xv': (('range', 'azimuth', 'pulse'), filtered_X_v_optimized),
        },
        coords={
            'range': R,
            'azimuth': az,
            'pulse': np.arange(ppr)
        }
    )
    # output
    dout = f'{drad}GRF-RAXPOL-{tstring}.nc'
    if os.path.exists(dout):
        os.remove(dout)
    ds.to_netcdf(dout, auto_complex=True)
    print(f"Output filtered I/Q to: {dout}")
    # --------------------------------------------------
    # get moments
    print("Getting moment data...")

    # set noise
    noise = rkid.header['config']['noise']
    N_h, N_v = noise 

    if rkid.header['buildNo'] >= 4:
        if rkid.header['dataType'] == 'raw':
            dr = rkid.header['config']['pulseGateSize']
            # dt = dr * 2 / (c/1e6)
        elif rkid.header['dataType'] == 'compressed':
            dr = rkid.header['desc']['pulseToRayRatio'] * rkid.header['config']['pulseGateSize']
            # dt = dr * 2 / (c/1e6)
        else:
            print("Inconsistency detected. This should not happen.")
            dr = 30.
            # dt = 1./50
    else:
        dr = 30.
        # dt = 1./50

    # range array
    R = np.arange(0, ngate) * dr
    # Calibrations
    C = rkid.header['config']['ZCal']
    Cd = rkid.header['config']['DCal']
    Cp = rkid.header['config']['PCal']

    # actually get moments
    if grf:
        moments = get_moments(filtered_X_h_optimized, filtered_X_v_optimized, N_h, N_v, R, va, C, Cd, Cp)
    else:
        moments = get_moments(X_h, X_v, N_h, N_v, R, va, C, Cd, Cp)
    print("Done getting moments!\n")
    # --------------------------------------------------
    # build radar object and output

    # full output string
    if grf:
        dout = f"{dgrf}RAXPOL-{tstring}-GRF-MOMENT-{des_az}AZRES-{b_factor}BF.nc"
    else:
        dout = f"{dnogrf}RAXPOL-{tstring}-MOMENT-{des_az}AZRES.nc"
        
    # build radar object
    # dimensions/metadata
    rcf = raxpolCf()
    rcf.setVolume()
    rcf.setSweep()
    rcf.setTime((rkid.pulses['time_tv_sec'][:n:ppr]).astype(np.float64))
    rcf.setRange(R.astype(np.float32))
    rcf.setPosition(np.nanmean(rkid.header['desc']['latitude']), np.nanmean(rkid.header['desc']['longitude']))
    rcf.setScanningStrategy('ppi')
    rcf.setTargetAngle(np.nanmean(rkid.pulses['elevationDegrees']))
    rcf.setAzimuth(az)
    rcf.setElevation(el)
    rcf.setPulseWidthSeconds((rkid.pulses['pulseWidthSampleCount'][:n:ppr]*sample_time).astype(np.float32))
    rcf.setPrtSeconds(np.tile(prt, (nray,1)))
    rcf.setWavelengthMeters(np.tile(lamb, (nray,1)))
    # moment data
    rcf.setDBZ(moments['DBZ'])
    rcf.setVEL(moments['VEL'])
    rcf.setWIDTH(moments['WIDTH'])
    rcf.setZDR(moments['ZDR'])
    rcf.setRHOHV(moments['RHOHV'])
    rcf.setPHIDP(moments['PHIDP'], units='degrees')
    rcf.setSNRH(moments['SNRH'])
    rcf.setSNRV(moments['SNRV'])

    # output to cfradial
    print(f"Outputting moment data to: {dout}")
    if os.path.exists(dout):
        os.remove(dout)
    rcf.saveToFile(dout)
    print(f"Done with: {tstring}!!!\n\n\n")