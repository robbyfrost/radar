# --------------------------------------------------
# Name: proq_iq.py
# Author: J.W. Thiesing and Robby M. Frost
# Advanced Radar Research Center
# University of Oklahoma
# Created: 23 Oct 2025
# Purpose: Adapted code from Vitor Goede, 
# --------------------------------------------------
from iq_utils import *
import numpy as np
import glob, os
import warnings
warnings.filterwarnings("ignore")
# --------------------------------------------------
# parameters

# project directory
dproj = "/data/arrcwx/raxpol/20250605/D1/"
# directory holding IQ data
diq = f"{dproj}iq/"
# directory to output moment data
dmom = f"{dproj}moment/reproc/"

# lore accurate speed of light
c = 299792458


# desired azimuthal resolution (degrees)
des_az = 0.5
# set b_factor
if des_az < 1.0:
    b_factor = 2.0
else:
    b_factor = 1.0

# flag to perform global regression clutter filtering
# grf = True
# --------------------------------------------------
# read iq

# list of all files in drad
files = sorted(glob.glob(f"{diq}RAXPOL*.rkc"))

# loop over files
for jt in range(0,len(files)):
    # time string
    tstring = os.path.basename(files[jt])[7:-8]
    print(f"-----------------Starting on {tstring}-----------------\n")
    # read in rkc file
    rkid = rkcfile(files[jt], verbose=False)
    
    # waveform
    wf = rkid.header['config']['waveformName']
    print(f"Waveform: {wf}\n")
    if wf[0] == 'h':
        grf = False
    if wf[0] == 's':
        grf = True
    
    # pulse width [s]
    pw = rkid.header['config']['pw']

    # --------------------------------------------------
    # detect sweep boundaries via cumulative azimuth rotation

    az_full = rkid.pulses['azimuthDegrees']
    el_full = rkid.pulses['elevationDegrees']

    az_unwrapped = np.unwrap(az_full, period=360)
    total_rotation = az_unwrapped - az_unwrapped[0]

    if total_rotation[-1] <= 360:
        nswp = 1
    else:
        nswp = int(total_rotation[-1] // 360)
    # --------------------------------------------------
    # extract data
    for swp in range(nswp):
        pidx0 = swp * 360
        pidx1 = (swp+1) * 360
        
        mask = (total_rotation >= pidx0) & (total_rotation < pidx1)
        pidx = np.where(mask)[0]

        # iq
        pulses = rkid.pulses['iq'][pidx] # shape (pulse,iq,gate,hv)
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
        # sample
        sample_freq = int(rkid.header['waveform']['fs'])
        sample_time = 1/sample_freq
        # gate size
        dr = 30.
        # nyquist velocity
        va = lamb/(4*prt)

        # number of pulses
        npulse = pulses.shape[1]
        # rays per sweeps
        local_rotation = total_rotation[pidx] - total_rotation[pidx[0]]
        nray = int(local_rotation[-1] / des_az)
        if nray == 0:
            print(f"Skipping {tstring}, not enough pulses\n\n\n")
            continue
        # number of range gates
        ngate = pulses.shape[0]
        # pulses per ray
        ppr = npulse // nray
        # total number of pulses
        n = nray * ppr

        # azimuths
        az = rkid.pulses['azimuthDegrees'][pidx]
        # elevation angle
        el = rkid.pulses['elevationDegrees'][pidx]
        el_str = round(np.median(el), 0)
        # time
        time_sec  = rkid.pulses['time_tv_sec'][pidx].astype(np.float64)
        time_usec = rkid.pulses['time_tv_usec'][pidx].astype(np.float64)
        time = time_sec + time_usec * 1e-6
        dtime = (time_sec.astype('datetime64[s]') + time_usec.astype('timedelta64[us]'))

        # horizontal iq
        X_h = pulses[:,:n,0] # shape (gate, pulse)
        X_h = np.reshape(X_h, (ngate, nray, ppr)) # shape (gate, ray, pulse)
        # vertical iq
        X_v = pulses[:,:n,1]
        X_v = np.reshape(X_v, (ngate, nray, ppr)) # shape (gate, ray, pulse)

        # reshape azimuths
        az = np.reshape(az[:n], (nray,ppr))
        az_rad = np.deg2rad(az)
        az = np.rad2deg(np.arctan2(np.nanmean(np.sin(az_rad), axis=1), np.nanmean(np.cos(az_rad), axis=1))) % 360
        # reshape elevations
        el = np.reshape(el[:n], (nray, ppr))
        el = np.nanmean(el, axis=1)
        # reshape time
        time = np.reshape(time[:n], (nray, ppr))
        time = time[:,0]
        # reshape datetime
        dtime = np.reshape(dtime[:n], (nray, ppr))
        dtime = dtime[:,0]
        tstring = np.array([t.strftime('%Y%m%d_%H%M%S') for t in dtime.astype('datetime64[us]').tolist()])[0]

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
            # output filtered I/Q

            # range array
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
            dout = f'{diq}RAXPOL-GRFIQ-{tstring}-E{el_str}.nc'
            if os.path.exists(dout):
                os.remove(dout)
            ds.to_netcdf(dout, auto_complex=True)
            print(f"Output filtered I/Q to: {dout}\n")
        # --------------------------------------------------
        # get moments
        print("Getting moment data...")

        # set noise
        noise = rkid.header['config']['noise']
        N_h, N_v = noise

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
            dout = f"{dmom}RAXPOL-{tstring}-AZRES{des_az}-E{el_str}-GRF.nc"
        else:
            dout = f"{dmom}RAXPOL-{tstring}-AZRES{des_az}-E{el_str}.nc"
            
        # build radar object
        # dimensions/metadata
        rcf = raxpolCf()
        rcf.setVolume()
        rcf.setSweep()
        # rcf.setTime((rkid.pulses['time_tv_sec'][:n:ppr]).astype(np.float64))
        rcf.setTime(time)
        rcf.setRange(R.astype(np.float32))
        rcf.setPosition(np.nanmean(rkid.header['desc']['latitude']), np.nanmean(rkid.header['desc']['longitude']))
        rcf.setScanningStrategy('ppi')
        rcf.setTargetAngle(np.nanmean(rkid.pulses['elevationDegrees'][pidx]))
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
        print(f"-----------------Done with: {tstring}!!!-----------------\n\n\n")