# ---------------------------------
# Name: radar_utils.py
# Author: Robby M. Frost
# Advanced Radar Research Center
# University of Oklahoma
# Created: 10 September 2024
# Purpose: Functions for using radar data

# Functions include:
# -- get_time_pyart
# -- plot_ppi_radar
# -- plot_ppi_barnes
# -- calc_vort_radar
# -- unet_dealias

# ---------------------------------
# imports

import pyart
import numpy as np
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import re
import sys

# ---------------------------------
# radar object time extract
def get_time_pyart(radar_time):
    """
    Format time data from Py-ART radar object.

    :param dict radar_time: Dictionary containing time information from the radar object (run radar.time)
    """
    tsf = re.split(r'[ \-:TZ]', radar_time['units'])[2:][:-1]
    yr = tsf[0]
    mo = tsf[1]
    da = tsf[2]
    hr = tsf[3]
    mi = tsf[4]
    se = tsf[5][:2]
    tstring = f"{yr}{mo}{da}_{hr}{mi}{se}"
    tdt = datetime.strptime(tstring, '%Y%m%d_%H%M%S')
    
    return yr, mo, da, hr, mi, se, tstring, tdt

# ---------------------------------
# plot ppi in radar coords
def plot_ppi_radar(radar, swp, xlims, ylims):
    """
    Plot reflectivity and radial velocity PPI 
    in radar coordinates

    :param dict radar: Py-ART radar object
    :param int swp: sweep index
    :param list xlims: x limits for plot
    :param list ylims: y limits for plot
    """
    # get time info
    yr, mo, da, hr, mi, se, tstring, tdt = get_time_pyart(radar.time)
    # plot
    display = pyart.graph.RadarDisplay(radar.extract_sweeps([swp]))
    fig, axs = plt.subplots(figsize=(16,10), ncols=2, constrained_layout=True, sharey=True, sharex=True)
    fig.suptitle(f"{radar.metadata['instrument_name']['data']} {yr}-{mo}-{da} {hr}:{mi}:{se} UTC (El={radar.elevation[swp]['data'][-1]}$^{{\\circ}}$)", 
                 fontsize=20, 
                 fontweight='bold', 
                 y=1)
    # reflectivity
    ax = axs[0]
    vmin, vmax = -20, 60
    display.plot_ppi('reflectivity', ax=ax, 
                    vmin=vmin, vmax=vmax,
                    cmap="pyart_Carbone42",
                    colorbar_flag=False,
                    title_use_sweep_time=True,
                    axislabels_flag=False)
    cbar0 = fig.colorbar(display.plots[0], ax=ax, orientation='horizontal', pad=0.025)
    cbar0.set_label("Reflectivity [dBZ]")
    cbar0.set_ticks(np.arange(vmin, vmax + 0.001, 10))
    ax.set_ylabel("Meridional Distance [km]")
    # radial velocity
    ax = axs[1]
    vmin, vmax = -50, 50
    display.plot_ppi('velocity', ax=ax, 
                    vmin=vmin, vmax=vmax,
                    cmap="pyart_Carbone42",
                    colorbar_flag=False,
                    title_use_sweep_time=True,
                    axislabels_flag=False)
    cbar1 = fig.colorbar(display.plots[1], ax=ax, orientation='horizontal', pad=0.025)
    cbar1.set_label("$V_r$ [m s$^{-1}$]")
    cbar1.set_ticks(np.arange(vmin, vmax + 0.001, 10))
    # clean up plot
    for iax in axs:
        iax.set_aspect('equal')
        iax.set_xlim(xlims)
        iax.xaxis.set_major_locator(MultipleLocator(5))
        iax.xaxis.set_minor_locator(MultipleLocator(1))
        iax.set_xlabel("Zonal Distance [km]")
        iax.set_ylim(ylims)
        iax.yaxis.set_major_locator(MultipleLocator(5))
        iax.yaxis.set_minor_locator(MultipleLocator(1))
        iax.grid(alpha=0.6)

        return fig, axs
    
# ---------------------------------
# plot gridded ppi
def plot_ppi_barnes(grid, elidx, xlims, ylims):
    """
    Plot Barnes interpolated reflectivity and radial velocity

    :param dataset grid: pyart grid object
    :param int elidx: elevation index
    :param list xlims: x limits for plot
    :param list ylims: y limits for plot
    """
    # fix messed up name
    if grid.instrument_name == 'RPUO':
        grid['instrument_name'] = 'OU-PRIME'
    # time info
    tdt = grid.time.values[elidx]
    # tdt = pd.to_datetime(tdt)
    yr = str(tdt.year)
    mo = str(tdt.month).zfill(2)
    da = str(tdt.day).zfill(2)
    hr = str(tdt.hour).zfill(2)
    mi = str(tdt.minute).zfill(2)
    se = str(tdt.second).zfill(2)

    # make fig
    fig, axs = plt.subplots(figsize=(15,8), ncols=2, constrained_layout=True, sharey=True)
    fig.suptitle(f"{grid.instrument_name.values} {yr}-{mo}-{da} {hr}:{mi}:{se} UTC (El={grid.elevation[elidx].values}$^{{\\circ}}$)", 
                 fontsize=20, 
                 fontweight='bold', 
                 y=1)
    # reflectivity
    ax = axs[0]
    pcm1 = ax.pcolormesh(grid.x/1e3, grid.y/1e3, grid.reflectivity[elidx,:,:],
                            cmap='pyart_Carbone42',
                            vmin=-10, vmax=70)
    cbar = fig.colorbar(pcm1, ax=ax, 
                        orientation='horizontal', 
                        pad=0.01)
    cbar.set_label('$Z$ [dBZ]')
    ax.set_title(f"Equivalent reflectivity factor")
    ax.set_ylabel('Meridional Distance [km]')
    # radial velocity
    ax = axs[1]
    pcm1 = ax.pcolormesh(grid.x/1e3, grid.y/1e3, grid.velocity[elidx,:,:],
                            cmap='pyart_Carbone42',
                            vmin=-50, vmax=50)
    cbar = fig.colorbar(pcm1, ax=ax, 
                        orientation='horizontal', 
                        pad=0.01)
    cbar.set_label('$V_r$ [m s$^{-1}$]')
    ax.set_title(f"Radial velocity")

    for ax in axs:
        ax.set_aspect('equal')
        ax.set_xlim(xlims)
        ax.set_xlabel('Zonal Distance [km]')
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.set_ylim(ylims)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.grid(alpha=0.6)

    return fig, axs

# ---------------------------------
# calculate inferred vertical vorticity
def calc_vort_radar(radar, vel_dea_str, vortz_str, az_sm_idx, r_sm_idx, repl=False):
    """
    Calculate inferred vertical vorticity in radar coordinates

    :param radar: pyart radar object
    :param str vel_dea_str: string for dealiased velocity
    :param str vortz_str: string for output vorticity
    :param int smooth_idx: number of azimuth/range indices to smooth vr over
    """

    vortz_full = np.zeros_like(radar.fields[vel_dea_str]['data'])
    for swp in range(radar.nsweeps):
        # extract sweep
        radswp = radar.extract_sweeps([swp])
        # extract fields
        az = radswp.azimuth['data']
        r = radswp.range['data']
        vr = radswp.fields[vel_dea_str]['data']
        # smooth vr
        vr_da = xr.DataArray(vr, dims=["az", "r"])
        vr = vr_da.rolling(az=az_sm_idx,r=r_sm_idx, center=True, min_periods=1).median().values
        # calculate vorticity
        vortz = np.zeros_like(vr)
        vortz = (1/r) * np.gradient(vr, np.deg2rad(az), axis=0)

        # add to sweep
        vortz_field = {
            'data': vortz,
            'units': '/s',
            'long_name': 'Inferred vertical vorticity from azimuthal shear',
            'standard_name': 'Vertical vorticity',
        }
        radswp.add_field(vortz_str, vortz_field, replace_existing=repl)
        # combine sweeps
        if swp == 0:
            azidx = az.size
            vortz_full[:azidx,:] = vortz
        else:
            vortz_full[azidx:azidx+az.size,:] = vortz
            azidx = azidx + az.size
    # add to radar object
    vortz_full_field = {
        'data': vortz_full,
        'units': '/s',
        'long_name': 'Inferred vertical vorticity',
        'standard_name': 'Inferred vertical vorticity',
    }
    radar.add_field(vortz_str, vortz_full_field, replace_existing=repl)

    return radar

# ---------------------------------
# ml dealiasing algorithm
def unet_dealias(vda, radar, nyquist, vel_str, vel_dea_str, rep_bad):
    """
    Dealias radial velocity using the UNet model (Veillette et al., 2023)
    TODO: Fix high resolution data (currently rotates data since I'm just dumping extra gates)

    :param vda: Velocity dealiaser model from U-Net (load in from file in their git repository)
    :param radar: pyart radar object
    :param float nyquist: nyquist velocity in m/s
    :param str vel_str: string for velocity field in radar object
    :param str vel_dea_str: string for output dealiased velocity
    :param rep_bad: boolean to replace bad data with NaN
    """
    import tensorflow as tf
    tstring = get_time_pyart(radar.time)[6]
    keys = radar.fields.keys()

    # prep velocity field for model
    vel = radar.fields[vel_str]['data'][:,:1152] # --> (naz,1152)
    # Reshape nyquist velocity to model input shape
    nyq = np.array(nyquist, dtype=np.float32)[None,None,None]  # --> (1,1,1)
    # retrieve radar dimensions
    az = radar.azimuth['data']
    r = radar.range['data']
    # replace bad data if desired
    if rep_bad:
        vel[vel <= -64] = np.nan
    # match model input shape (batch, nframes, naz, nr, 1)
    vel = np.expand_dims(vel, axis=[0,3]) # --> (1,naz,1152,1)
    vel = np.expand_dims(vel, axis=0) # --> (1,1,naz,1152,1)
    
    # make sure there are 360 azimuths
    naz_orig = vel.shape[2]
    # lower azimuthal resolution or sector scan
    if naz_orig < 360:
        az_pad = 360 - vel.shape[2]
        vel = np.pad(vel, ((0,0),(0,0),(0,az_pad),(0,0),(0,0)), mode='constant', constant_values=np.nan)
        repeated = False
    # repeated elevation rapid scan
    if (naz_orig > 360) and ((np.abs(az[-2] - az[-1])) > 0.9):
        vel = vel[:,:,-360:,:,:]
        repeated = True
    else:
        repeated = False
    # finer azimuthal resolution
    if (naz_orig >= 720 and (repeated is False)):
        s = vel.shape
        nb = s[2] // 360
        vel = vel[:,:,:int(360*nb),:,:]
        s = vel.shape
        vel = np.reshape(vel, (s[0], s[1], s[2]//nb, nb, s[3], s[4]))
        vel = np.transpose(vel, (0,3,1,2,4,5))
        vel = np.reshape(vel, (nb*s[0],s[1],s[2]//nb,s[3],s[4]))
        nyq = np.repeat(nyq, nb, axis=0)
        recombine = True
    else:
        recombine=False
    # less range gates than L2
    nr = vel.shape[3]
    if nr < 1152:
        padw = 1152 - nr
        vel = np.pad(vel, ((0,0),(0,0),(0,0),(0,padw),(0,0)), mode="constant", constant_values=np.nan)
        rpad = True
    else:
        rpad = False

    # pad azimuth for some reason (periodic wrap)
    pad_deg = 12
    vel = np.concatenate(
        (vel[:,:,-pad_deg:,:,:], vel, vel[:,:,:pad_deg,:,:]),
        axis=2)
    
    # run model
    inp = {'vel': tf.constant(vel), 'nyq': tf.constant(nyq)}
    out = vda.predict(inp)
    veld = out['dealiased_vel']

    # Recombine to finer azimuthal resolution if needed
    if recombine:
        s = veld.shape
        veld = np.reshape(veld, (s[0]//nb, nb, s[1], s[2], s[3]))
        veld = np.transpose(veld, (0,2,1,3,4))
        veld = np.reshape(veld, (s[0]//nb, -1, s[2], s[3]))
    
    # return to original shape to add to radar object
    veld = veld[0,:,:,0] # Remove batch and channel dimensions
    veld = veld[pad_deg:-pad_deg,:] # Remove azimuth padding (undo periodic wrap)
    if naz_orig < 360:
        veld = veld[:naz_orig,:]
    # pad range
    if not rpad:
        veldshape = veld.shape
        tshape = radar.fields[vel_str]['data'].shape
        padw = tshape[1] - veldshape[1]
        veld = np.pad(veld, ((0,0), (0,padw)), mode='constant', constant_values=np.nan)
        veldshape = veld.shape
    if rpad:
        veld = veld[:,:nr]
        veldshape = veld.shape
    # pad azimuth if needed
    if recombine:
        padw = tshape[0] - veldshape[0]
        if padw >= 0:
            veld = np.pad(veld, ((0,padw), (0,0)), mode='constant', constant_values=np.nan)
            veldshape = veld.shape
        else:
            veld = veld[:tshape[0],:]
            veldshape = veld.shape
    
    # add metadata for pyart
    veld_field = {
        'data': veld,
        'units': 'm/s',
        'long_name': 'Dealiased radial velocity',
        'standard_name': 'Dealiased radial velocity',
    }
    
    # match shape of veld if repeated elevation rapid scan
    if repeated:
        del veld_field
        print("Repeated elevation rapid scan")
        # empty radar object
        rade = pyart.testing.make_empty_ppi_radar(veldshape[1], veldshape[0], 1)
        for key in keys:
            # make fields
            field = {
                'data': radar.fields[key]['data'][-360:,:veldshape[1]],
                'units': radar.fields[key]['units'],
                # 'long_name': radar.fields[key]['long_name'],
                'standard_name': radar.fields[key]['standard_name']
            }
            # add field to new radar object
            rade.add_field(key, field, replace_existing=True)
            # add dealiased velocity field
            veld_field = {
                'data': veld[-360:,:],
                'units': 'm/s',
                'long_name': 'Dealiased radial velocity',
                'standard_name': 'Dealiased radial velocity',
            }
            # add dealiased velocity to radar object
            rade.add_field(vel_dea_str, veld_field, replace_existing=True)
        # add other data
        rade.metadata = radar.metadata
        rade.instrument_parameters = radar.instrument_parameters
        rade.longitude = radar.longitude
        rade.latitude = radar.latitude
        # add time
        tfield = {
            'data': radar.time['data'][-360:],
            'units': radar.time['units'],
            'long_name': radar.time['long_name'],
            'standard_name': radar.time['standard_name'],
            'calendar': radar.time['calendar']
        }
        rade.time = tfield
        # add azimuth
        azfield = {
            'data': radar.azimuth['data'][-360:],
            'units': radar.azimuth['units'],
            'long_name': radar.azimuth['long_name'],
            'standard_name': radar.azimuth['standard_name'],
            'axis': radar.azimuth['axis']
        }
        rade.azimuth = azfield
        # add range
        rfield = {
            'data': radar.range['data'],
            'units': radar.range['units'],
            'long_name': radar.range['long_name'],
            'standard_name': radar.range['standard_name'],
            'spacing_is_constant': radar.range['spacing_is_constant'],
            'meters_to_center_of_first_gate': radar.range['meters_to_center_of_first_gate'],
            'meters_between_gates': radar.range['meters_between_gates'],
            'axis': radar.range['axis']
        }
        rade.range = rfield
        # add elevation
        elfield = {
            'data': radar.elevation['data'][-360:],
            'units': radar.elevation['units'],
            'long_name': radar.elevation['long_name'],
            'standard_name': radar.elevation['standard_name'],
            'axis': radar.elevation['axis']
        }
        rade.elevation = elfield

        # replace original radar object with new one
        del radar
        radar = rade
    else:
        print("Not repeated elevation rapid scan")
        # add to radar object
        radar.add_field(vel_dea_str, veld_field, replace_existing=True)

    return radar

# # --------------------------------------------------
# # Adjust grid to RFGF
# # --------------------------------------------------
# def RFGF_adjust(GFpos, grid, tstring, dnc):
#     """
#     Adjust y coords of Barnes interpolated 
#     radar data to be relative to RFGF. Grid
#     must be rotated to mean surface wind direction.

#     :param GFpos: gust front position
#     :param grid: pyart rotated grid object
#     :param str tstring: time string
#     :param str dnc: output data directory
#     """
#     print("Starting RFGF adjustment...")
    
#     # RFGF positions
#     xGF = GFpos[:,0][::-1]
#     yGF = GFpos[:,1][::-1]
#     # grid positions
#     x = grid.x.values / 1e3
#     y = grid.y.values / 1e3

#     # adjust y coords
#     xidx = np.zeros_like(xGF, dtype=int)
#     yidx = np.zeros_like(yGF, dtype=int)
#     # loop over GF positions
#     for i in range(xGF.size):
#         xidx[i] = np.argmin(np.abs(x - xGF[i]))
#         yidx[i] = np.argmin(np.abs(y - yGF[i]))
#     xGFfull = x[xidx[-1]:xidx[0]]
#     xGFidxFull = np.arange(xidx[-1],xidx[0]).astype(int)
#     yGFfull = np.interp(xGFfull, xGF[::-1], yGF[::-1])
#     # fields along gust front azimuths
#     refGF = grid.reflectivity[:,:,xGFidxFull]
#     vrGF = grid.velocity[:,:,xGFidxFull]
#     vortzGF = grid.vorticity[:,:,xGFidxFull]
#     # distance from RFGF in streamwise direction
#     yadj = np.zeros((xGFfull.size, grid.y.size))
#     for jx in range(xGFfull.size):
#         yadj[jx] = y - yGFfull[jx]

#     # make dataset
#     ds = xr.Dataset(
#         {
#             "reflectivity": (["el", "y", "x"], refGF.values, {"long_name": "Reflectivity intersecting the RFGF", "units": "dBZ"}),
#             "velocity": (["el", "y", "x"], vrGF.values, {"long_name": "Velocity intersecting the RFGF", "units": "m/s"}),
#             "vorticity": (["el", "y", "x"], vortzGF.values, {"long_name": "Vorticity intersecting the RFGF", "units": "/s"}),
#             # "yadj": (["y", "x"], yadj.T, {"long_name": "Distance from RFGF in streamwise direction", "units": "km"}),
#             "xGF": (["x"], xGFfull, {"long_name": "X-coordinates of RFGF", "units": "km"}),
#             "yGF": (["x"], yGFfull, {"long_name": "Y-coordinates of RFGF", "units": "km"}),
#         },
#         coords={
#             "elevation": (["el"], grid.elevation.values, {"description": "Radar elevation angles", "units": "degrees"}),
#             "y": (["y"], y, {"description": "Meridional distance from radar", "units": "km"}),
#             "x": (["x"], xGFfull, {"description": "Zonal distance from radar", "units": "km"}),
#             "yadj": (["y", "x"], yadj.T, {"long_name": "Distance from RFGF in streamwise direction", "units": "km"}),
#         },
#         attrs={
#             "time": tstring,
#         }
#     )

#     return ds
# # --------------------------------------------------
# # Calculate power spectra
# # --------------------------------------------------
# def calc_power_spectra_1d(grid, var, ylim, xlim, yv, xv, zero_pad, hanning):
#     """
#     Calculate power spectra of spanwise 
#     vertical vorticity and radial velocity 
#     on each elevation and y position.

#     :param grid: rotated pyart grid object
#     :param str var: variable to calculate power spectra for ('vorticity' or 'velocity')
#     :param tuple ylim: y limits of grid from draw_inflow_bounds
#     :param tuple xlim: x limits of grid from draw_inflow_bounds
#     :param float yv: y vortex position in km
#     :param float xv: z vortex position in km
#     :param bool zero_pad: flag to zero pad space series for FFT (2*nx)
#     :param bool hanning: flag to apply Hanning window to time series before FFT
#     """
#     xmin, xmax = xlim[0], xlim[1]
#     ymin, ymax = ylim[0], ylim[1]
#     # params
#     delx = grid.x[1].values - grid.x[0].values
#     nel = grid.elevation.size
#     # subset data
#     sub0 = grid[var].sel(x=slice((xmin+xv)*1e3, (xmax+xv)*1e3), 
#                          y=slice((ymin+yv)*1e3, (ymax+yv)*1e3))
#     # get dimensions
#     nx, ny = sub0.x.size, sub0.y.size
#     nyquist = nx//2
#     # remove the mean
#     sub0_mean = sub0.mean(dim=["x", "y"], skipna=True)
#     sub0 = sub0 - sub0_mean

#     # frequency
#     freq = np.fft.rfftfreq(nx*2, d=delx) if zero_pad else np.fft.rfftfreq(nx, d=delx)
#     # wavelengths
#     wlen = 1 / freq[1:]
#     # angular frequency
#     omega = 2 * np.pi * freq[1:]
#     delta = omega[1] - omega[0]
#     # calculate white noise spectrum
#     rho = 0
#     sigma_e2 = 1
#     power_wn = ((4. * sigma_e2 / nx) / (1 + 2 * rho * np.cos(omega)))

#     # arrays to store
#     fft_raw = np.empty((ny,nx+1)) if zero_pad else np.empty((ny,nx//2+1))
#     power_sm = np.empty((nel,ny,freq.size))
#     power_scaled = np.empty((nel,ny,freq.size))
#     power_scaled_mean = np.empty((nel,freq.size))

#     # loop over elevations
#     for jel in range(nel):
#         # subset data
#         sub = sub0.isel(elevation=jel)
#         sub = sub.fillna(0)
#         if hanning:
#             w = np.hanning(sub.x.size)
#             sub = sub * w
#         # calculate FFT at each y position
#         for jy in range(ny):
#             if zero_pad:
#                 fft_raw[jy,:] = np.fft.rfft(sub.values[jy,:], n=sub.x.size*2)
#             else:
#                 fft_raw[jy,:] = np.fft.rfft(sub.values[jy,:])
#         # power spectra
#         power_raw = np.abs(fft_raw)**2
#         # put into xr
#         power_raw = xr.DataArray(power_raw, dims=["y","freq"], coords={"y":sub.y.values, "freq":freq})
#         # rolling mean over frequency
#         power_sm[jel,:,:] = power_raw.rolling(freq=10, center=True, min_periods=1).mean().values
#         # scale power spectrum
#         for jy in range(ny):
#             power_scaled[jel,jy,:] = power_sm[jel,jy,:] / np.nansum(delta*power_sm[jel,jy,:])
#         # spatial average
#         power_scaled_mean[jel,:] = np.nanmean(power_scaled[jel,:,:], axis=0)

#     # scale power spectrum
#     wn_scaled = power_wn / np.nansum(delta*power_wn)
#     # significance curve
#     alpha = 0.05
#     alpha_star = alpha / nyquist
#     dof = 2 * ny * 10
#     chi2 = stats.chi2.isf(alpha_star, dof)
#     sig_curve = (wn_scaled / dof) * chi2

#     # xr dataset
#     spectra_ds = xr.Dataset(
#         {
#             "power_smoothed": (["elevation", "y", "freq"], power_sm[:,:,1:]),
#             "power_scaled": (["elevation", "y", "freq"], power_scaled[:,:,1:]),
#             "power_scaled_mean": (["elevation", "freq"], power_scaled_mean[:,1:]),
#             "wn_scaled": (["freq"], wn_scaled),
#             "sig_curve": (["freq"], sig_curve),
#             "wavelength": (["freq"], wlen),
#         },
#         coords={
#             "freq": freq[1:],
#             "elevation": grid.elevation.values,
#             "y": sub0.y.values,
#             "time": grid.time,
#         },
#     )
#     return spectra_ds
# # --------------------------------------------------
# # Spectra time average
# # --------------------------------------------------
# def spectra_time_avg(dnc, tstring):
#     """
#     Calculate time average of power spectra and pad
#     arrays to account for different elevations and 
#     spectral analysis area

#     :param str dnc: output data directory
#     :param str tstring: time string
#     """
#     print("Starting time average of power spectra...")

#     # read in data
#     sall = []
#     for file in os.listdir(dnc):
#         if file.startswith(f"power_spectrum_{tstring}") and file.endswith(".nc"):
#             sall.append(xr.open_dataset(f"{dnc}/{file}"))

#     # account for different elevations/spectral analysis area
#     nel = sall[0].elevation.size
#     ny = sall[0].y.size
#     ntime = len(sall)
#     max_nwlen = max(s.freq.size for s in sall)
#     power_scaled = np.empty((ntime, nel, ny, max_nwlen))
#     sig_curve_full = np.empty((ntime, max_nwlen))

#     for jt in range(ntime):
#         nwlen = sall[jt].freq.size
#         cny = sall[jt].y.size
#         cnel = sall[jt].elevation.size
#         # Determine padding for elevation and y dimensions
#         pad_elev = max(0, nel - cnel)  # Padding for elevation
#         pad_y = max(0, ny - cny)       # Padding for y dimension
#         pad_freq = max(0, max_nwlen - nwlen)  # Padding for frequency
#         # Create the padded array
#         pwr_scl = np.pad(
#             sall[jt].power_scaled.values,
#             pad_width=((pad_elev, 0), (0, pad_y), (0, 0)),  # Pad elevation and y dimensions
#             mode='constant',
#             constant_values=np.nan
#         )
#         # Store pwr_scl for each time step
#         power_scaled[jt,:cnel,:cny,:nwlen] = pwr_scl[:cnel,:cny,:nwlen]
#         for jel in range(nel):
#             for jy in range(ny):
#                 power_scaled[jt,jel,jy,:nwlen] = pwr_scl[jel,jy]
#         # padded array of significance curves
#         sig_curve = np.pad(
#             sall[jt].sig_curve.values,
#             pad_width=((0, pad_freq)),
#             mode='constant',
#             constant_values=np.nan
#         )
#         sig_curve_full[jt,:nwlen] = sall[jt].sig_curve.values
#     # time averaged spectra
#     power_scaled_time_avg = np.nanmean(power_scaled, axis=0)
#     # time averaged significance curve
#     sig_curve_time_avg = np.nanmean(sig_curve_full, axis=0)

#     # xr dataset
#     spectra_ds = xr.Dataset(
#         {
#             "power_scaled_time_avg": (["el", "y", "wavelength"], power_scaled_time_avg),
#             "sig_curve_time_avg": (["wavelength"], sig_curve_time_avg),
#         },
#         coords={
#             "wavelength": (["wavelength"], sall[0].wavelength.values),
#             "el": sall[0].elevation.values,
#             "y": sall[0].y.values,
#         },
#     )

#     return spectra_ds

# # --------------------------------------------------
# # Identify inflow wind direction
# # --------------------------------------------------
# def inf_wind_dir(radar, elidx=0):
#     """
#     Manually draw inflow wind direction based
#     on worm orientation

#     :param grid: pyart grid object
#     :param int elidx: elevation index
#     """
#     matplotlib.use('TkAgg')
#     # plot
#     fig, ax = plot_ppi_radar(radar, elidx, xlims=[-10,10], ylims=[-5,15])
#     klicker = clicker(ax[1], ["start_end_point"], markers=["o"], colors='black')
#     plt.show()
#     coords = klicker.get_positions()
#     # Get RFGF points
#     p1, p2 = coords["start_end_point"][0], coords["start_end_point"][1]
#     x1, y1 = p1[0], p1[1]
#     x2, y2 = p2[0], p2[1]
#     # Calculate angle
#     angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
#     # Convert angle to met coords
#     met_angle = (270 - angle) % 360

#     return met_angle