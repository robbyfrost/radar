# --------------------------------------------------
# Name: calc_adv_cor.py
# Author: Robby M. Frost
# Advanced Radar Research Center
# University of Oklahoma
# Created: 11 Sep 2025
# Purpose: Script to advection correct radar grids
# and output to netCDF
# --------------------------------------------------
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import rc
import sys
import glob
import os
import contextlib
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy
sys.path.append('/home/robbyfrost/Analysis/TurbTor_Radar/')
# from functions import *
sys.path.append('/home/robbyfrost/Analysis/ADV_Cor/')
from ADVCor import *
# suppress pyart/pydda messages
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
with suppress_output():
    import pyart
    import pydda
# --------------------------------------------------
# parameters

# path to radar files
drad_ktlx = "/data/arrcwx/robbyfrost/20100510_mwr/ktlx_cfrad/"
dgrid_ktlx = "/data/arrcwx/robbyfrost/20100510_mwr/ktlx_grid/"
drad_oupr = "/data/arrcwx/robbyfrost/20100510_mwr/oupr_cfrad/"
dgrid_oupr = "/data/arrcwx/robbyfrost/20100510_mwr/oupr_grid/"
ddda = "/data/arrcwx/robbyfrost/20100510_mwr/dda/"
dsm = "/data/arrcwx/robbyfrost/20100510_mwr/storm_attr/"

# vertical interpolation
self_interp = True

# barnes parameters
dely, delx, delz = 200, 200, 100 # grid spacing in meters
Ly, Lx, Lz = 60000, 60000, 1100 # domain size in meters
ylim = (0,Ly)
xlim = (-Lx//2,Lx//2)
zlim = (0,Lz)
Ny, Nx, Nz = Ly//dely+1, Lx//delx+1, Lz//delz+1 # number of grid points
# --------------------------------------------------
# Find sweeps close in time

# OUPR
ffoupr = sorted(glob.glob(drad_oupr + "nexrad*"))
foupr = [f.split('/') for f in ffoupr]
foupr = [f[-1] for f in foupr]
# volume start time info
tstr_oupr = np.array([f[7:-11] for f in foupr])
toupr = np.array([datetime.strptime(tstr, "%Y%m%d_%H%M%S") for tstr in tstr_oupr], dtype="datetime64[ns]")
t0 = toupr[0]
t1 = toupr[-1]

# KTLX
ffktlx = sorted(glob.glob(drad_ktlx + "DEALIASED_KTLX*"))
fktlx = [f.split('/') for f in ffktlx]
fktlx = [f[-1] for f in fktlx]
# volume start time info
tstr_ktlx = np.array([f[14:-4] for f in fktlx])
tktlx = np.array([datetime.strptime(tstr, "%Y%m%d_%H%M%S") for tstr in tstr_ktlx], dtype="datetime64[ns]")
tktlx_idx = np.where((tktlx > (t0-np.timedelta64(5,'m'))) & (tktlx < (t1+np.timedelta64(5,'m'))))[0].astype(int)

# valid times for DDA
tktlx = tktlx[tktlx_idx]
tstr_ktlx = tstr_ktlx[tktlx_idx]

# --------------------------------------------------
# read in grids

gktlx, goupr = [], []

if not self_interp:
    ff_gktlx = sorted(glob.glob(f"{dgrid_ktlx}barnes_{delx}m*[!_zlin_interp].nc"))
    ff_goupr = sorted(glob.glob(f"{dgrid_oupr}barnes_{delx}m*[!_zlin_interp].nc"))
    # ktlx
    for f in ff_gktlx:
        grid = pyart.io.read(f)
        gktlx.append(grid)
    # oupr
    for f in ff_goupr:
        grid = pyart.io.read(f)
        goupr.append(grid)

if self_interp:
    ff_gktlx = sorted(glob.glob(f"{dgrid_ktlx}barnes_{delx}m_*_zlin_interp.nc"))
    ff_goupr = sorted(glob.glob(f"{dgrid_oupr}barnes_{delx}m_*_zlin_interp.nc"))
    # ktlx
    for f in ff_gktlx:
        grid = xr.open_dataset(f)
        gktlx.append(grid)
    # oupr
    for f in ff_goupr:
        grid = xr.open_dataset(f)
        goupr.append(grid)

# --------------------------------------------------
# Advection correction

# read storm motion first guess
sm = xr.open_dataset(f"{dsm}/storm_motion.nc")

# KTLX
for jt in range(len(goupr)):
    print(tstr_oupr[jt])
    time_oupr = toupr[jt]
    # find ktlx file before and after
    time_ktlx_bef = tktlx[tktlx < time_oupr][-1]
    time_ktlx_aft = tktlx[tktlx > time_oupr][0]
    # indices
    time_ktlx_bef_idx = np.where(tktlx == time_ktlx_bef)[0][0]
    time_ktlx_aft_idx = np.where(tktlx == time_ktlx_aft)[0][0]

    # set grids
    g1 = gktlx[time_ktlx_bef_idx]
    g2 = gktlx[time_ktlx_aft_idx]

    # time info for correction
    bigT = (time_ktlx_aft - time_ktlx_bef) / np.timedelta64(1, 's')
    offset = (time_oupr - time_ktlx_bef) / np.timedelta64(1, 's')
    bnt, bdt = None, None
    min_error = 9999
    for nt in range(2,15):
        dt = bigT / (nt-1)
        # nearest grid idx to oupr
        idx = round(offset / dt)
        tapprox = idx * dt
        error = np.abs(tapprox - offset)
        if error < min_error:
            min_error = error
            bnt, bdt = nt, dt
    time_adv = time_ktlx_bef + np.arange(bnt) * np.timedelta64(int(bdt), "s")
    tadv_idx = np.argmin(np.abs(time_oupr - time_adv))
    
    # advection first guess
    smt = sm.sel(time=time_ktlx_bef, method='nearest')
    ug = np.full((Nz,Ny,Nx), smt.u.values)
    vg = np.full((Nz,Ny,Nx), smt.v.values)
    wg = np.full((Nz,Ny,Nx), 0)
    # reflectivity
    if not self_interp:
        field1 = g1.fields['reflectivity']['data']#.fill(np.nan)
        field2 = g2.fields['reflectivity']['data']#.fill(np.nan)
    if self_interp:
        field1 = g1.reflectivity.values#.fill(np.nan)
        field2 = g2.reflectivity.values#.fill(np.nan)
    # get horizontal advection components
    ua, va, _, _ = ADV3D(
         field1,
         field2,
         ug,
         vg,
         wg,
         delx,
         dely,
         delz,
         bigT,
         bnt,
         bdt,
         beta=500,
         gamma=500,
         eta=500,
         nu=500,
         relax=1.75,
         under=0.25,
         itermainmax=50
    )
    # correct reflectivity
    ref = precomputed_ADV3D(
         field1,
         field2,
         ua,
         va,
         wg,
         delx,
         dely,
         delz,
         bdt,
         bnt
    )
    # correct radial velocity
    if not self_interp:
        field1 = g1.fields['corrected_velocity']['data']
        field2 = g2.fields['corrected_velocity']['data']
    if self_interp:
        field1 = g1.velocity.values
        field2 = g2.velocity.values
    vel = precomputed_ADV3D(
         field1,
         field2,
         ua,
         va,
         wg,
         delx,
         dely,
         delz,
         bdt,
         bnt
    )
    # index nearest OUPR
    ref = ref[:,:,:,tadv_idx]
    ref = ref[:,:,:,None]
    vel = vel[:,:,:,tadv_idx]
    vel = vel[:,:,:,None]

    # output to netcdf
    if not self_interp:
        zds = g1.z['data']
        yds = g1.y['data']
        xds = g1.x['data']
    if self_interp:
        zds = g1.z.values
        yds = g1.y.values
        xds = g1.x.values
    ds = xr.Dataset(
         data_vars={
              'reflectivity': (('z','y','x','time'), ref),
              'velocity': (('z','y','x','time'), vel),
         },
         coords={
              'z': zds,
              'y': yds,
              'x': xds,
              'time': np.array([time_oupr])
         }
    )
    if not self_interp:
        dout = f"{dgrid_ktlx}advcor_barnes_{delx}m_{tstr_oupr[jt]}.nc"
    if self_interp:
        dout = f"{dgrid_ktlx}advcor_barnes_{delx}m_{tstr_oupr[jt]}_zlin_interp_fixed.nc"
    if os.path.exists(dout):
        os.remove(dout)
    ds.to_netcdf(dout)
    print(f"Output to: {dout}")
