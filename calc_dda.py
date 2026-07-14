# --------------------------------------------------
# Name: calc_dda_zlin_interp.py
# Author: Robby M. Frost
# Advanced Radar Research Center
# University of Oklahoma
# Created: 25 Jan 2026
# Purpose: Script to perform DDA with advection 
# corrected radar grids on a linear interpolated
# grid in the vertical. THIS IS THE CORRECTED VERSION
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
from copy import copy
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
# params

# path to radar files
dgrid_ktlx = "/data/arrcwx/robbyfrost/20100510_mwr/ktlx_grid/"
dgrid_oupr = "/data/arrcwx/robbyfrost/20100510_mwr/oupr_grid/"
ddda = "/data/arrcwx/robbyfrost/20100510_mwr/dda/"
dsm = "/data/arrcwx/robbyfrost/20100510_mwr/storm_attr/"

# horizontal grid spacing
delh = 200 # m

# linear interpolation in vertical
zlin = True
# --------------------------------------------------
# read in grids

# ktlx
ffktlx = sorted(glob.glob(dgrid_ktlx + f"advcor_barnes_{delh}m_*_zlin_interp_fixed.nc"))
fktlx = [f.split('/') for f in ffktlx]
fktlx = [f[-1] for f in fktlx]
acktlx = []
for ff in ffktlx:
    ds = xr.open_dataset(ff)
    acktlx.append(ds)
# oupr
ffoupr = sorted(glob.glob(dgrid_oupr + f"barnes_{delh}m_*_zlin_interp.nc"))
foupr = [f.split('/') for f in ffoupr]
foupr = [f[-1] for f in foupr]
goupr = []
for ff in ffoupr:
    grid = xr.open_dataset(ff)
    goupr.append(grid)
# time info
tstr_oupr = np.array([f[-30:-15] for f in foupr])
ndda = len(goupr)

# put ktlx into pyart format, dda
gktlx = []
for jt, (ac,oupr) in enumerate(zip(acktlx,goupr)):
    print(f"Starting on {tstr_oupr[jt]}...")
    # reference ktlx barnes grid
    gktlx_ref = pyart.core.Grid(
        time={'data': np.array([0.0])},
        fields={},  # <-- empty!
        metadata={},
        # origin_latitude={'data': np.array([goupr[0].radar_latitude['data'][0]])},
        # origin_longitude={'data': np.array([goupr[0].radar_longitude['data'][0]])},
        # origin_altitude={'data': np.array([goupr[0].radar_altitude['data'][0]])},
        origin_latitude={'data': np.array([35.18013763])}, #TODO: remove hard coding
        origin_longitude={'data': np.array([-97.43367004])},
        origin_altitude={'data': np.array([30.0])},
        radar_latitude={'data': np.array([35.3330574])},
        radar_longitude={'data': np.array([-97.27748108])},
        radar_altitude={'data': np.array([30.0])},
        x={'data': goupr[0].x.values, 'units': 'meters'}, #TODO: check this
        y={'data': goupr[0].y.values, 'units': 'meters'},
        z={'data': goupr[0].z.values, 'units': 'meters'},
        projection = {
            "proj": "pyart_aeqd",
            "lat_0": 35.3330574,
            "lon_0": -97.27748108,
            }
        )
    # copy and format
    ktlx = copy(gktlx_ref)
    ktlx.fields['reflectivity'] = {
        'data': ac.reflectivity.values[:,:,:,0],
        'units': 'dBZ',
        'long_name': 'Radar reflectivity',
        }
    ktlx.fields['velocity'] = {
        'data': ac.velocity.values[:,:,:,0],
        'units': 'm/s',
        'long_name': 'Radial velocity',
    }
    ktlx.time['units'] = f"seconds since {str(ac.time.values[0])[:-10]}Z"
    # pydda format
    ktlx = pydda.io.read_from_pyart_grid(ktlx)
    ktlx = pydda.initialization.make_constant_wind_field(ktlx, (0.0, 0.0, 0.0))

    # reference oupr barnes grid
    goupr_ref = pyart.core.Grid(
        time={'data': np.array([0.0])},
        fields={},  # <-- empty!
        metadata={},
        origin_latitude={'data': np.array([35.18013763])}, #TODO: remove hard coding
        origin_longitude={'data': np.array([-97.43367004])},
        origin_altitude={'data': np.array([30.0])},
        radar_latitude={'data': np.array([35.18013763])},
        radar_longitude={'data': np.array([-97.43367004])},
        radar_altitude={'data': np.array([30.0])},
        x={'data': goupr[0].x.values, 'units': 'meters'},
        y={'data': goupr[0].y.values, 'units': 'meters'},
        z={'data': goupr[0].z.values, 'units': 'meters'},
        projection = {
            "proj": "pyart_aeqd",
            "lat_0": 35.18013763,
            "lon_0": -97.43367004,
            }
        )
    # copy and format
    oupr = copy(goupr_ref)
    oupr.fields['reflectivity'] = {
        'data': goupr[jt].reflectivity.values,
        'units': 'dBZ',
        'long_name': 'Radar reflectivity',
        }
    oupr.fields['velocity'] = {
        'data': goupr[jt].velocity.values,
        'units': 'm/s',
        'long_name': 'Radial velocity',
    }
    oupr.time['units'] = f"seconds since {str(ac.time.values[0])[:-10]}Z" # ktlx time same as oupr
    # format for pydda
    oupr = pydda.io.read_from_pyart_grid(oupr)
    oupr = pydda.initialization.make_constant_wind_field(oupr, (0.0, 0.0, 0.0))

    # dda
    dda, _ = pydda.retrieval.get_dd_wind_field(
        [ktlx, oupr],
        Cm=256.0, Co=1, Cx=1, Cy=1, #TODO: set Co=1, lower smoothness?
        Cz=1, Cmod=0,
        refl_field='reflectivity',
        vel_name='velocity',
        wind_tol=0.5,
        mask_outside_opt=True,
        mask_w_outside_opt=True,
        low_pass_filter=False,
        engine='scipy',
        upper_bc=False,
    )
    
    # construct dataset
    u = dda[1].u.values
    v = dda[1].v.values
    w = dda[1].w.values
    vortz = dda[1].v.differentiate('x') - dda[1].u.differentiate('y')
    conv = -(dda[1].u.differentiate('x') + dda[1].v.differentiate('y'))
    ktlx_vel = dda[0].velocity.values
    ktlx_ref = dda[0].reflectivity.values
    oupr_vel = dda[1].velocity.values
    oupr_ref = dda[1].reflectivity.values
    x = dda[1].x.values
    y = dda[1].y.values
    z = dda[1].z.values
    time = dda[1].time.values
    ds = xr.Dataset(
        data_vars={
            'u': (('time','z','y','x'), u),
            'v': (('time','z','y','x'), v),
            'w': (('time','z','y','x'), w),
            'vortz': (('time','z','y','x'), vortz.values),
            'conv': (('time','z','y','x'), conv.values),
            'ktlx_ref': (('time','z','y','x'), ktlx_ref),
            'ktlx_vel': (('time','z','y','x'), ktlx_vel),
            'oupr_ref': (('time','z','y','x'), oupr_ref),
            'oupr_vel': (('time','z','y','x'), oupr_vel),
        },
        coords={
            'time': time,
            'z': z,
            'y': y,
            'x': x
        }
    )
    # output
    dout = f"{ddda}DDA_{delh}m_{tstr_oupr[jt]}_zlin_interp_fixed.nc"
    if os.path.exists(dout):
        os.remove(dout)
    ds.to_netcdf(dout)
    print(f"Output DDA to: {dout}\n")