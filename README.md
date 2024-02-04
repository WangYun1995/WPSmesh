# WPSmesh
The Python module `WPSmesh` is built upon the [nbodykit](https://nbodykit.readthedocs.io/en/latest/). We use it to measure the environment-dependent Wavelet Power Spectrum (env-WPS) of the cosmic density field. 

## The contents of `WPSs(_sub).npz`
```Python
>>import numpy as np

# Specify the path
>>path = '/.../.../data/hydro/TNG300/'

# Load .npz files
>>WPSs     = np.load(path+WPSs.npz)      # measured from the full volume
>>WPSs_sub = np.load(path+WPSs_sub.npz)  # measured from the 8 sub-volumes

# View all the constituent files
>>WPSs.files
>>['k_pseu', 'f_vol', 'env_WPS', 'global_WPS']
>>WPSs_sub.files
>>['k_pseu', 'f_vol_sub', 'env_WPS_sub', 'global_WPS_sub']

>>k         = WPSs['k_pseu']     # The pseudo wavenumber, shape: (25,), unit: h/Mpc
>>fvol      = WPSs['f_vol']      # The volume fraction of the local density environment, shape: (8,)
>>envWPS    = WPSs['env_WPS']    # The total matter env-WPS, shape: (25,8), unit: (Mpc/h)^3
>>globalWPS = WPSs['global_WPS'] # The total matter global-WPS, shape: (25,), unit: (Mpc/h)^3

>>fvol_sub      = WPSs_sub['f_vol_sub']      # The volume fraction of the local density environment, shape: (8,8)
>>envWPS_sub    = WPSs_sub['env_WPS_sub']    # The total matter env-WPS, shape: (25,8,8), unit: (Mpc/h)^3
>>globalWPS_sub = WPSs_sub['global_WPS_sub'] # The total matter global-WPS, shape: (25,8), unit: (Mpc/h)^3
```

## References
- Wang, Yun, and Ping He. "How do baryonic effects on the cosmic matter distribution vary with scale and local density environment?" Monthly Notices of the Royal Astronomical Society, [2024;, stae229](https://doi.org/10.1093/mnras/stae229). 
- Hand, Nick, et al. "nbodykit: An open-source, massively parallel toolkit for large-scale structure." The Astronomical Journal [156.4 (2018): 160](https://iopscience.iop.org/article/10.3847/1538-3881/aadae0/meta).

## Acknowledgement

We especially thank [Dr. Yu Feng](https://github.com/rainwoodman) and [Dr. Simon Foreman](https://github.com/sjforeman) for their help.
