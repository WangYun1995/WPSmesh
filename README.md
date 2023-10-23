# WPSmesh
`WPSmesh` is the Python module that we used to measure the environment-dependent Wavelet Power Spectrum (env-WPS) of the cosmic density field. 

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

>>k         = WPSs['k_pseu']     # The pseudo wavenumber, shape: (25,)
>>fvol      = WPSs['f_vol']      # The volume fraction of the local density environment, shape: (8,)
>>envWPS    = WPSs['env_WPS']    # The total matter env-WPS, shape: (25,8)
>>globalWPS = WPSs['global_WPS'] # The total matter global-WPS, shape: (25,)

>>fvol_sub      = WPSs_sub['f_vol_sub']      # The volume fraction of the local density environment, shape: (8,8)
>>envWPS_sub    = WPSs_sub['env_WPS_sub']    # The total matter env-WPS, shape: (25,8,8)
>>globalWPS_sub = WPSs_sub['global_WPS_sub'] # The total matter global-WPS, shape: (25,8)
```

## References

## Acknowledgement

We especially thank Dr. Yu Feng and Dr. Simon Foreman for their help.
