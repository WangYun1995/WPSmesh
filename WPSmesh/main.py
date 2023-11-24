import numpy as np
from scipy import stats
from nbodykit import CurrentMPIComm
from nbodykit.lab import FieldMesh
from pmesh.pm import ParticleMesh, RealField
from pmesh.window import Affine

comm = CurrentMPIComm.get()
rank = comm.rank

def iso_cwgdw(k_pseu):
    '''The isotropic cosine weighted Gaussian-derived wavelet in Fourier domain.
    '''
    def filter( k, v ):
        cw       = 0.3883323511065930
        cn       = 4.0*np.sqrt( np.sqrt(np.pi**3)/(9.0+55.0*np.e) )
        k2       = k[0]**2 + k[1]**2 + k[2]**2
        scale    = cw*k_pseu
        k2      /= scale**2 
        k        = np.sqrt(k2)
        wavelet  = cn*( np.exp(k-0.5*k2)*(k2-k) + np.exp(-0.5*k**2-k)*(k2+k) )
        wavelet /= np.sqrt(scale**3)
        return v*wavelet
    return filter

def iso_rmw(k_pseu):
    '''The isotropic real-valued Morlet wavelet in Fourier domain.
    '''
    def filter ( k, v ):
        cw       = 0.1222000884469385
        cn       = 8*(np.pi**(3/4))/np.sqrt(3*(1-44*np.exp(16)+43*np.exp(64)))
        k2       = k[0]**2 + k[1]**2 + k[2]**2
        scale    = cw*k_pseu
        k2      /= scale**2
        k        = np.sqrt(k2)
        wavelet  = cn*(-0.5*np.exp(-0.5*k2)+0.75*(np.exp(-8*k-0.5*k2)+np.exp(8*k-0.5*k2)))
        wavelet /= np.sqrt(scale**3)
        return v*wavelet
    return filter

def WPSs( mesh1, maskmesh, Nscales, densbins, kmax=0.4, wavelet=iso_cwgdw, comm=None ):
    '''The function used to compute the Wavelet Power Spectra (WPSs) of the mesh1.

    Inputs:
    -------
    mesh1:    the mesh object of which we want to compute the WPSs, which can be the
              dark matter, baryons, or total matter.
    maskmesh: the mesh object used to specify local density environments, i.e. the
              total matter density field.
    Nscales:  the number of scales.
    densbins: 1-D array-like the bin edges.
    kmax:     the maximum value of the pseudo wavenumber we take, by default 0.4; unit: k_Nyquist.
    comm:     the MPI communicator.

    Outputs:
    --------
    k_pseu:     1-D array-like with shape (Nscales,), 
                the pseudo wavenumbers which is equal to the wavelet scale divided by 
                c_w 
    f_vol:      1-D array-like with shape (len(densbins)-1,), 
                the fraction of the volume that is occupied by the local density 
                environment.
    env_WPS:    2-D array-like with shape (Nscales, len(densbins)-1),
                the environment-dependent wavelet power spectrum.
    global_WPS: 1-D array-like with shape (Nscales,), 
                the global wavelet power spectrum. 

    '''

    # Parameters
    Lbox      = mesh1.attrs['BoxSize'][0] # Unit: Mpc/h
    Nmesh     = mesh1.attrs['Nmesh'][0]   # Integer
    kf        = 2*np.pi/Lbox
    kNyq      = Nmesh*np.pi/Lbox
    k_pseu    = np.geomspace(kf, kmax*kNyq, Nscales)

    if ( (Lbox!=maskmesh.attrs['BoxSize'][0]) or (Nmesh!=maskmesh.attrs['Nmesh'][0]) ):
        raise Exception("The BoxSize and Nmesh should be the same between mesh1 and maskmesh.")
    
    # Compute the number of grids for each density environment
    maskfield          = maskmesh.compute(mode='real', Nmesh=Nmesh)
    maskfield_         = np.ravel(maskfield)
    Nvol_rank, _, _ = stats.binned_statistic(maskfield_, maskfield_, 'count', bins=densbins)

    # Initialze the env_WPS
    env_WPS_rank = np.empty( (Nscales, len(densbins)-1) )
    # Perform the CWT and compute the env-WPS
    cfield1 = mesh1.compute(mode='complex', Nmesh=Nmesh)
    cmesh1  = FieldMesh(cfield1)
    for i, k in enumerate(k_pseu):
        cwt_mesh                = cmesh1.apply(wavelet(k), mode='complex', kind='wavenumber')
        cwt_field               = cwt_mesh.compute(mode='real', Nmesh=Nmesh)
        cwt2                    = np.ravel(cwt_field**2)
        env_WPS_rank[i,:], _, _ = stats.binned_statistic(maskfield_, cwt2, 'sum', bins=densbins)

    if (comm==None):
        env_WPS    = env_WPS_rank/np.expand_dims(Nvol_rank, axis=0)
        f_vol      = Nvol_rank/Nmesh**3    
        global_WPS = np.sum( env_WPS_rank, axis=1 )/Nmesh**3

    else:
        # Reduction
        Nvol       = comm.allreduce(Nvol_rank)
        env_WPS    = comm.allreduce(env_WPS_rank)
        f_vol      = Nvol/Nmesh**3    
        global_WPS = np.sum( env_WPS, axis=1 )/Nmesh**3
        env_WPS    = env_WPS/np.expand_dims(Nvol,axis=0)

    return k_pseu, f_vol, env_WPS, global_WPS

def WCCs( mesh1, mesh2, maskmesh, Nscales, densbins, kmax=0.4, wavelet=iso_cwgdw, comm=None ):
    '''The function used to compute the Wavelet Cross-Correlation functions (WCCs) between 
       the mesh1 and mesh2.

    Inputs:
    -------
    mesh1:    the mesh object of which we want to compute the WPSs, which can be the
              dark matter, baryons, or total matter.
    maskmesh: the mesh object used to specify local density environments, i.e. the
              total matter density field.
    Nscales:  the number of scales.
    densbins: 1-D array-like the bin edges.
    kmax:     the maximum value of the pseudo wavenumber we take, by default 0.4; unit: k_Nyquist.
    comm:     the MPI communicator.

    Outputs:
    --------
    k_pseu:     1-D array-like with shape (Nscales,), 
                the pseudo wavenumbers which is equal to the wavelet scale divided by 
                c_w (see 'Yun Wang and Ping He 2022 ApJ 934 112')
    f_vol:      1-D array like with shape (len(densbins)-1,), 
                the fraction of the volume that is occupied by the local density 
                environment.
    env_WCC:    2-D array-like with shape (Nscales, len(densbins)-1),
                the environment-dependent wavelet cross-correlation function.
    global_WCC: 1-D array-like with shape (Nscales,), 
                the global wavelet cross-correlation function. 

    '''

    # Parameters
    Lbox      = mesh1.attrs['BoxSize'][0] # Unit: Mpc/h
    Nmesh     = mesh1.attrs['Nmesh'][0]   # Integer
    kf        = 2*np.pi/Lbox
    kNyq      = Nmesh*np.pi/Lbox
    k_pseu    = np.geomspace(kf, kmax*kNyq, Nscales)

    if ( (Lbox!=maskmesh.attrs['BoxSize'][0]) or (Nmesh!=maskmesh.attrs['Nmesh'][0]) ):
        raise Exception("The BoxSize and Nmesh should be the same between mesh1 and maskmesh.")
    
    # Compute the number of grids for each density environment
    maskfield       = maskmesh.compute(mode='real', Nmesh=Nmesh)
    maskfield_      = np.ravel(maskfield)
    Nvol_rank, _, _ = stats.binned_statistic(maskfield_, maskfield_, 'count', bins=densbins)

    # Initialze the env_WCC
    env_WCC_rank    = np.empty( (Nscales, len(densbins)-1) )
    global_WCC_rank = np.empty( Nscales )
    # Perform the CWT and compute the env-WPS
    cfield1 = mesh1.compute(mode='complex', Nmesh=Nmesh)
    cfield2 = mesh2.compute(mode='complex', Nmesh=Nmesh)
    cmesh1  = FieldMesh(cfield1)
    cmesh2  = FieldMesh(cfield2)
    for i, k in enumerate(k_pseu):
        cwt_mesh1               = cmesh1.apply(wavelet(k), mode='complex', kind='wavenumber')
        cwt_mesh2               = cmesh2.apply(wavelet(k), mode='complex', kind='wavenumber')
        cwt_field1              = cwt_mesh1.compute(mode='real', Nmesh=Nmesh)
        cwt_field2              = cwt_mesh2.compute(mode='real', Nmesh=Nmesh)
        xwt                     = np.ravel(cwt_field1*cwt_field2)
        env_WCC_rank[i,:], _, _ = stats.binned_statistic(maskfield_, xwt, 'sum', bins=densbins)
        global_WCC_rank[i]      = np.sum(xwt)

    if (comm==None):
        env_WCC    = env_WCC_rank/np.expand_dims(Nvol_rank, axis=0)
        f_vol      = Nvol_rank/Nmesh**3    
        global_WCC = global_WCC_rank/Nmesh**3

    else:
        # Reduction
        Nvol       = comm.allreduce(Nvol_rank)
        env_WCC    = comm.allreduce(env_WCC_rank)
        env_WCC   /= np.expand_dims(Nvol,axis=0)
        f_vol      = Nvol/Nmesh**3    
        global_WCC = comm.allreduce(global_WCC_rank)/Nmesh**3

    return k_pseu, f_vol, env_WCC, global_WCC

def WPSs_subbox( mesh1, maskmesh, Nscales, densbins, kmax=0.4, Nsub=2, wavelet=iso_cwgdw, comm=None):
    '''The function used to compute the Wavelet Power Spectra (WPSs) for the sub boxes of the mesh1.

    Inputs:
    -------
    mesh1:    the mesh object of which we want to compute the WPSs, which can be the
              dark matter, baryons, or total matter.
    maskmesh: the mesh object used to specify local density environments, i.e. the
              total matter density field.
    Nscales:  the number of scales.
    densbins: 1-D array-like the bin edges.
    kmax:     the maximum value of the pseudo wavenumber we take, by default 0.4; unit: k_Nyquist.
    Nsub:     the cubic root of number of subboxes in full box
    comm:     the MPI communicator.

    Outputs:
    --------
    k_pseu:         1-D array-like with shape (Nscales,), 
                    the pseudo wavenumbers which is equal to the wavelet scale divided by 
                    c_w (see 'Yun Wang and Ping He 2022 ApJ 934 112')
    f_vol_sub:      2-D array-like with shape (len(densbins)-1,Nsub**3), 
                    the fraction of the volume that is occupied by the local density 
                    environment for sub-boxes.
    env_WPS_sub:    3-D array-like with shape (Nscales, len(densbins)-1, Nsub**3),
                    the environment-dependent wavelet power spectrum for sub-boxes.
    global_WPS_sub: 2-D array-like with shape (Nscales,Nsub**3), 
                    the global wavelet power spectrum for sub-boxes. 

    '''

    # Parameters
    Lbox      = mesh1.attrs['BoxSize'][0] # Unit: Mpc/h
    Nmesh     = mesh1.attrs['Nmesh'][0]   # Integer
    kf        = 2*np.pi/Lbox
    kNyq      = Nmesh*np.pi/Lbox
    k_pseu    = np.geomspace(kf, kmax*kNyq, Nscales)

    if ( (Lbox!=maskmesh.attrs['BoxSize'][0]) or (Nmesh!=maskmesh.attrs['Nmesh'][0]) ):
        raise Exception("The BoxSize and Nmesh should be the same between mesh1 and maskmesh.")
    
    # Compute the number of grids for each density environment
    Nsub3       = Nsub**3
    Nvol_rank   = np.empty( (len(densbins)-1, Nsub3) )
    maskfield      = maskmesh.compute(mode='real', Nmesh=Nmesh)
    maskfield_subs = {}
    for i in range(Nsub):
        for j in range(Nsub):
            for k in range(Nsub):
                indx                    = subbox_multiindex_to_index((i,j,k),Nsub)
                maskfield_sub              = field_subbox_pm((i,j,k),Nsub,maskfield)
                maskfield_sub_             = np.ravel(maskfield_sub)
                Nvol_rank[:,indx], _, _ = stats.binned_statistic(maskfield_sub_, maskfield_sub_, 
                                                                 'count', bins=densbins)
                maskfield_subs[indx]       = maskfield_sub_           

    # Initialze the env_WPS
    env_WPS_rank = np.empty( (Nscales, len(densbins)-1, Nsub3) )
    # Perform the CWT and compute the env-WPS
    cfield1 = mesh1.compute(mode='complex', Nmesh=Nmesh)
    cmesh1  = FieldMesh(cfield1)
    for ii, kk in enumerate(k_pseu):
        cwt_mesh  = cmesh1.apply(wavelet(kk), mode='complex', kind='wavenumber')
        cwt_field = cwt_mesh.compute(mode='real', Nmesh=Nmesh)
        cwt2      = cwt_field**2
        for i in range(Nsub):
            for j in range(Nsub):
                for k in range(Nsub):
                    indx                          = subbox_multiindex_to_index((i,j,k),Nsub)
                    cwt2_sub                      = field_subbox_pm((i,j,k),Nsub,cwt2)
                    cwt2_sub_                     = np.ravel(cwt2_sub)
                    env_WPS_rank[ii,:,indx], _, _ = stats.binned_statistic(maskfield_subs[indx], cwt2_sub_, 
                                                                           'sum', bins=densbins)

    if (comm==None):
        env_WPS_sub    = env_WPS_rank/np.expand_dims(Nvol_rank, axis=0)
        f_vol_sub      = Nvol_rank/(Nmesh/Nsub)**3    
        global_WPS_sub = np.sum( env_WPS_rank, axis=1 )/(Nmesh/Nsub)**3

    else:
        # Reduction
        Nvol_sub       = comm.allreduce(Nvol_rank)
        env_WPS_sub    = comm.allreduce(env_WPS_rank)
        f_vol_sub      = Nvol_sub/(Nmesh/Nsub)**3 
        global_WPS_sub = np.sum( env_WPS_sub, axis=1 )/(Nmesh/Nsub)**3
        env_WPS_sub    = env_WPS_sub/np.expand_dims(Nvol_sub,axis=0)

    return k_pseu, f_vol_sub, env_WPS_sub, global_WPS_sub

def subbox_multiindex_to_index(multiindex,nsub_per_side):
    """Convert a subbox multi-index to a single index.
    
    Given a subbox multi-index of the form (a,b,c), return a single
    index corresponding to that, via i = a*N**2 + b*N + c
    where N = nsub_per_side.
    
    Parameters
    ----------
    multiindex : array_like
        tuple of form (a,b,c) that indexes subregion of each coordinate axis
    nsub_per_side : int
        cubic root of number of subboxes in full box
    
    Returns
    -------
    int
        single index corresponding to subbox referred to by `multiindex`
    
    Author:
    -------
    Simon Foreman, https://github.com/sjforeman/bskit/blob/master/bskit/main.py
    """
    assert(len(multiindex)==3)
    
    return int(multiindex[0]*nsub_per_side**2 + multiindex[1]*nsub_per_side + multiindex[2])

def field_subbox_pm(box_multiindex,
                    nsub_per_side,
                    source):
    """Construct a RealField corresponding to a subregion of an input RealField.
    
    The size of the subregion is Lsub = source.BoxSize/nsub_per_side, and the
    location of the subregion is specified by the coordinate origin of the
    subregion in units of Lsub by box_multiindex.
    
    For example, for box_multiindex=(1,0,0), the location of the box within
    the full box is ((Lsub,2Lsub),(0,Lsub),(0,Lsub)).
    
    Parameters
    ----------
    box_multiindex : array_like
        tuple of form (a,b,c) that indexes subregion of each coordinate axis
    nsub_per_side : int
        cubic root of number of subboxes in full box
    source : RealField
        field within full box that we want to cut the subbox from
    
    Returns
    -------
    RealField:
        field within requested subbox
    
    Author:
    -------
    Simon Foreman, https://github.com/sjforeman/bskit/blob/master/bskit/main.py
    """
    assert isinstance(source,RealField)

    # Define ParticleMesh and RealField with size and Nmesh of desired subregion
    sub_pm = ParticleMesh(BoxSize=source.BoxSize/nsub_per_side,
                          Nmesh=source.Nmesh/nsub_per_side)
    sub_field = RealField(sub_pm)
    
    # Get coordinate indices of the slab of the subbox grid stored on this task
    coords = sub_field.pm.mesh_coordinates(dtype='i4')

    # Find coordinate indices of origin of subregion coordinates within full region
    start_coord = box_multiindex*source.Nmesh/nsub_per_side
    
    # Define transformation between coordinates within subregion and coordinates
    # within full-region slab stored on this task. Specifically, the subregion
    # coordinates are shifted to the specific subregion within the full-region
    # (by adding start_coord), and then shifted back such that their origin
    # coincides with the origin of the full-region slab (by subtracting source.start)
    transform = Affine(sub_field.ndim,
                translate=start_coord-source.start,
                scale=1.0,
                period=source.Nmesh)

    # Find domain decomposition of desired subregion within full region,
    # to know which tasks ask for different coordinates. We need to manually add
    # start_coord, because the transform argument passed to the decompose routine
    # only shifts based on each slab's origin (it ignores the "translate" argument
    # specified above)
    layout_src = source.pm.decompose(coords+start_coord, smoothing=0, transform=transform)

    # Find values of source grid in subregion, as 1d list matched to 1d coords list
    vals = source.readout(coords, resampler='nnb', transform=transform, layout=layout_src)
    
    # Make new RealField corresponding to subregion. No "layout" argument is needed
    # because the coords list was fetched directly from the subbox slab on this task
    # i.e. each task will only paint its own values to the field
    return sub_field.pm.paint(coords, mass=vals, resampler='nnb', 
                               transform=sub_field.pm.affine_grid )
