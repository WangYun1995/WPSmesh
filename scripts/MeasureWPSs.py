import time
import numpy as np
from WPSmesh.main import WPSs
from nbodykit.lab import BigFileMesh
from nbodykit import CurrentMPIComm

comm = CurrentMPIComm.get()
rank = comm.rank

# Parameters used to specify the path
simulation = "/TNG"            # EAGLE, Illustris, SIMBA, TNG
run        = "/TNG100-1-Dark"  # hydro: RefL0100N1504, Illustris-1, m100n1024, TNG100-1, TNG300-1
                               # DMO: DMONLYL0100N1504, Illustris-1-Dark, m100n1024-DMO, TNG100-1-Dark, TNG300-1-Dark
redshift   = "/z0"
matter     = "dmo"             # hydro: tot, dm, ba; DMO: dmo

# Routine to print script status to command line, with elapsed time
def print_status(comm,start_time,message):
    if comm.rank == 0:
        elapsed_time = time.time() - start_time
        print('%d\ts: %s' % (elapsed_time,message))

# Load the mesh
if ( matter == 'tot' ):
   
   path_mesh  = '/.../...'+simulation+run+redshift+'/dens_fields/tot/dens_field.bigfile'
   mesh1      = BigFileMesh(path_mesh, 'Field', comm=comm)
   mesh2      = mesh1

elif ( matter == 'dmo' ):
   
   path_mesh  = '/.../...'+simulation+run+redshift+'/dens_fields/dmo/dens_field.bigfile'
   mesh1      = BigFileMesh(path_mesh, 'Field', comm=comm)
   mesh2      = mesh1

elif ( (matter == 'dm') or (matter == 'ba') ):
   
   path_mesh1 = '/.../...'+simulation+run+redshift+'/dens_fields/'+matter+'/dens_field.bigfile'
   path_mesh2 = '/.../...'+simulation+run+redshift+'/dens_fields/tot/dens_field.bigfile'
   mesh1      = BigFileMesh(path_mesh1, 'Field', comm=comm)
   mesh2      = BigFileMesh(path_mesh2, 'Field', comm=comm)

#--------------------------------------------------------
start_time = time.time()
print_status(comm,start_time,'Starting the calculations')
#--------------------------------------------------------

# Measure the WPSs from the mesh
Nscales   = 25
Ndens     = 8
bins_temp = np.geomspace(1e-1,1e+2,Ndens-1,endpoint=True) 
densbins  = np.pad( bins_temp, (1, 1), 'constant', constant_values=(0,1e+100) )
k_pseu, f_vol, env_WPS, global_WPS = WPSs( mesh1, mesh2, Nscales, densbins, kmax=0.4, wavelet=iso_cwgdw, comm=comm)

#--------------------------------------------------------
print_status(comm,start_time,'Done')
#--------------------------------------------------------

# Save results to the npz file
if (rank==0):
   np.savez("WPSs.npz", k_pseu=k_pseu, f_vol=f_vol, env_WPS = env_WPS, global_WPS = global_WPS)
