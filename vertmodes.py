from matplotlib import pylab


def vertModes(N2,dz,nmodes=0): 
    """" psi,phi,ce,z=vertModes(N2,dz,nmodes=0)
    
    Compute the vertical eigen modes of the internal wave solution on a flat bottom
    
    Parameters:
    ----------- 

    N2 : (M) is buoyancy frequency squared (rad^2/s^2) as an 1-D
         array.  If there are M values of N2, the first one is assumed
         to be at dz/2 deep, and the last one is H-dz/2 deep.  The
         water column is assumed to be H=M*dz deep.  No gaps are
         allowed, and N2>0 everywhere.       
    dz : is a single value, and the distance (in meters) between the N2 estimates
    nmodes : number of modes to return.  nmodes = 0 means return M-3 modes.
            
         
    Returns:
    --------
    psi : (M,M-2) is the vertical structure function at
         z=dz/2,3dz/2,2dz...,H-dz/2.  Note there is one extra value
         compared to N2 (ie there are M+1 values in depth). psi is
         normalized so that sum(psi^2 dz) = 1.  For internal waves,
         psi is approriate for velocity and pressure vertical
         structure.          
    phi : (M,M-2) is the vertical integral of psi (phi = int psi dz)
         and represents the vertical velocity structure.  It is
         interpolated onto the same grid as psi.
    ce : (M-2) is the non-rotating phase speed of the waves in m/s.
    z :  (M) is the vertical position of the psi and phi vector elements.
             
    Notes: 
    ------

    This solves 1/N**2 psi_{zz} + (1/ce**2)psi = 0 subject to a
    boundary condition of zero vertical velocity at the surface and
    seafloor.
  
    psi(0)=0 (rigid lid approx)
    psi(H)=0
    
    It is solved as an eigenvalue problem.  

    Also note that if 
             
    J. Klymak (Based on code by Sam Kelly and Gabe Vecchi)           
    """

    import numpy as np
    
    # First we are solving for w on dz,2dz,3dz...H-dz
    M = np.shape(N2)[0]-1
        
    if M>200:
        sparse = True
        if nmodes==0:
            nmodes = 100 # don't try too many eigenvectors in sparse mode...
    else:
        sparse = False
        if nmodes==0:
            nmodes = M-2
    
    N2mid = N2[:-1]+np.diff(N2)/2.
    # matrix for second difference operator
    D = np.diag(-2.*np.ones(M),0)
    D += np.diag(1.*np.ones(M-1),-1)
    D += np.diag(1.*np.ones(M-1),1)

    D=-D/dz/dz
    D = np.diag(1./N2mid).dot(D)
    ce,W = np.linalg.eig(D)
    # psi is such that sum(psi^2)=1 but we want sum(psi^2 dz)=1.
    W = W/np.sqrt(dz)
    ce = 1./np.sqrt(ce)
    ind=np.argsort(-ce)
    
    ce=ce[ind[:-2]]
    W=W[:,ind[:-2]]
    # zphi
    zphi = np.linspace(dz/2.,(M+1)*dz-dz/2.,M+1)

    # now get phi (w structure) on dz/2,3dz/2...
    phi = np.zeros((M+1,M+1-3))
    phi[0,:]=0.5*(W[0,:])
    phi[1:-1,:]=0.5*(W[:-1,:]+W[1:,:])
    phi[-1,:]=0.5*(W[-1,:])
    
    # Now get psi (u/p structure) on dz/2,3dz/2...
    psi = np.zeros((M+1,M+1-3))
    psi[0,:] = W[0,:]
    psi[1:-1,] = np.diff(W,axis=0)
    psi[-1,:] = -W[-1,:]
    
    A = np.sqrt(np.sum(psi*psi,axis=0)*dz)
    psi = psi/A
    phi = phi/A
    # flip sign so always same sign in psi at top:
    phi[:,psi[0,:]<0] *= -1
    psi[:,psi[0,:]<0] *= -1

    return psi,phi,ce,zphi


