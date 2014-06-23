#!/usr/local/bin/python
# Filename: jmkfigure.py

# from matplotlib import rc
from pylab import *

def djmkfigure(width,vext):
    """
    djmkfigure(width,vext):
    width is column widths, and vext is fractional 10 page height.  
    """
    wid = 3*width+3./8.;
    height = 10*vext;
    rc('figure',figsize=(wid,height),dpi=96)
    rc('font',size=9)
    rc('font',family='sans-serif');
    # rcParams['font.sans-serif'] = ['Verdana']
    rc('axes',labelsize='large') 
    leftin = 0.75
    rightin = 0.25
    botin = 0.4
    rc('figure.subplot',left=leftin/wid) 
    rc('figure.subplot',right=(1-rightin/wid)) 
    rc('figure.subplot',bottom=botin/height) 

def jmkprint(fname,pyname,dirname='doc'):
    """
    def jmkprint(fname,pyname)
    def jmkprint(fname,pyname,dirname='doc')
    """
    import os
    
    try:
        os.mkdir(dirname)
    except:
        pass

    if dirname=='doc':
        pwd=os.getcwd()+'/doc/'
    else:
        pwd=dirname+'/'
    savefig(dirname+'/'+fname+'.pdf',dpi=400)
    savefig(dirname+'/'+fname+'.png',dpi=400)
    
    fout = open(dirname+'/'+fname+'.tex','w')
    str="""\\begin{{figure*}}[htbp]
  \\begin{{center}}
    \\includegraphics[width=\\twowidth]{{{fname}}}
    \\caption{{
      \\tempS{{\\footnotesize {pwd}/{pyname} ;     
        {pwd}{fname}.pdf}}
      \\label{{fig:{fname}}} }}
  \\end{{center}}
\\end{{figure*}}""".format(pwd=pwd,pyname=pyname,fname=fname)
    fout.write(str)
    fout.close()
    
    cmd = 'less '+dirname+'/%s.tex | pbcopy' % fname
    os.system(cmd) 


def tsdiagramjmk(salt,temp,cls=[]):
    import numpy as np
    import seawater.gibbs as gsw
    import matplotlib.pyplot as plt
     
     
    # Figure out boudaries (mins and maxs)
    smin = salt.min() - (0.01 * salt.min())
    smax = salt.max() + (0.01 * salt.max())
    tmin = temp.min() - (0.1 * temp.max())
    tmax = temp.max() + (0.1 * temp.max())
 
    # Calculate how many gridcells we need in the x and y dimensions
    xdim = round((smax-smin)/0.1+1,0)
    ydim = round((tmax-tmin)+1,0)

     
    # Create empty grid of zeros
    dens = np.zeros((ydim,xdim))
     
    # Create temp and salt vectors of appropiate dimensions
    ti = np.linspace(1,ydim-1,ydim)+tmin
    si = np.linspace(1,xdim-1,xdim)*0.1+smin
     
    # Loop to fill in grid with densities
    for j in range(0,int(ydim)):
        for i in range(0, int(xdim)):
            dens[j,i]=gsw.rho(si[i],ti[j],0)
     
    # Substract 1000 to convert to sigma-t
    dens = dens - 1000
 
    # Plot data ***********************************************
    if not(cls==[]):
        CS = plt.contour(si,ti,dens, cls,linestyles='dashed', colors='k')
    else:
        CS = plt.contour(si,ti,dens,linestyles='dashed', colors='k')
        
    plt.clabel(CS, fontsize=9, inline=1, fmt='%1.2f') # Label every second level
    ax1=gca()
    #    ax1.plot(salt,temp,'or',markersize=4)
     
    ax1.set_xlabel('S [psu]')
    ax1.set_ylabel('T [C]')

#########################################################################

def facetpcolor(x,y,z,**kwargs):
    # keep y the same.  Expland x
    x0=1.*x
    [M,N]=shape(z)
    dx = diff(x)
    x = np.tile(x,(2,1))
    x[0,1:]=x[0,1:]-dx/2.
    x[0,0]=x[0,0]-dx[0]/2.
    x[1,:-1]=x[1,:-1]+dx/2
    x[1,-1]=x[1,-1]+dx[-1]/2

    z = np.tile(z,(1,1,2))
    [m,n]=shape(x)
    x=np.reshape(x.T,[2*n],order='C')
    z=np.reshape(z,[M,-1],order='C')
    zz = 0.*z
    zz[:,0:-1:2]=z[:,0:N]
    zz[:,1:-1:2]=z[:,(N):(2*N-1)]
    
    return pcolormesh(x,y,zz,**kwargs) 

#########################################################################

def pcolormeshRdBu(x,y,z,**kwargs):
    # def pcolormeshRdBu(x,y,z,**kwargs):
    return pcolormesh(x,y,z,rasterized=True,cmap=cm.RdBu_r,**kwargs)    

    
    