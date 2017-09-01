#!/usr/bin/env python
import sys
import os
import os.path
import numpy as np
import json
import time
import multiprocessing
import glob 

from scipy.optimize    import curve_fit
from matplotlib.ticker import FormatStrFormatter


#=========================================================================||===#
FULL_PATH = os.path.realpath(__file__)
FULL_PATH = os.getcwd()
#print FULL_PATH 

path, filename = os.path.split(FULL_PATH)
#print path 

current_folder_path, current_folder_name = os.path.split(os.getcwd())
#print current_folder_path, current_folder_name

date = time.strftime("%Y%b%d")
date = date.upper()

n_cpus = multiprocessing.cpu_count()

#mySelf = sys.argv[0].split(".")[0]
mySelf = sys.argv[0].split("/")[-1].split(".")[0]

#-------------------------------------------------------------------------||---#
def PlotSimpleXY( *Plots, **kwargs):
  nPlotsX = len(Plots)
  nPlotsY = {}   
  ShapesY = {} 
  for i,plot in enumerate(Plots): 
    Ys         = plot[1]
    nPlotsY[i] = len(Ys) 
    ShapesY[i] = { type(ys):[ j for j,y in enumerate(Ys) if type(y)==type(ys)] for ys in Ys} 

  import matplotlib.pyplot   as plt
  import matplotlib.gridspec as gridspec
  from   matplotlib.ticker   import FormatStrFormatter

  import itertools
  marker = itertools.cycle(('o', 's', 'v')) 

  fig  = plt.figure()
  Grs  = gridspec.GridSpec(1,nPlotsX) 

  Subs = [ gridspec.GridSpecFromSubplotSpec(nPlotsY[i],1,subplot_spec=Grs[i],hspace=0.0) for i in ShapesY.keys() ]  
  Axs  = [ [plt.subplot(sub[j,0]) for j in range(nPlotsY[i])] for i,sub in enumerate(Subs) ]

  Figs = [] 
  for i1,v1 in ShapesY.iteritems(): 
    ps = Plots[i1]
    Xs = ps[0]
    Ys = ps[1] 

    ax = Axs[i1]
    for i2,v2 in v1.iteritems(): 
      for j in v2: 
        ax[j].set_ylabel("[%d,%d]"% (i1,j) )  
        #print i1,j
        ys = Ys[j]
        if type(ys)==np.ndarray:
           F = ax[j].plot(Xs, ys, label="%d"%(0), marker=marker.next() )
           Figs.append(F)
        else: 
           F = [ ax[j].plot(Xs,y,label="%d"%(l), marker=marker.next(), ms=7.0) for l,y in enumerate(ys)] 
           for f in F: Figs.append(f)

    ax[-1].set_xlabel("[%d]"% (i1) ) 
    [ plt.setp(a.get_xticklabels(), visible=False) for a in ax[0:-1] ]
    [ plt.setp(a.get_xticklabels(), fontsize=16  ) for a in ax[:] ]
    [ plt.setp(a.get_yticklabels(), fontsize=16  ) for a in ax[:] ]

    dx = 0.02 * ( Xs.max() - Xs.min() )
    [ plt.setp(a.set_xlim(Xs.min()-dx, Xs.max()+dx))     for a in ax[ :  ] ]
    [ a.yaxis.get_major_ticks()[ 0].label1.set_visible(False) for a in ax[ :  ] ]
    [ a.yaxis.get_major_ticks()[-1].label1.set_visible(False) for a in ax[ :  ] ]
    [ a.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) for a in ax[ :  ] ]

  [ a.grid()   for axs in Axs for a in axs ]
  [ a.legend() for axs in Axs for a in axs ]

  if len(Axs)>1: 
    [ [a.yaxis.tick_right()                for a in axs] for axs in Axs[-1:] ]
    [ [a.yaxis.set_label_position("right") for a in axs] for axs in Axs[-1:] ]
    [ [plt.setp(a.get_yticklabels(),fontsize=10) for a in axs] for axs in Axs[-1:] ]
    [ [a.set_yticks( a.get_yticks()[1:-1] ) for a in ax] for axs in Axs[-1:] ]

  #plt.show()

  import itertools
  Axs  = list(itertools.chain.from_iterable(Axs ))
  Figs = list(itertools.chain.from_iterable(Figs))
  #print "2", Figs  

  return plt, Axs, Figs # PlotSimpleXY  


#-------------------------------------------------------------------------||---#
getDicFromArray = lambda _A:{_u:np.nonzero(_A==_u)[0] for _u in np.unique(_A) }


#-------------------------------------------------------------------------||---#
def REeff( _Re_in, _L0, _U0, _Tinf, _Tw, C=0.5 ):
  #print _T0, 

  ## Phys. Fluids, Vol. 15, No. 7, July 2003  
  CONDU01  = lambda _T, _B1, _S1     : _B1 * _T**(3.0/2) / ( _T + _S1 * (1e-12)**(1.0/_T) )
  K01      = lambda _T: CONDU01(_T, 0.6325e-5, 245.4)

  SUTHE01  = lambda _T, _C1, _C2     : _C1 * _T**(3.0/2) / ( _T + _C2 )
  MU01     = lambda _T: SUTHE01(_T, 1.4500e-6, 110.0)

  RHO      = lambda _P,  _T          : _P  / ( _T * Rair / Mair  )
  MA       = lambda _V,  _T          : _V / np.sqrt( gamma * Rair / Mair * _T )
  VIS      = lambda _R, _rho, _L, _U : _rho * _U * _L / _R
  KAPPA    = lambda _mu, _cp, _Pr    : _mu * _cp / _Pr

  ## props = props( _Re_in, T_in ) 
  Pr       = 0.71
  Rair     = 8.314621
  Mair     = 0.0289531
  gamma    = 7.0/5.0
  P0       = 101325.00
  cp_in    = 1016.56
  T_in     = 288.15          # <--- T_in
  rho_in   = RHO( P0, T_in )
  mu_in    = VIS(_Re_in, rho_in, _L0, _U0 )
  k_in     = KAPPA(mu_in,cp_in,Pr)
  nu_in    = mu_in / rho_in

  ## props = props( _Tf )
  _Tf      = _Tinf + C * (_Tw - _Tinf)
  _muf     = (mu_in / MU01(T_in)) *  MU01(_Tf)
  _rhof    = RHO( P0, _Tf )
  _Reff    = _U0 * _L0 * _rhof / _muf
  _kf      = (k_in / K01(T_in)) *  K01(_Tf)
  _nuf     =  _muf / _rhof

  _kw      = (k_in / K01(T_in)) *  K01(_Tw)

  ## Nuf (Tf/T0)**-0.17 = A1 + B1 Ref**n1 
  ## (kw/kf) Nu (Tf/T0)**-0.17 = A1 + B1 ( nu0/nuf Re )**n1   
  _props   = [ _Re_in, _Tw/_Tinf, _Reff, _Tf/T_in, nu_in/_nuf, _kw/_kf ]
  return _props

#-------------------------------------------------------------------------||---#
def LookinForFile( fname ):
  path, filename = os.path.split( fname )
  #filename, file_extension = os.path.splitext( filename )

  import glob
  fname = glob.glob( filename )

#Aux  = [ os.path.split(fin) for fin in options.Files ]
#Aux  = { a[1]:glob.glob(a[0]) for a in Aux }
#Aux0 = [ k for k,v in Aux.iteritems() if len(v)==0 ]
#Aux1 = [ ["%s/%s" %(f,k) for f in v] for k,v in Aux.iteritems() ]
#Aux  = Aux0 + sorted( sum(Aux1,[]) )
#Aux  = sorted(Aux)
#print "# + ", Aux

  return sorted(fname) 

#-------------------------------------------------------------------------||---#
def GetExtremes( _Z ): 
    from scipy.signal import argrelextrema
    if 0 : 
      _IDsmin = argrelextrema(_Z, np.less)
      _IDsmax = argrelextrema(_Z, np.greater)
      _IDs    = np.sort( np.append(_IDsmin,_IDsmax) )
    else:
      _IDsmin = np.argmin(_Z)
      _IDsmax = np.argmax(_Z)
      _IDs    = np.sort( np.append(_IDsmin,_IDsmax) )

    return _IDs 


#-------------------------------------------------------------------------||---#
def GetAllZeros( _X, _Y ):
    from scipy.signal import argrelextrema
    from scipy        import optimize
    from scipy.misc   import derivative

    ## Interpolador 
    Y  = lambda x:  np.interp(x, _X[:], _Y[:])

    ## Maximos y minimos 
    IDsmin = argrelextrema(_Y, np.less)
    IDsmax = argrelextrema(_Y, np.greater)
    #print IDsmin, Wz[IDsmin]; print IDsmax, Wz[IDsmax]

    IDs    = np.append( IDsmin, IDsmax )
    IDs    = np.sort(IDs)
    #print IDs  
   
    if len(IDs)==0:
      print "#\tWARNING: len(IDs)==0 !!"
      #exit() 
      return np.zeros(0)

    Xex     = _X[IDs]
    Yex     = _Y[IDs]
    #print [ derivative(Y,x,dx=1e-6) for x in Xex ]

    zeroCrossings = np.where(np.diff(np.sign(Yex)))[0]
    ##print [ (vmin,vmax) for vmin,vmax in zip(Yex[zeroCrossings], Yex[zeroCrossings+1]) if vmin*vmax<=0.0 ] 
    ##print [ (Y(vmin),Y(vmax)) for vmin,vmax in zip(Xex[zeroCrossings], Xex[zeroCrossings+1]) ] 
    Zeros = [ optimize.brentq(Y,xleft,xright) for xleft,xright in zip(Xex[zeroCrossings], Xex[zeroCrossings+1]) if Y(xleft)*Y(xright)<=0.0 ]
    Zeros = np.array(Zeros)

    Check = [ (z,Y(z)) for z in Zeros if abs(Y(z))>1e-5 ]
    if( len(Check) ): print Check

    return np.sort(Zeros)

#-------------------------------------------------------------------------||---#
def GetLastCl( _Files ):
  _Cl   = [ np.loadtxt(fin,usecols=[1,6-1,5-1]) for fin in _Files ]
  _Cl   = [ [np.average(c[:,0]),np.sum(c[:,1]),np.sum(c[:,2])] for c in _Cl]
  _Cl   = np.array(_Cl)
  _Clr  = GetAllZeros(_Cl[:,0], _Cl[:,1])

  _nClr = _Clr.shape[0]
#  if _nClr<3:
#    print "# \t ERROR: nRoots not enough", _nClr
#    return np.zeros(0), np.zeros(0)

  if _nClr>=3:
    _Clr  = _Clr[-3:]
    print "# |_Cl roots:", _Clr

    _OK = np.logical_and( _Clr[0] <= _Cl[:,0], _Cl[:,0] <= _Clr[-1] )
    _OK = np.nonzero(_OK)[0]

    return _Cl, _OK
  else:
    #print _nClr, _Cl.shape  
    #np.savetxt("x.dat", _Cl)
    print "## \t WARNING: nRoots not enough: %d \n## \t gnuplot -e \"plot 'drag.dat' u 1:2  t 'Cl' w lp\"  \n" % _nClr

    _OK = [ _Cl.shape[0]-1 ]
    _OK = np.arange(_Cl.shape[0]).astype(int)

    return _Cl, _OK   


#-------------------------------------------------------------------------||---#
def GetStrouhal( _signal, dt=1.0, d0=None, v0=None):
    import warnings
    warnings.filterwarnings("ignore")

    Strouhal = lambda _f, _d , _v : _f * _d / _v

    n_sample = _signal.shape[0]
    dB    = np.abs(np.fft.fft( _signal ))
    dB    = 20 * np.log10(dB)
    _id   = np.argmax(dB)
    freqs = np.fft.fftfreq(n_sample+0, d=dt) # sample frequencies
    freq  = np.abs( freqs[_id] )
    if d0!=None and v0!=None :
      St    = Strouhal(freq,d0,v0)
     #print "  |_Fr:%f Hz, St:%f, dt:%e" % (freq,St,dt)    
      return St
    else:
      return freq

#-------------------------------------------------------------------------||---#
def KerDat( Keys=['Re =', 'Ma =', 'Pr =', 'Tr =']):
  Files = glob.glob("*.dat")
  Files = [ f for f in Files if "NASTIN_MODULE" in open(f).read() ]
  Dt    = False
  if len(Files)>0:
    Dt  = [ l.split()[-1] for l in open(Files[0]).readlines() if "TIME_STEP_SIZE" in l ]
    Dt  = float(Dt[0])

  Files = glob.glob("*.ker.dat")
  Files = [ f for f in Files if "Re" in open(f).read() ]
  Files = [ f for f in Files if "Ma" in open(f).read() ]
  Files = [ f for f in Files if "Pr" in open(f).read() ]

  PropDic = False
  if len(Files)>0:
    data = open(Files[0], "r")
    lines = data.readlines()
    data.close()

    Dict = {}
    for key in Keys:
      Dict[key] = None
      for line in lines:
        line = line.split("#")[0]
        line = line.replace("$", "")
        if(line.find(key)>0):
          line = line[:-1]
          line = line.split(key)
          line = line[1]
          line = line.replace(' ', '')
          Dict[key] = line

    Ma, Re, Pr, Tr = None,None,None,None
    U, Ti, rho, mu, L, mu, cp, k, Tw = None, None,None,None,None,None,None,None,None
    for k,v in Dict.items():
      if("Ma" in k):  Ma = eval(v)
      if("Re" in k):  Re = eval(v)
      if("Pr" in k):  Pr = eval(v)
      if("Tr" in k):  Tr = eval(v)

      v = v.replace("/", " ")
      v = v.replace("*", " ")
      v = v.replace("np.sqrt", " ")
      v = v.replace("(", " ")
      v = v.replace(")", " ")
      v = v.strip()
      v = v.split()
      if("Tr" in k): [Tw, Ti] = [eval(x) for x in v];
      if("Ma" in k): [U, dummy, dummy, dummy, dummy, Ti] = [eval(x) for x in v];
      if("Re" in k): [rho, U, mu, L] = [eval(x) for x in v];
      if("Pr" in k): [mu, cp, kappa] = [eval(x) for x in v];

  print "# |_Ma:%f, Re:%f, Pr:%f, Tr:%f" % ( Ma, Re, Pr, Tr )

  return { "Tw":Tw, "Ti":Ti, "U":U, "rho":rho, "mu":mu, "L":L, "cp":cp, "k":kappa, "Re":Re, "Ma":Ma, "Pr":Pr, "Tr":Tr, "dt":Dt}

#==============================================================================#

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-F', action='store', dest='Files',
                    default=[], type=str, nargs='+',
                    help='Files',
                    )
options = parser.parse_args()

#-------------------------------------------------------------------------||---#
#
if(
    options.Files == []
  ):
  parser.print_help()
  print
  sys.exit()
#-------------------------------------------------------------------------||---#
import glob
Aux  = [ os.path.split(fin) for fin in options.Files ]
Aux  = { a[1]:glob.glob(a[0]) for a in Aux }
Aux0 = [ k for k,v in Aux.iteritems() if len(v)==0 ]
Aux1 = [ ["%s/%s" %(f,k) for f in v] for k,v in Aux.iteritems() ]
Aux  = Aux0 + sorted( sum(Aux1,[]) )
Aux  = sorted(Aux)
print "# + ", Aux

Fout00 = open("statistics_AllVsTime.dat", "w")
STATIs = [ [] for i in range(4) ]  
for ifin,fin in enumerate(Aux):
  fin = "/".join([FULL_PATH,fin])
  os.chdir(fin)
  print "# |_'%s' " % fin

  #-----------------------------------------------------------------------||---#
  ## Read files 
  ## Cols 
  ## 1:Ids 2:time 3:X   4:Y  
  ## 5:Fx  6:Fy   7:Wz  8:dP 9:dT 
  Files    = LookinForFile("fsi_*.vtk")
  if len(Files)==0: 
    print "##\t WARNING: 0 'fsi_*.vtk' files !!\n"

    Files    = LookinForFile("pv_*.vtk")
    if len(Files)==0:    
      exit("##\tERROR: 0 'pv_*.vtk' files !!\n")
    else:
      FilesDic = { f:f.replace(".vtk", "").replace("pv_", "") for f in Files }
  else: 
    FilesDic = { f:f.replace(".vtk", "").replace("fsi_", "") for f in Files }


  #-----------------------------------------------------------------------||---#
  Fout = [ open("drag.dat","w"), open("separation.dat","w") ]

  ## 1.0 Cd,Cl 
  ClCd,IDx = GetLastCl(Files)
  if len(IDx)==0: continue

  ClCd    = ClCd[IDx,:] 
  PropDic = KerDat() 
  if PropDic :
    ClCd[:,0 ] = ClCd[:,0 ] *  PropDic["U"] / PropDic["L"]   
    ClCd[:,1:] = ClCd[:,1:] / (0.5 * PropDic["rho"] * PropDic["L"] * PropDic["U"]**2) * 2 
    St = GetStrouhal(ClCd[:,1], dt=PropDic["dt"], d0=PropDic["L"], v0=PropDic["U"])
    print>> Fout[0], "## ", St 

  print>> Fout[0], "## gnuplot -e \"plot 'drag.dat' u 1:2  t 'Cl' w lp\" "
  print>> Fout[0], "## gnuplot -e \"plot 'drag.dat' u 1:3  t 'Cd' w lp\" "
  np.savetxt(Fout[0], ClCd)

  ## 2.0 Roots 
  nRutPhi  = np.array([-1,-1])
  nSects   = 0  
  SeccDict = {}

  Roots    = [] 
  NSects   = []
  for idx in IDx:
    ## 2.1 
    fin    = Files[idx]
    Props  = np.loadtxt(fin)
    Phi    = np.arctan2(Props[:,4-1], Props[:,3-1])
    Sorted = np.argsort(Phi)
    Props  = Props[Sorted,:]
    Phi    = Phi[Sorted]

    ## 2.2 Time 
    T  = np.average( Props[:,2-1] )

    ## 2.3 Vorticity 
    Wz = Props[:,7-1] 

    ## 2.4 Roots. 
    RutPhi     = GetAllZeros(Phi,Wz)
    RutPhi     = np.sort(RutPhi)
    Roots.append( [T] + RutPhi[:].tolist() )

    nRutPhi[1] = nRutPhi[0]
    nRutPhi[0] = len(RutPhi)
    if (not nRutPhi[0] == nRutPhi[1]) and (np.all(nRutPhi>=0)): nSects += 1 
    NSects.append( (nSects,len(RutPhi)) )

#  print NSects 

  ## 2.5 T = (T - Tmin)/(Tmax-Tmin) 
  Time  = np.array([r[0] for r in Roots]) 
  Time -= Time.min()
  Time /= np.where( Time.max()==0.0, 1.0, Time.max() )   
  #print Time , Time.min(), Time.max() 

  for i,t in enumerate(Time): Roots[i][0] = t 
  ClCd[:,0] = Time[:]

  ## 2.6 Sort roots's curves 
  print>> Fout[1], "## gnuplot -e \" plot 'separation.dat' u 1:(-\$2) w lp, '' u 1:3 w lp, '' u 1:4 w l  \" " 

  RootCurves = [] 
  NSects     = np.array(NSects)
  NSectsDic  = getDicFromArray(NSects[:,0])
  for k,v1 in NSectsDic.iteritems():
    v2 = np.array([ [NSects[i,0]] for i in v1 if NSects[i,1]<=3 and NSects[i,1]>0]) 
    v3 = np.array([ Roots[i]      for i in v1 if NSects[i,1]<=3 and NSects[i,1]>0])

    if v3.shape[0] > 0:
      v3 = np.concatenate( (v3,v2), axis=1)
      np.savetxt(Fout[1], v3); #print>> Fout[1], "\n"  
      RootCurves.append(v3)

    v4 = np.array([ [-NSects[i,0]] for i in v1 if not NSects[i,1]<=3 ])
    v5 = np.array([ Roots[i] for i in v1 if not NSects[i,1]<=3 ])
    if v5.shape[0] > 0: 
      T      = v5[:, 0]
      First  = v5[:, 1]
      Last   = v5[:,-1]
      Middle = v5[:,2:-1]
      for j,M in enumerate(Middle[:,:].T):
        C   = np.array([ [T[i],First[i],m,Last[i],v4[i]] for i,m in enumerate(M) ])
        ind = np.argsort( C[:,2] )
        C   = C[ind,:]
        np.savetxt(Fout[1], C); #print>> Fout[1], "\n"
        RootCurves.append(C)

    print>> Fout[1], "\n"   

  if not all( [ Curves.shape[1]>4 for Curves in RootCurves] ): 
    print  "## ERROR: at least 3 'curves' are necessaries "  
    print  "##\t gnuplot -e \" plot 'separation.dat' u 1:(-\$2) w lp, '' u 1:3 w lp, '' u 1:4 w l  \"  \n"   
    continue 

  ## 3.2.1 External roots (Time=0, Theta1=1, Theta3=3) 
  Aux1 = [ curve for Curves in RootCurves for curve in Curves[:,(0,1,3)]]
  Aux1 = np.array(Aux1)

  ## 3.2.2 Internal roots (Time=0, Theta2=2, Type=4)  
  Aux2 = [ curve for Curves in RootCurves for curve in Curves[:,(0,2,4)] ]
  Aux2 = np.array(Aux2)

  ## 3.3.1 Internal roots (Theta1=1, Theta2=2, Type=4)  
  Aux4 = [ curve for Curves in RootCurves for curve in Curves[:,(1,2,4)] ]
  Aux4 = np.array(Aux4)

  ## 3.3.1 Internal roots (Theta3=3, Theta2=2, Type=4)  
  Aux5 = [ curve for Curves in RootCurves for curve in Curves[:,(3,2,4)] ]
  Aux5 = np.array(Aux5)

  #-----------------------------------------------------------------------||---#
  if PropDic :
    Re28, Tw0, Reff, Tf0, nu0f, kwf = REeff(PropDic["Re"], PropDic["L"], PropDic["U"], PropDic["Ti"], PropDic["Tw"], C=0.28)
    Re50, Tw0, Reff, Tf0, nu0f, kwf = REeff(PropDic["Re"], PropDic["L"], PropDic["U"], PropDic["Ti"], PropDic["Tw"], C=0.50)

    St = GetStrouhal(ClCd[:,1], dt=PropDic["dt"], d0=PropDic["L"], v0=PropDic["U"])
    #Cd = np.percentile(a, 50)  

    _avers = [] 
    _avers.append( 1   )
    _avers.append( ifin )
    _avers.append( PropDic["Re"] )
    _avers.append( St  ) 
    for _x in np.average( Aux1[:,1:], axis=0): _avers.append( _x ) 
    _avers.append( np.average( Aux2[:,1], axis=0) )
    _avers.append( np.average( ClCd[:,1] ) )
    _avers.append( np.average( ClCd[:,2] ) )
    STATIs[0].append( _avers )
    for _a in  _avers: print _a,
    print "\n#", 

    _per25 = []
    _per25.append( 2 )
    _per25.append( ifin )
    _per25.append( PropDic["Re"] )
    _per25.append( St )
    for _x in np.percentile( Aux1[:,1:], 25, axis=0): _per25.append( _x )
    _per25.append( np.average( Aux2[:,1], axis=0) )
    _per25.append( np.percentile(ClCd[:,1],25) )
    _per25.append( np.percentile(ClCd[:,2],25) ) 
    STATIs[1].append( _per25 )

    _per50 = []
    _per50.append( 3 )
    _per50.append( ifin )
    _per50.append( PropDic["Re"] )
    _per50.append( St )
    for _x in np.percentile( Aux1[:,1:], 50, axis=0): _per50.append( _x )
    _per50.append( np.average( Aux2[:,1], axis=0) )
    _per50.append( np.percentile(ClCd[:,1],50) )
    _per50.append( np.percentile(ClCd[:,2],50) )
    STATIs[2].append( _per50 )

    _per75 = []
    _per75.append( 4 )
    _per75.append( ifin )
    _per75.append( PropDic["Re"] )
    _per75.append( St )
    for _x in np.percentile( Aux1[:,1:], 75, axis=0): _per75.append( _x )
    _per75.append( np.average( Aux2[:,1], axis=0) )
    _per75.append( np.percentile(ClCd[:,1],75) )
    _per75.append( np.percentile(ClCd[:,2],75) )
    STATIs[3].append( _per75 )


  #-----------------------------------------------------------------------||---#
  #-----------------------------------------------------------------------||---#
  #-----------------------------------------------------------------------||---#
  ## 3.0 Plots 
  import fractions
  _X = np.linspace(0.0, 1.0, 13)
  _L = [ round(x,2) for x in np.linspace(0.0,1.0, _X.shape[0]) ]
  _L = [ fractions.Fraction(x).limit_denominator(12) for x in _L ]

  if 0 :
    ## Plot1 
    P1,A1,F1 = PlotSimpleXY( [ClCd[:,0],[ClCd[:,1], ClCd[:,2]]] ) #, [ClCd[:,0],[ClCd[:,0]]] )
    [ P1.setp(f1,marker='None') for f1 in F1 ]
    [ a1.legend().remove()      for a1 in A1 ]
    F1[1].remove(); #F1[2].remove()

    ## 3.1 Cd Vs Time  
    idx  = 0
    P1.setp( A1[idx], xticks=_X, xticklabels=_L)
    A1[idx].set_ylabel("$Cl$", fontsize=14) 
    P1.setp(F1[0], linewidth=2.0, color="darkgreen")

    MaxMinZeros  = [ ClCd[_i,0] for _i in GetExtremes(ClCd[:,1]) ]
    MaxMinZeros += [ ex for ex in GetAllZeros(ClCd[:,0],ClCd[:,1]) ]
    MaxMinZeros  = sorted(MaxMinZeros)

    Z  = lambda _x:  np.interp(_x,ClCd[:,0],ClCd[:,1])
    [ A1[idx].plot(_m, Z(_m), "o", color="darkgreen", ms=10) for _m in MaxMinZeros ]

    ## 3.2 Angle Vs Time  
    idx  = 1 
    #A1[idx].cla()  
    P1.setp( A1[idx], xticks=_X, xticklabels=_L)
    P1.setp( A1[idx].get_xticklabels(),fontsize=12)
    A1[idx].set_ylabel("$\\theta_s$", fontsize=18)
    A1[idx].set_ylim(-1.5,1.5)
    A1[idx].set_xlabel("$\\tau$", fontsize=20)

    ## 3.2.1 External roots (Time=0, Theta1=1, Theta3=3) 
    X   = Aux1[:,0]
    for Y,_c in zip(Aux1[:,1:].T,["r","b"]) :
      A1[idx].plot( X,Y, "-", color=_c, lw=2.0)
      #A1[idx].axhline( np.percentile(Y,50),ls="--",lw=1.0,color='k') 

      #from scipy.misc   import derivative
      #Z  = lambda _x:  np.interp(_x,X,Y)
      #dZ = np.array([ derivative(Z,x,dx=1e-6) for x in X])
      #print [ np.interp(ex,X,dZ) for ex in GetAllZeros(X,dZ)] 
      #[ A1[idx].axvline(ex, ls="dotted",lw=2.0,color='k') for ex in GetAllZeros(X,dZ)]  
      #
      #from scipy.signal import argrelextrema
      #IDsmin = argrelextrema(Y, np.less)
      #IDsmax = argrelextrema(Y, np.greater)
      #IDs    = np.sort( np.append(IDsmin,IDsmax) )
      #
      #[ A1[idx].axvline(ex, ls="--",lw=1.0,color='k', zorder=0) for ex in X[IDs] if abs(np.interp(ex,X,dZ))<=1e-1 ]  
      #[ A1[  0].axvline(ex, ls="--",lw=1.0,color='k', zorder=0) for ex in X[IDs] if abs(np.interp(ex,X,dZ))<=1e-1 ]

      Z  = lambda _x:  np.interp(_x,X,Y)
      [ A1[idx].plot(_m, Z(_m), "o", color="darkgreen", ms=10) for _m in MaxMinZeros ]
      [ A1[idx].axvline(_m, ls="--",lw=1.0,color='darkgreen', zorder=0) for _m in MaxMinZeros ]
      [ A1[  0].axvline(_m, ls="--",lw=1.0,color='darkgreen', zorder=0) for _m in MaxMinZeros ]

    ## 3.2.2 Internal roots (Time=0, Theta2=2, Type=4)  
    for k,v in getDicFromArray(Aux2[:,-1]).iteritems():
      X = Aux2[v,0] 
      if k>=0: A1[idx].plot( X, Aux2[v,1], "k-",  lw=2.0)
      if k< 0: A1[idx].plot( X, Aux2[v,1], "k--", lw=2.0)

    Z  = lambda _x:  np.interp(_x,Aux2[:,0],Aux2[:,1])
    [ A1[idx].plot(_m, Z(_m), "o", color="darkgreen", ms=10) for _m in MaxMinZeros ]

    P1.savefig('%s_1.pdf' % (mySelf) )
    #P1.show() 

  #-----------------------------------------------------------------------||---#
  #-----------------------------------------------------------------------||---#
  ## Plot2 
  ## 3.3 Angle Vs Angle  
  if 0 :
    P2,A2,F2 = PlotSimpleXY( [ClCd[:,0],[ClCd[:,0]]] )
    [ P2.setp(f2,marker='None') for f2 in F2 ]
    [ a2.legend().remove()      for a2 in A2 ]
    F2[0].remove()

    idx  = 0  

    A3 = A2[idx].twiny() 
    A2 = A2[idx] #.twiny()

    A2.cla()
    P1.setp( A2.get_xticklabels(),fontsize=16)
    P1.setp( A2.get_yticklabels(),fontsize=16)
    A2.grid()
    A2.set_xlabel("$-\\theta_{s_1}$", fontsize=20)
    A2.spines['top'].set_color('r')
    A2.xaxis.label.set_color('r')
    A2.tick_params(axis='x', colors='r')
    A2.set_ylabel("$\\theta_{s_2}$", fontsize=20)

    #A3.cla()
    A3.grid()
    P1.setp( A3.get_xticklabels(),fontsize=16)
    P1.setp( A3.get_yticklabels(),fontsize=16)
    A3.set_xlabel("$\\theta_{s_3}$", fontsize=20)
    #A3.spines['bottom'].set_color('b')
    A2.spines['bottom'].set_color('b')
    A3.xaxis.label.set_color('b')
    A3.tick_params(axis='x', colors='b')

    A2.spines['bottom'].set_color('b')

    Yr  = np.amax([np.amin(np.abs(Aux1[:,(1,2)]),axis=0), np.amax( np.abs(Aux1[:,(1,2)]), axis=0)], axis=1)
    dYr = 0.025 * ( Yr.max() - Yr.min() )
    A2.set_xlim(Yr.min()-dYr, Yr.max()+dYr)
    A3.set_xlim(Yr.min()-dYr, Yr.max()+dYr)

    #for Curve in RootCurves:
    #  A3.plot( np.abs(Curve[:,1]),  Curve[:,2], "b-")
    #  A3.plot( np.abs(Curve[:,3]),  Curve[:,2], "r-")

    ## 3.3.1 Internal roots (Theta1=1, Theta2=2, Type=4)  
    for k,v in getDicFromArray(Aux4[:,-1]).iteritems():
      if k>=0: A2.plot(-Aux4[v,0], Aux4[v,1], "r-",  lw=2.0)
      if k< 0: A2.plot(-Aux4[v,0], Aux4[v,1], "r--", lw=2.0)

    #A2.axvline( np.percentile(-Aux4[:,0],50), lw=1.0, ls=':', color='r') 
    #A2.axhline( np.percentile( Aux4[:,1],50), lw=1.0, ls=':', color='r') 
    #[ A2.plot(-Aux4[_i,0],Aux4[_i,1], "ro") for _i in GetExtremes( Aux4[:,1]) ] 
    #[ A2.plot(-Aux4[_i,0],Aux4[_i,1], "rs") for _i in GetExtremes(-Aux4[:,0]) ]

    ## 3.3.1 Internal roots (Theta3=3, Theta2=2, Type=4)  
    for k,v in getDicFromArray(Aux5[:,-1]).iteritems():
      if k>=0: A3.plot( Aux5[v,0], Aux5[v,1], "b-",  lw=2.0)
      if k< 0: A3.plot( Aux5[v,0], Aux5[v,1], "b--", lw=2.0)

    #A3.axvline( np.percentile( Aux5[:,0],50), lw=1.0, ls=':', color='b') 
    #A3.axhline( np.percentile( Aux5[:,1],50), lw=1.0, ls=':', color='b')
    #[ A3.plot( Aux5[_i,0],Aux5[_i,1], "bo") for _i in GetExtremes(Aux5[:,1]) ]
    #[ A3.plot( Aux5[_i,0],Aux5[_i,1], "bs") for _i in GetExtremes(Aux5[:,0]) ]

    for Curve in RootCurves:
      Z1  = lambda _x:  np.interp(_x, Curve[:,0],-Curve[:,1] )
      Z2  = lambda _x:  np.interp(_x, Curve[:,0], Curve[:,2] )
      Z3  = lambda _x:  np.interp(_x, Curve[:,0], Curve[:,3] )

      ## (Theta1=1, Theta2=2)  
      [      A2.plot( Z1(_m), Z2(_m), "o", color="darkgreen", ms=22) for _m in MaxMinZeros if (Curve[:,0].min()<=_m and _m<=Curve[:,0].max())]
      [      A2.annotate(str(_i), xy=(Z1(_m),Z2(_m)),size=12,color="w",ha="center", va="center")
        for _i,_m in zip(["1/4", "1/2", "3/4"], MaxMinZeros) if (Curve[:,0].min()<=_m and _m<=Curve[:,0].max())]

      ## (Theta3=3, Theta2=2)  
      [ A3.plot( Z3(_m), Z2(_m), "o", color="darkgreen", ms=22 ) for _m in MaxMinZeros if (Curve[:,0].min()<=_m and _m<=Curve[:,0].max())]
      [ A3.annotate(str(_i), xy=(Z3(_m),Z2(_m)),size=12,color="w",ha="center", va="center") 
        for _i,_m in zip(["1/4", "1/2", "3/4"],MaxMinZeros) if (Curve[:,0].min()<=_m and _m<=Curve[:,0].max())]

    P2.savefig('%s_2.pdf' % (mySelf) )
    #P2.show()

  #-----------------------------------------------------------------------||---#
  ## Plot3 
  ## 3.2 Angle Vs Time  
  if 1 :
    _Ly = np.linspace(-np.pi,np.pi,19) * 180 / np.pi 
    _Ly = [ str( round(l,0) ) for l in _Ly ] 

    P1,A1,F1 = PlotSimpleXY( [np.random.uniform(0,1,10), [np.random.uniform(0,1,10)]] )  
    [ P1.setp(f1,marker='None') for f1 in F1 ]
    [ a1.legend().remove()      for a1 in A1 ]
    F1[0].remove(); #F1[2].remove()

    idx = 0  
    A1[idx].cla()  
    A1[idx].grid() 
    P1.setp( A1[idx], yticks=np.linspace(-np.pi,np.pi,len(_Ly)), yticklabels=_Ly )
    P1.setp( A1[idx], xticks=_X, xticklabels=_L)
    P1.setp( A1[idx].get_xticklabels(),fontsize=12)
    A1[idx].set_ylabel("$\\theta_s$", fontsize=18)
    A1[idx].set_xlabel("$\\tau$", fontsize=20)
    A1[idx].set_ylim(-1.5,1.5)


    ## 3.2.1 External roots (Time=0, Theta1=1, Theta3=3) 
    X   = Aux1[:,0]
    for Y,_c in zip(Aux1[:,1:].T,["r","b"]) :
      A1[idx].plot( X,Y, "-", color=_c, lw=2.0)
      median = np.percentile(Y,50)
      A1[idx].axhline( median,ls="--",lw=1.0,color='k') 
      A1[idx].annotate( "% 6.3f" % (median*180/np.pi), xy=(1.0,median),size=13,color="k", va="center") #, va="center")

      from scipy.misc   import derivative
      Z  = lambda _x:  np.interp(_x,X,Y)
      dZ = np.array([ derivative(Z,x,dx=1e-6) for x in X])
      #print [ np.interp(ex,X,dZ) for ex in GetAllZeros(X,dZ)] 
      #[ A1[idx].axvline(ex, ls="dotted",lw=2.0,color='k') for ex in GetAllZeros(X,dZ)]  
      #
      from scipy.signal import argrelextrema
      IDsmin = argrelextrema(Y, np.less)
      IDsmax = argrelextrema(Y, np.greater)
      IDs    = np.sort( np.append(IDsmin,IDsmax) )

      if not IDs.shape[0]>100 : 
        [ A1[idx].axvline(ex, ls="--",lw=1.0,color=_c, zorder=0) for ex in X[IDs] if abs(np.interp(ex,X,dZ))<=1e-1 ]  
        Z1  = lambda _x:  np.interp(_x, X,Y )
        [ A1[idx].plot( _m, Z1(_m), "o", color=_c, ms=10) for _m in X[IDs] if abs(np.interp(_m,X,dZ))<=1e-1 ]
      else:
        print "## WARNING: len(IDsmin+IDsmax)>100 no plotted: %d !!\n" % IDs.shape[0] 


    ## 3.2.2 Internal roots (Time=0, Theta2=2, Type=4)  
    for k,v in getDicFromArray(Aux2[:,-1]).iteritems():
      X = Aux2[v,0]
      if k>=0: A1[idx].plot( X, Aux2[v,1], "k-",  lw=2.0)
      if k< 0: A1[idx].plot( X, Aux2[v,1], "k--", lw=2.0)

    P1.savefig('%s_3.pdf' % (mySelf) )
    #P1.show() 

  #-----------------------------------------------------------------------||---#

#-------------------------------------------------------------------------||---#
print>> Fout00,"""
## 1   2    3   4    5    6     7     8   9 
## Idx ifin Re  St  Ang1  Ang3  Ang2  Cl  Cd 
## -------------------------- 
## Idx=1 -> _avers 
## Idx=2 -> _per25  
## Idx=3 -> _per50   
## Idx=4 -> _per75 """
for Stati in STATIs:  
  np.savetxt(Fout00, Stati)
  print>> Fout00, "\n"  

print "#|_\t-->'python plotFrmCplngCdVsTime01_AllVsRe01.py '<--"
print "#|_OK!"

#-------------------------------------------------------------------------||---#
#-------------------------------------------------------------------------||---#
#-------------------------------------------------------------------------||---#
"""
SftpMarenostrum.py  /home/bsc21/bsc21704/z2016/REPOSITORY/TOOLs
"""
