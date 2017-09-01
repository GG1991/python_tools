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
getDicFromArray = lambda _A:{_u:np.nonzero(_A==_u)[0] for _u in np.unique(_A) }


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
    [ plt.setp(a.get_xticklabels(), fontsize=10  ) for a in ax[:] ]
    [ plt.setp(a.get_yticklabels(), fontsize=10  ) for a in ax[:] ]

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
  if _nClr<3:
    print "# \t ERROR: nRoots not enoghf", _nClr
    return np.zeros(0), np.zeros(0)

  _Clr  = _Clr[-3:]
  print "# |_Cl roots:", _Clr

  _OK = np.logical_and( _Clr[0] <= _Cl[:,0], _Cl[:,0] <= _Clr[-1] )
  _OK = np.nonzero(_OK)[0]

  return _Cl, _OK,


#-------------------------------------------------------------------------||---#
def GetStrouhal( _signal, dt=1.0, d0=None, v0=None):
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


#-------------------------------------------------------------------------||---#

def GetRootSections( Roots ):
  nRutPhi  = np.array([-1,-1])
  nSects   = 0
  NSects   = []
  for RutPhi in Roots:
    nRutPhi[1] = nRutPhi[0]
    nRutPhi[0] = len(RutPhi)
    if (not nRutPhi[0] == nRutPhi[1]) and (np.all(nRutPhi>=0)): nSects += 1
    NSects.append( (nSects,len(RutPhi)) )

  NSects     = np.array(NSects)
  NSectsDic  = getDicFromArray(NSects[:,0])
  return  NSectsDic


#==============================================================================#

#-------------------------------------------------------------------------||---#
## 1:T, 2:T*U/L, 3:Cl, 4:Cd, 5:Nu, 6:Nu, 7:Theta1, 8:Theta2, 9:Theta3, [10:Theta4, 11:Theta5], 12:T/T0  
Data02 = [ np.array([ float(c) for c in line.split()]) for line in open("pv_props02.dat")  ]

## Streams 
if 1 :
  Files    = LookinForFile("pv_streams*.dat")
  FilesDic = { f:f.replace(".dat", "").replace("pv_streams", "") for f in Files }
  if not len(Files)==len(Data02): exit("\tERROR: 'len(Files)==Fin02.shape[0]' !!\n") 

  for i in range( len(Files) ):
    #i=0 
    P2,A2,F2 = PlotSimpleXY( [np.random.uniform(0,1,10), [np.random.uniform(0,1,10)]] )
    [ a2.legend().remove()      for a2 in A2 ]

    F2[0].remove()
    P2.axis('off')
    P2.setp( [A2[0].set_xlim(-2.0,2.0)] )
    P2.setp( [A2[0].set_ylim(-1.5,3.5)] )
    P2.setp( [A2[0].get_xticklabels()], visible=False )
    P2.setp( [A2[0].get_yticklabels()], visible=False )
    A2[0].set_xlabel("")
    A2[0].set_ylabel("")

    import matplotlib.patches as patches
    circ = patches.Circle((0.0,0.0),1.0, lw=2.0, fc="gray", ec="k")
    A2[0].add_patch(circ)

    import warnings 
    warnings.filterwarnings("ignore") 
    fin   = Files[i]; #print "\t '%s' " % fin 
    S     =  np.loadtxt(fin)

    if not len(S)==0: 
      for k1,v1 in getDicFromArray(S[:,2]).iteritems():
        s   = S[v1,:-1] * 2
        lines, = A2[0].plot( -s[:,1], s[:,0], linewidth=0.5, color='g' )

      #Roots02 = np.array([ data[6:-1] for data in Data02 ])
      Roots02 = [ data[6:-1] if len(data)>7 else [np.nan] for data in Data02 ]

      Ruts  = np.array([ [np.cos(r), -np.sin(r)] for r in Roots02[i]])
      rut1  = A2[0].plot(Ruts[0   ,1],Ruts[   0,0],'ob', ms=15.0)
      rut2  = A2[0].plot(Ruts[1:-1,1],Ruts[1:-1,0],'ok', ms=15.0)
      rut3  = A2[0].plot(Ruts[-1  ,1],Ruts[  -1,0],'or', ms=15.0)

      P2.savefig('%s_Streams%03d.pdf' % (mySelf, i) )
      #P2.show()  


if 0 : 
  Data01 = [ np.array([ float(c) for c in line.split()]) for line in open("pv_props.dat")  ]

## Theta Vs Time 
if 0 : 
  Roots01 = [ np.append(data[0],data[6:]  ) for data in Data01 ]
  Roots02 = [ np.append(data[0],data[6:-1]) for data in Data02 ]

  P1,A1,F1 = PlotSimpleXY( [np.random.uniform(0,1,10), [np.random.uniform(0,1,10)]] )
  A1[0].cla() 

  Roots    = Roots01[:]
  Sections = GetRootSections( Roots )

  for k in Sections.keys():
    v    = np.sort(Sections[k])
    Ruts = [ Roots[i] for i in v ]
    Ruts = np.array(Ruts)
    X    = Ruts[:,0]

    for Y in Ruts[:,1:].T:
      root, = A1[0].plot(X,Y,'k-')
      if X.shape[0]<=2:
        root, = A1[0].plot(X,Y,'ko')

  P1.show() 


## Cl Vs T 
if 0 :
  D1 = np.array([ data[[0,2]] for data in Data01 ])
  D2 = np.array([ data[[0,2]] for data in Data02 ])

  P3,A3,F3 = PlotSimpleXY( [ D1[:,0],[D1[:,1]]  ] )
  [ P3.setp(f3,marker='None') for f3 in F3 ]
  [ a3.legend().remove()      for a3 in A3 ]
  A3[0].set_xlabel("$\\tau$", fontsize=16)
  A3[0].set_ylabel("$Cd$", fontsize=12)

  A3[0].plot( D2[:,0], D2[:,1], "ro", ms=15)
  [ A3[0].annotate(str(_i), xy=_r, size=10,color="w",ha="center", va="center")
    for _i,_r in enumerate(D2) ] 

  X = np.linspace( D2[:,0].min(), D2[:,0].max(), 5)
  L = [ round(x,2) for x in np.linspace(0.0,1.0, 5) ]  

  import fractions
  L = [ fractions.Fraction(x).limit_denominator(10) for x in L ]

  P3.xticks(X, L) 

  P3.savefig('%s_ClVsT.pdf' % (mySelf) )
  P3.show()


  #-----------------------------------------------------------------------||---#


#-------------------------------------------------------------------------||---#
#-------------------------------------------------------------------------||---#
#-------------------------------------------------------------------------||---#
"""
SftpMarenostrum.py  /home/bsc21/bsc21704/z2016/REPOSITORY/TOOLs
"""
