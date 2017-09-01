#!/usr/bin/env python
import sys
import os
import os.path
import numpy as np
import json
import time
import multiprocessing

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
## Read files 
## 
henderson   = np.loadtxt("/Users/poderozo/z2017_2/RUNNER/CYlINDER01/CERFACS05/UNCOUPLED01_03/henderson1995.dat")
mhWuSepVsRe = np.loadtxt("/Users/poderozo/z2017_2/RUNNER/CYlINDER01/CERFACS05/UNCOUPLED01_03/MHWuSepVsRe01.dat")

## 
## IMPORTANTE!! 
## From 'plotFrmCplngCdVsTime01 -F T000_R0*'  
## 
## NEW: 
## 1   2    3   4    5    6     7     8   9 
## Idx ifin Re  St  Ang1  Ang3  Ang2  Cl  Cd 
## -------------------------- 
## Idx=1 -> _avers 
## Idx=2 -> _per25  
## Idx=3 -> _per50   
## Idx=4 -> _per75 
## 
## OLD: 
## 1  2   3   4    5    6     
## ID Re  St  Ang1 Ang3 Ang2   
## -------------------------- 
## Idx=1 -> _avers
## Idx=2 -> _per25  
## Idx=3 -> _per50   
## Idx=4 -> _per75   
##
Files = [ "TEST04/statistics_AllVsTime.dat", "TEST05/statistics_AllVsTime.dat", "TEST06/statistics_AllVsTime.dat"] 
#Files = [ "TEST04_03/statistics_AllVsTime.dat", "TEST05_03/statistics_AllVsTime.dat", "TEST06_03/statistics_AllVsTime.dat"]
Files = [ "statistics_AllVsTime.dat"] 

Res1 = [] 
Sts1 = [] 
Sps1 = [] 
Sps2 = []
Sps3 = []
Cds1 = []

for f in Files:
  Data = np.loadtxt(f) 
  IDx  = getDicFromArray(Data[:,0])

  Idx  = IDx.get(3,None) 
  Res1.append( Data[Idx,3-1]  ) ## Re   2 -> 3 
  Sts1.append( Data[Idx,4-1]  ) ## St   3 -> 4
  Sps1.append( Data[Idx,6-1]  ) ## Ang3 5 -> 6
  Cds1.append( Data[Idx,9-1]  ) ## Cd   9  

  Idx  = IDx.get(2,None)
  Sps2.append( Data[Idx,6-1]  ) ## Ang3 5 -> 6 

  Idx  = IDx.get(4,None)
  Sps3.append( Data[Idx,6-1]  ) ## Ang3 5 -> 6  

## Functions 
St   = lambda _Re, _T=1.0 : (0.2660 - 1.0160 * (0.72 + 0.28 * _T)**0.8887 / _Re**0.5 ) #* np.where(_Re>=40.0,1.0, 0.0) 
Cd   = lambda _Re, _a, _b, _c : _a + _b * _Re**_c
Nu   = lambda _Re, _a=0.565, _b=0.505 : _a * 0.71**(0.333) * _Re**_b # Hilpert  
Sep  = lambda _Re : 180.0 - ( 78.5 - 155.2 / _Re**0.5 )  
Sep  = lambda _Re, _a=95.7, _b=267.1, _c=-625.9, _d=1046.6 : _a + _b * _Re**-0.5 + _c * _Re**-1.0 + _d * _Re**-1.5

## 
DUMMY = np.random.uniform(0,1,10)
Re = np.linspace(5.0,500.0,100)

## Plots   
if 1 :  
  P1,A1,F1 = PlotSimpleXY( [DUMMY,[DUMMY,DUMMY,DUMMY]] )
  [ a.legend().remove() for a in A1 ]
  [ f.remove() for f in F1 ]
  [ a.xaxis.set_ticks( np.linspace(0.0,200.0,11) ) for a in A1 ]
  [ a.set_xlim(0.0,210.0) for a in A1 ]
  A1[-1].set_xlabel("$Re$",fontsize=16)

  if 1 : 
    idx = 0 
    #A1[idx].cla() 
    #A1[idx].grid() 
    A1[idx].set_ylabel("$St$",fontsize=16)
    A1[idx].set_ylim( 0.11, 0.21)

    ok = Re>=40 
    A1[idx].plot(Re[ok], St(Re,0.62)[ok], "k--", lw=2.0)
    A1[idx].plot(Re[ok], St(Re,1.01)[ok], "k--", lw=2.0)
    A1[idx].plot(Re[ok], St(Re,1.50)[ok], "k--", lw=2.0)

    for X,Y,m in zip(Res1,Sts1,["v","o","s"]):  
      ok = Y>0.0 
      A1[idx].plot(X[ok],Y[ok], "-r%s"%m , ms=8)    

  if 1 :
    idx = 1  
    #A1[idx].cla() 
    #A1[idx].grid() 
    A1[idx].set_ylabel("$\\theta_s$",fontsize=16)
    A1[idx].set_ylim(106.0,130.0) 
    A1[idx].set_ylim(100.0,181.0)

    A1[idx].plot(Re, Sep(Re), "k--", lw=2.0)

    if len(Res1)>1: 
      A1[idx].plot( Res1[1], 180-Sps1[1]*180/np.pi ) 
    else:
      A1[idx].plot( Res1[0], 180-Sps1[0]*180/np.pi )

    for X,Y,m in zip(Res1,Sps1,["v","o","s"]):  A1[idx].plot(X,180-Y*180/np.pi, "-r%s"%m , ms=8)

    #for i,m in enumerate(["v","o","s"]):  
    #  per25 = 180-Sps2[i]*180/np.pi
    #  per50 = 180-Sps1[i]*180/np.pi 
    #  per75 = 180-Sps3[i]*180/np.pi
    #  e1    = np.abs(per50-per25)
    #  e2    = np.abs(per50-per25)
    #  A1[idx].errorbar( Res1[i], per50, yerr=[e1,e2], capthick=2) 

  if 1 :
    idx = 2
    #A1[idx].cla() 
    #A1[idx].grid() 
    A1[idx].set_ylabel("$Cd$",fontsize=16)
    A1[idx].set_ylim(1.0,4.1)
    A1[idx].plot(henderson[:,0], henderson[:,1], "k--", lw=2.0)

    for X,Y,m in zip(Res1,Cds1,["v","o","s"]):
      A1[idx].plot(X,Y, "-r%s"%m , ms=8)

  P1.savefig('%s_1.pdf' % (mySelf) )
  #P1.show()

if 1 :
  P2,A2,F2 = PlotSimpleXY( [DUMMY,[DUMMY]] )
  [ a.legend().remove() for a in A2 ]
  [ f.remove() for f in F2 ]
  [ a.xaxis.set_ticks( np.linspace(0.0,200.0,11) ) for a in A2 ]
  [ a.set_xlim(30.0,210.0) for a in A2 ]
  A2[-1].set_xlabel("$Re$",fontsize=16)

  if 1 :
    idx = 0
    A2[idx].cla() 
    A2[idx].grid() 
    A2[idx].set_xlabel("$\\theta_s$",fontsize=16)
    A2[idx].set_ylabel("$St$",fontsize=16)
    A2[idx].set_ylim( 0.12, 0.24)
    A2[idx].set_xlim(106.0,125.0) #(100.0, 181.0)

    func = lambda x, a, b, c :  a + b * x + c * x * x 
    func = lambda x, a, b :  a + b * x 

    for X,Y1,m in zip(Sps1,Sts1,["v","o","s"]):  
      X = 180-X*180/np.pi
      A2[idx].plot(X, Y1, "r%s"%m , ms=8)

      ok = Y1>0 
      X  = X[ok] 
      Y1 = Y1[ok]
      if np.any(ok) : 
        popt, pcov = curve_fit(func, X, Y1)

        Y2 = func(X,*popt)  
        A2[idx].plot(X, Y2, "-g")


    ok = Re>=40
    X  = Sep(Re)[ok] 
    Y  = St(Re,1.00)[ok]
    A2[idx].plot(X,Y) 

  P2.savefig('%s_2.pdf' % (mySelf) )
  #P2.show()


#-------------------------------------------------------------------------||---#
#-------------------------------------------------------------------------||---#
#-------------------------------------------------------------------------||---#
"""
SftpMarenostrum.py  /home/bsc21/bsc21704/z2016/REPOSITORY/TOOLs
"""
