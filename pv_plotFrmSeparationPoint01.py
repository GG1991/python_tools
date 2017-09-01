
from paraview.simple import *
import numpy as np
import scipy as sc 
import scipy
import os
import vtk 
import sys 

from scipy.misc   import derivative
#import matplotlib.pyplot   as plt

 
PWD = os.getcwd()

print 
print "|_PWD: '%s'" % PWD
print "+Paraview Version>4.4", GetParaViewVersion(),  
print vtk.vtkVersion.GetVTKSourceVersion(),  
print scipy.__version__

#--------------------------------------------------------------------------||--#    
#--------------------------------------------------------------------------||--#    
getDicFromArray = lambda _A:{_u:np.nonzero(_A==_u)[0] for _u in np.unique(_A) }

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

#----------------------------------------------------------------# 
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
    IDs    = np.sort(IDs).astype(int) 
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
    if( len(Check) ): print "[GetAllZeros] Warning abs(Y(z))>1e-5 : ", Check  

    return np.sort(Zeros)  


#----------------------------------------------------------------# 
def extract_block( MultiBlock ):
  BLOCKs = None
  if( MultiBlock.GetClassName() == 'vtkMultiBlockDataSet'):
    n_blocks = MultiBlock.GetNumberOfBlocks()
    #print "  |_MultiBlock: n_blocks: %d" % n_blocks
    BLOCKs = [  MultiBlock.GetBlock(i) for i in range(n_blocks)]
    #for i, block in enumerate(BLOCKs): print "  |_MultiBlock: %i) '%s'" % ( i, block.GetClassName() )

  return BLOCKs


#----------------------------------------------------------------# 
def getCellType( _Obj, _T ):    
    from paraview.numpy_support import vtk_to_numpy

    _Obj.SMProxy.UpdatePipeline( _T )
    _Obj.UpdatePipelineInformation()

    GetOutput = servermanager.Fetch( _Obj )
    if( GetOutput.GetClassName()== 'vtkMultiBlockDataSet' ): GetOutput = extract_block(GetOutput)[0]
    if( GetOutput.GetClassName()== 'vtkMultiBlockDataSet' ): GetOutput = extract_block(GetOutput)[0]

    vtk_data    = GetOutput
    vtk_n_cells = vtk_data.GetNumberOfCells()
    vtk_n_pts   = vtk_data.GetNumberOfPoints()

    if(vtk_n_cells==0): 
      print "[save_data_cell] WARNING: n_cells==0"
      #sys.exit() 

    if(vtk_n_pts==0):
      print "[save_data_coords] WARNING: n_pts==0"
      #sys.exit()


    CELLIDS = [] 
    TYPES = {} 
    for idx in range(vtk_n_cells):
      cell_type = vtk_data.GetCellType(idx)
      cell_ids  = vtk.vtkIdList()
      vtk_data.GetCellPoints(idx, cell_ids) 
      n_cell_ids = cell_ids.GetNumberOfIds()  

      CellIDs = [] 
      for j in range(n_cell_ids):
        cell_id = cell_ids.GetId(j) 
        CellIDs.append( cell_id ) 
      CELLIDS.append(CellIDs) 

    #print "CELLIDS:", len(CELLIDS)

    Streams = [] 
    for i,CellIDs in enumerate(CELLIDS):
      stream = [ [ r for r in vtk_data.GetPoint(idx)] for idx in CellIDs ]
      stream = np.array(stream)
      stream[:,-1] = i+1  
      Streams.append( stream )

    return Streams  


#----------------------------------------------------------------# 
def getPtsCoord( _Obj, _T ):
    from paraview.numpy_support import vtk_to_numpy

    _Obj.SMProxy.UpdatePipeline( _T )
    _Obj.UpdatePipelineInformation()

    GetOutput = servermanager.Fetch( _Obj )
    if( GetOutput.GetClassName()== 'vtkMultiBlockDataSet' ): GetOutput = extract_block(GetOutput)[0]
    if( GetOutput.GetClassName()== 'vtkMultiBlockDataSet' ): GetOutput = extract_block(GetOutput)[0]

    vtk_n_pts = GetOutput.GetNumberOfPoints()
    Pts       = [] 
    for idx in range(vtk_n_pts):
      pt    = GetOutput.GetPoint(idx)
      n_pt  = len(pt)
      coord = []  
      for j in range(n_pt): coord.append( pt[j] ) 
      Pts.append( coord )

    return np.array(Pts) 


#----------------------------------------------------------------# 
def getPtsData( _Obj, _T=-1.0, _Prop=None, _fname=None ): 
    from paraview.numpy_support import vtk_to_numpy

    if(_Prop==None): 
      print _Obj.GetPointDataInformation().keys()
      sys.exit() 

    _Obj.SMProxy.UpdatePipeline( _T )
    _Obj.UpdatePipelineInformation()

    B = servermanager.Fetch( _Obj )
    if( B.GetClassName()== 'vtkMultiBlockDataSet' ): B = extract_block(B)[0]
    if( B.GetClassName()== 'vtkMultiBlockDataSet' ): B = extract_block(B)[0]

    Data = vtk_to_numpy( B.GetPointData().GetArray(_Prop) )

    if not _fname==None:
      time = B.GetInformation().Get(vtk.vtkDataObject.DATA_TIME_STEP())
      print "  |_Time: %f" % ( time )

      nout = "%s_%s" % (_fname, _Prop)  
      Fout = open(nout+".dat", "w")

      Fout.close() 

      print "  |_'%s' " % ( Fout.name )

    return Data 

#----------------------------------------------------------------# 
def GetLastCl(tref=False, Clref=False, Every=1):
  _Cl = [] 
  for _t1 in Times[::Every]: 
     r   = getPtsCoord( plotOnIntersectionCurves1, _t1 )
     phi = np.arctan2(r[:,1], -r[:,0]) # * 180 / np.pi
     idx = np.argsort(phi)
     phi = phi[idx]
     cl  = getPtsData( plotOnIntersectionCurves1, _t1, "TRACT" ) 
     cl  = cl[idx,:]
     aux = np.trapz(cl,phi,axis=0) / (2*np.pi) 

     if tref and Clref: 
       _Cl.append( [_t1*tref, -aux[1]*Clref, aux[0]*Clref ]  )
     else:
       _Cl.append( [_t1     , -aux[1]      , aux[0]       ]  )


  _Cl   = np.array(_Cl)
  _Clr  = GetAllZeros(_Cl[:,0], _Cl[:,1])
  _nClr = _Clr.shape[0]

  if _nClr>=3:
    _Clr  = _Clr[-3:]
    print "# |_Cl roots:", _Clr

    _OK1 = np.logical_and( _Clr[0] <= _Cl[:,0], _Cl[:,0] <= _Clr[-1] )
    _OK1 = np.nonzero(_OK1)[0]
    np.savetxt("getLastCl.dat", _Cl[_OK1,:] )

    dCr = 0.025*( _Clr[-1] - _Clr[0] )
    _OK2 = np.logical_and( _Clr[0]-dCr<= Times, Times <= _Clr[-1]+dCr )
    _OK2 = np.nonzero(_OK2)[0]

    return _Cl[_OK1,:], _OK2, _Clr 
  else: 
#    print "# \t [GetLastCl] ERROR: nRoots not enough", _nClr
#    exit()
    print "# \t [GetLastCl] WARNING: nRoots not enough: %d \n## \t gnuplot -e \"plot 'drag.dat' u 1:2  t 'Cl' w lp\"  \n" % _nClr

    _OK = [ _Cl.shape[0]-1 ]
    _OK = np.arange(_Cl.shape[0]).astype(int)

    return _Cl, _OK, _Clr 


#----------------------------------------------------------------# 
def SortRootsCurves():  
  RootCurves = []
  NSects     = np.array(NSects)
  NSectsDic  = getDicFromArray(NSects[:,0])
  for k,v1 in NSectsDic.iteritems():
    v2 = np.array([ [NSects[i,0]] for i in v1 if NSects[i,1]<=3 ])
    v3 = np.array([ Roots[i] for i in v1 if NSects[i,1]<=3 ])
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

#----------------------------------------------------------------# 
Ensi   = EnSightReader(CaseFileName='vortex2D.ensi.case')
KEYs   = Ensi.GetPointDataInformation().keys()
#print KEYs 

Times  = np.array(Ensi.TimestepValues)
nTimes = len(Times)
print "|_nTimes: %d" % nTimes

TimesDic = { time:i for i,time in enumerate(Times)  }


#----------------------------------------------------------------# 
computeDerivatives1 = ComputeDerivatives(Input=Ensi)
computeDerivatives1.Scalars = ['POINTS', 'DENSI']
computeDerivatives1.Vectors = ['POINTS', 'VELOC']
computeDerivatives1.OutputVectorType = 'Vorticity'
computeDerivatives1.OutputTensorType = 'Strain'
cellDatatoPointData1 = CellDatatoPointData(Input=computeDerivatives1)
cellDatatoPointData1.UpdatePipeline()


#----------------------------------------------------------------# 
plotOnIntersectionCurves1 = PlotOnIntersectionCurves(Input=cellDatatoPointData1)
plotOnIntersectionCurves1.SliceType = 'Cylinder'
plotOnIntersectionCurves1.SliceType.Center = [0.0, 0.0, 0.0]
plotOnIntersectionCurves1.SliceType.Axis = [0.0, 0.0, 1.0]
plotOnIntersectionCurves1.SliceType.Radius = 0.0035001
plotOnIntersectionCurves1.UpdatePipeline()


#----------------------------------------------------------------# 
streamTracer1 = StreamTracer(Input=Ensi, SeedType='High Resolution Line Source')
streamTracer1.Vectors = ['POINTS', 'VELOC']
streamTracer1.MaximumStreamlineLength = 0.1
streamTracer1.SeedType.Point1 = [0.0036, -0.0025, 0.0]
streamTracer1.SeedType.Point2 = [0.0036,  0.0025, 0.0]
streamTracer1.SeedType.Resolution = 10
streamTracer1.UpdatePipeline( )
##for t in Times: PF02.UpdatePipeline(time=t)

KEYs   = plotOnIntersectionCurves1.GetPointDataInformation().keys()


#----------------------------------------------------------------# 
#----------------------------------------------------------------# 
## Push: 
## Read files 
## Cols 
## 1:Ids 2:time 3:X   4:Y  
## 5:Fx  6:Fy   7:Wz  8:dP 9:dT 

class SetFsiVtkFiles:
  def __init__(self, _time, idx=0):
    self.time  = _time
    self.IDs   = ["X", "Y", "Fx", "Fy", "Wz", "dP", "dT"] 
    self.Props = { _idx:None for _idx in self.IDs } 

    self.Fout  = open("pv_%06d.vtk" % idx ,"w")
    return 

  def Push(self, _array, _id): 
    self.Props[_id] = _array.copy()  
    return 

  def Save(self):
    nprops   = len(self.IDs) + 2 
    nprops   = len(self.Props) + 2
    naux     = self.Props["X"].shape[0]
    aux      = np.zeros( (naux,nprops)  )
    aux[:,0] = np.arange(naux) 
    aux[:,1] = self.time  

    jdx = 1     
    print>> self.Fout, "## %d %s" % (jdx,"IDx"); jdx+=1 
    print>> self.Fout, "## %d %s" % (jdx,"Time"); jdx+=1    
    for i,_idx in enumerate(self.IDs): 
      _array = self.Props.get(_idx, None)
      if _array==None: 
        print>> self.Fout, "## %d %s None" % (jdx, _idx)  
      else:
        print>> self.Fout, "## %d %s" % (jdx, _idx) 
        aux[:,i+2] = _array 
      self.Props.pop(_idx,None)
      jdx+=1   

    for _idx, _array in self.Props.iteritems():
      print>> self.Fout, "## %d %s" % (jdx, _idx)
      aux[:,jdx-1] = _array
      jdx+=1   

    np.savetxt(self.Fout, aux)
    print "|_'%s' saved \n" % self.Fout.name  

    return 


#----------------------------------------------------------------# 
def GetStreamsLines( _time, pvInput, f_handle, _D=1.0):  
  _Streams  = getCellType( pvInput, _time )
  #f_handle = file('streams%04d.dat' % Idt , 'w')
  for _s in _Streams:
    _s[:,[0,1]] /= _D 
    _aux         =  np.zeros( (_s.shape[0],_s.shape[1]+1) ) 
    _aux[:,:-1]  = _s 
    _aux[:, -1]  = _time  
    np.savetxt(f_handle, _aux) 
    print>> f_handle, '\n\n#' 
  #f_handle.close()  


#----------------------------------------------------------------# 
def GetAllProps(_time, pvInput ):
  #Idt    = TimesDic[_time] 
  _Props  = [] 

  ## (0) Rs, Phi  
  R   = getPtsCoord( pvInput, _time )
  Phi = np.arctan2(R[:,1],R[:,0]) # * 180 / np.pi
  Id  = np.argsort(Phi)
  Phi = Phi[Id]
  OK  = Phi >= 0.0
  _Props.append( [_time] )  

  vtks.Push( R[Id,0], "X" ) 
  vtks.Push( R[Id,1], "Y" )

  ## Cd, Cl [ /= (0.5 * rho * D * v**2) * ( pi * D )  ]
  _f = [ 0.0, 0.0, 0.0 ]
  if "TRACT" in KEYs :
    _dF = getPtsData( pvInput, _time, "TRACT" )
    _dF = _dF[Id,:]
    vtks.Push( _dF[:,0], "dFx")
    vtks.Push( _dF[:,1], "dFy")

    _F  = np.trapz(_dF,x=Phi,axis=0) / (2*np.pi)  # <-- TTRAC   
    _f  = [ _time, -_F[1], _F[0] ]
    if UNITS : _F /= ( 0.5 * rho * D * U**2 ) 
    if UNITS : _F *= (np.pi * D)
    if UNITS : _t = _time * U / D  
    if UNITS : _f = [ _t, -_F[1], _F[0] ]
  _Props.append( _f  )

  ## TTRAC = \int TRACT -> F = \int dF  <-----  
  if "TTRAC" in KEYs :
    _F = getPtsData( pvInput, _time, "TTRAC" )
    _F = _F[Id,:]
    vtks.Push( _F[:,0], "Fx")
    vtks.Push( _F[:,1], "Fy")
  elif "REACT" in KEYs :
    _F = getPtsData( pvInput, _time, "REACT" )
    _F = _F[Id,:]
    #vtks.Push( _F[:,0], "Fx")
    #vtks.Push( _F[:,1], "Fy")

  if "PRESS" in KEYs :
    _aux = getPtsData( pvInput, _time, "PRESS" )
    vtks.Push( _aux[Id], "P")

  if "TEMPE" in KEYs :
    _aux = getPtsData( pvInput, _time, "TEMPE" )
    vtks.Push( _aux[Id], "T")

  ##  Nu 
  if "GRATE" in KEYs :
    Nu  = getPtsData( pvInput, _time, "GRATE" )
    Nu  = Nu[Id,:]
    Nu  = np.linalg.norm(  Nu, axis=1 )
    vtks.Push(Nu, "dT")

    if UNITS : Nu *= D / (T2 - T1)

    aux2 = np.average(Nu)
    aux3 = np.trapz( Nu, Phi ) / (2*np.pi)  

    _Props.append( [aux2, aux3] )  

  ## (1) Vorticity/Finding separation points   
  ##     C_tau = mu_W ( dU/dy - dV/dx )_W / ( rho_0 U_0**2 / 2 )
  if "Vorticity" in KEYs :
    W   = getPtsData( pvInput, _time, "Vorticity" ) 
    W   = W[Id,:]
    Wz  = W[:,-1]
    vtks.Push(Wz, "Wz")

    RutPhi  = GetAllZeros( Phi, Wz ) 
    RutPhi  = np.sort(RutPhi) 
    _Props.append( RutPhi.tolist() ) 

  return _Props 

#----------------------------------------------------------------# 
#----------------------------------------------------------------# 
DIAMETER = 7e-3 

UNITS = False 
if UNITS:
  rho = 1.224477
  D   = 7e-3
  U   = 3.403667
  T1  = 288.150000
  T2  = 291.031500

#----------------------------------------------------------------# 
if "TRACT" in KEYs : 
  LD,IDx,Clr = GetLastCl(Every=1)

  if 1 :
    ## Save all properties ...  
    Fout   = [ open("pv_props01.dat","w"), open("pv_drag.dat","w") ]
    Time   = []
    ClCd   = []
    Nu     = []
    Roots  = []

    for i,idx in enumerate(IDx):
      time = Times[idx]
      vtks = SetFsiVtkFiles( time, i )  
      prop = GetAllProps(time, plotOnIntersectionCurves1)
      vtks.Save() 
      Time.append(  prop[0] ) 
      ClCd.append(  prop[1] )
      Nu.append(    prop[2] )
      Roots.append( prop[3] )

      for p in prop: 
        for x in p: print>> Fout[0], x,
      print>> Fout[0], round(1.0,3)

    Fout[0].close()

    ClCd = np.array(ClCd) 
    np.savetxt(Fout[1], ClCd )

  if 1 : 
    ## Check a whole cycle ... 
    X = LD[:,0]
    Y = LD[:,1]

    ## Max/Min 
    from scipy.misc   import derivative
    Z    = lambda _x:  np.interp(_x,X,Y)
    dZ   = np.array([ derivative(Z,x,dx=1e-6) for x in X])
    Tex  = GetAllZeros(X,dZ); #print [ np.interp(ex,X,dZ) for ex in Tex] 
    if( len(Tex)==0 ): print "#\tWARNING: 'len(Tex)==0' !!\n"

    ## Zeros+Max/Mins 
    Ts0   = np.append(Tex,Clr)
    if len(Ts0) > 0 : 
      Ts0   = np.sort(Ts0) 
      dTs0  = Ts0[1: ]-Ts0[ :-1]
      Ts1   = Ts0[:-1] + dTs0 * 1.0 / 3.0  
      Ts    = np.sort( np.append(Ts0,Ts1) )
      Ts2   = Ts0[:-1] + dTs0 * 2.0 / 3.0 
      Ts    = np.sort( np.append(Ts ,Ts2) )

      Fout1 = open("pv_props02.dat","w") 
      for te in Ts: 
        inter = TemporalInterpolator(Input=plotOnIntersectionCurves1)
        inter.DiscreteTimeStepInterval = te 
        props = GetAllProps( te, inter )

        _te = (te-Ts.min())/(Ts.max()-Ts.min()) 
        for p in props:
          for x in p: print>> Fout1, x,
        print>> Fout1, round(_te,3),  
        print>> Fout1

        Fout2 = open('pv_streams%04d.dat' % (round(_te,3)*1000), "w") 
        GetStreamsLines( te, streamTracer1, Fout2, D ) 
        Fout2.close() 
    else: 
        print "#\tWARNING: no cycle..."

        import shutil 
        shutil.copy2("pv_props01.dat", "pv_props02.dat")

        for ii, time in enumerate(Times): 
          Fout2 = open('pv_streams%04d.dat' % (ii), "w")
          GetStreamsLines( time, streamTracer1, Fout2, DIAMETER )
          Fout2.close()


else: ## if not "TRACT" in KEYs :

  if 1 :
    Fout1 = open("pv_props02.dat","w")

    for ii, time in enumerate(Times): 
      vtks  = SetFsiVtkFiles( time , ii )
      props = GetAllProps(time, plotOnIntersectionCurves1)
      vtks.Save()
      for p in props:
        for x in p: print>> Fout1, x,
      print>> Fout1, round(1.0,3),
      print>> Fout1

      Fout2 = open('pv_streams%04d.dat' % (ii), "w")
      GetStreamsLines( time, streamTracer1, Fout2, DIAMETER )
      Fout2.close()


#----------------------------------------------------------------# 


print "|_\t-->'python plotFrmPvSeparationPoint01.py'<--"
print "|_OK!"
#----------------------------------------------------------------# 
"""
SftpMarenostrum.py  /home/bsc21/bsc21704/z2016/REPOSITORY/TOOLs
"""
