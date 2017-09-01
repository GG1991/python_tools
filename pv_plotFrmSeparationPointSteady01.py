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

#=========================================================================||===#
FULL_PATH = os.path.realpath(__file__)
FULL_PATH = os.getcwd()
#print FULL_PATH 

path, filename = os.path.split(FULL_PATH)
#print path 

current_folder_path, current_folder_name = os.path.split(os.getcwd())
#print current_folder_path, current_folder_name

#date = time.strftime("%Y%b%d")
#date = date.upper()

#n_cpus = multiprocessing.cpu_count()

#mySelf = sys.argv[0].split(".")[0]
mySelf = sys.argv[0].split("/")[-1].split(".")[0]

#--------------------------------------------------------------------------||--#    
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-F', action='store', dest='Files',
                    default=[], type=str, nargs='+',
                    help='Files',
                    )
options = parser.parse_args()

#
if(
    options.Files == []
  ):
  parser.print_help()
  print
  sys.exit()
import glob
Aux  = [ os.path.split(fin) for fin in options.Files ]
Aux  = { a[1]:glob.glob(a[0]) for a in Aux }
Aux0 = [ k for k,v in Aux.iteritems() if len(v)==0 ]
Aux1 = [ ["%s/%s" %(f,k) for f in v] for k,v in Aux.iteritems() ]
Aux  = Aux0 + sorted( sum(Aux1,[]) )
Aux  = sorted(Aux)
print "# + ", Aux

#--------------------------------------------------------------------------||--#    
getDicFromArray = lambda _A:{_u:np.nonzero(_A==_u)[0] for _u in np.unique(_A) }


#-------------------------------------------------------------------------||---#
def KerDatTem( Keys=["CONDUCTIVITY", "SPECIFIC_HEAT", "DENSITY"] ): 
  Files = glob.glob("*.ker.dat")
  for K in Keys: 
    Files = [ f for f in Files if K in open(f).read() ]

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
        line = line.replace(":", "")
        line = line.replace(",", "")
        line = line.replace("=", "")
        if(line.find(key)>0):
          line = line[:-1]
          line = line.split(key)
          line = line[1]
          Dict[key] = line.split()

    return Dict  
 

#-------------------------------------------------------------------------||---#
def KerDatNsi( Keys=['Re =', 'Ma =', 'Pr =', 'Tr =']):
  ## Dt 
  Files = glob.glob("*.dat")
  Files = [ f for f in Files if "NASTIN_MODULE" in open(f).read() ]
  Dt    = False
  if len(Files)>0:
    Dt  = [ l.split()[-1] for l in open(Files[0]).readlines() if "TIME_STEP_SIZE" in l ]
    Dt  = float(Dt[0])

  ## Fluid props 
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
    #print IDsmin, IDsmax

    IDs    = np.append( IDsmin, IDsmax )
    if len(IDs)<=1: 
      IDs  = np.append( IDs, [0,_X.shape[0]-1] )
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
    print "# \t [GetLastCl] WARNING: nRoots not enough: %d \n## \t gnuplot -e \"plot 'drag.dat' u 1:2  t 'Cl' w lp\"  \n" % _nClr

    _OK = [ _Cl.shape[0]-1 ]
    _OK = np.arange(_Cl.shape[0]).astype(int)

    return _Cl, _OK, _Clr 


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

  def Average(self, _id):
    _array = self.Props.get(_id, None)
    if not _array==None:   
      return np.average(_array) 

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

    if NsiDic :
      print>> self.Fout, "## ",  
      print>> self.Fout, " ".join([ "%s:%f"%(k,v) for k,v in NsiDic.iteritems() ])


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
def GetPloOverLine( _time, pvInput, f_handle, _D=1.0 ):
  plotOverLine1 = PlotOverLine(Input=pvInput, Source='High Resolution Line Source')
  plotOverLine1.Tolerance         = 1e-16
  plotOverLine1.Source.Point1     = [-0.08749999850988388, 0.0, 0.0]
  plotOverLine1.Source.Point2     = [ 0.14000000059604645, 0.0, 0.0]
  plotOverLine1.Source.Resolution = 999  

  if "VELOC" in KEYs :
    _R = getPtsCoord( plotOverLine1, _time )
    _V = getPtsData(  plotOverLine1, _time, "VELOC" )

    _aux        =  np.zeros( (_V.shape[0],2) )
    _aux[:,0 ]  = _R[:,0] / _D   #np.linalg.norm(R)   
    _aux[:,1 ]  = _V[:,0] 
    
    #_ok         = np.isnan(_V[:,0])
    #_aux        = _aux[-_ok,:]
    _ok         = _aux[:,0 ] >= 0.5  
    _aux        = _aux[_ok,:]
    _zeros      = GetAllZeros( _aux[:,0], _aux[:,1] )

    print>> f_handle, '# Zeros:', " ".join([ str(_z) for _z in _zeros ])
    np.savetxt(f_handle, _aux)

    return _aux, _zeros   

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

  ## 
  if "PRESS" in KEYs :
    _aux = getPtsData( pvInput, _time, "PRESS" )
    vtks.Push( _aux[Id], "P")

  ## 
  if "TEMPE" in KEYs :
    _aux = getPtsData( pvInput, _time, "TEMPE" )
    vtks.Push( _aux[Id], "TW")

  ## 
  if "VISCO" in KEYs :
    _aux = getPtsData( pvInput, _time, "VISCO" )
    vtks.Push( _aux[Id], "MU")

  ##  Nu 
  if "GRATE" in KEYs :
    Nu  = getPtsData( pvInput, _time, "GRATE" )
    Nu  = Nu[Id,:]
    Nu  = np.linalg.norm(  Nu, axis=1 )
    vtks.Push(Nu, "dT")

    aux2 = np.trapz( Nu*D/abs(T2-T1), Phi ) / (2*np.pi)
    if UNITS : 
      if "TEMPE" in KEYs :
        _T1  = getPtsData( pvInput, _time, "TEMPE" )
        Nu  *= D / abs(T2 - _T1)
        aux3 = np.trapz( Nu, Phi ) / (2*np.pi)  
      else:  
        Nu  *= D / abs(T2 -  T1)
        aux3 = np.average(Nu)

    #print "|_ Nu:", aux2, aux3  
    #print>> Fout[0], aux2, aux3, 
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
    #for r in RutPhi: print>> Fout[0],r, 
    _Props.append( RutPhi.tolist() ) 

  return _Props 

#----------------------------------------------------------------# 
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
  T_in     = 288.15
  rho_in   = RHO( P0, T_in) 
  mu_in    = VIS(_Re_in, rho_in, _L0, _U0 ) 
  k_in     = KAPPA(mu_in,cp_in,Pr)
  nu_in    = mu_in / rho_in 

  ## props = props( _Tinf )
  _muinf   = (mu_in / MU01(T_in)) *  MU01(_Tinf)
  _rhoinf  = RHO( P0, _Tinf )
  _Reinf   = _U0 * _L0 * _rhoinf / _muinf
  _kinf    = (k_in / K01(T_in)) *  K01(_Tinf)
  _nuinf   =  _muinf / _rhoinf

  ## props = props( _Tf )
  _Tf      = _Tinf + C * (_Tw - _Tinf)  
  _muf     = (mu_in / MU01(T_in)) *  MU01(_Tf) 
  _rhof    = RHO( P0, _Tf )  
  _Reff    = _U0 * _L0 * _rhof / _muf 
  _kf      = (k_in / K01(T_in)) *  K01(_Tf)
  _nuf     =  _muf / _rhof  

  ## props = props( _Tw )
  _kw      = (k_in / K01(T_in)) *  K01(_Tw)   

  ## Nuf (Tf/T0)**-0.17 = A1 + B1 Ref**n1 
  ## (kw/kf) Nu (Tf/T0)**-0.17 = A1 + B1 ( nu0/nuf Re )**n1   
  #_props   = [ _Reinf, _Tw/_Tinf, _Reff, _Tf/_Tinf, _nuinf/_nuf, _kw/_kf ]
  _props   = [ _Reinf, _Tw/_Tinf, _Reff, _Tw/_Tinf, _nuinf/_nuf, _kw/_kf ]
  return _props  


#----------------------------------------------------------------# 
#----------------------------------------------------------------# 
#----------------------------------------------------------------# 
NUs   = [] 
Props = [] 
for ifin,fin in enumerate(Aux):
  fin = "/".join([FULL_PATH,fin])
  os.chdir(fin)
  print "# |_'%s' <--- " % fin

  #----------------------------------------------------------------# 
  DIA     = 1.0
  UNITS   = False
  NsiDic = KerDatNsi()
  if NsiDic :
    kappa = NsiDic["k"]  
    rho   = NsiDic["rho"]
    D     = NsiDic["L"]
    U     = NsiDic["U"]
    T1    = NsiDic["Tw"]
    T2    = NsiDic["Ti"]
    Re100 = NsiDic["Re"]

    DIA   = D
    UNITS = True

  TemDic = KerDatTem() 
  kappaS = float(TemDic["CONDUCTIVITY"][2]) 
  Ksf    = kappaS/kappa  

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

  KEYs   = plotOnIntersectionCurves1.GetPointDataInformation().keys(); #print KEYs 

  #----------------------------------------------------------------# 
  if 1 :
      ## Save all properties ...  
      Fout   = [ open("pv_props01.dat","w"), open("pv_drag.dat","w") ]
      print>> Fout[0], "## 1  2     3  4  5  6  7      8      9      [10      11   ] 12    13     14    "
      print>> Fout[0], "## T  T*U/L Cl Cd Nu Nu Theta1 Theta2 Theta3 [Theta4 Theta5] T/T0  Vxroot ks/kf "
      _Props = [] 
      _NUs   = [] 
      for ii, time in enumerate(Times):
        vtks = SetFsiVtkFiles( time, ii )  
        prop = GetAllProps(time, plotOnIntersectionCurves1)
        vtks.Save() 

        Fout2 = open('pv_streams%04d.dat' % (ii), "w")
        GetStreamsLines( time, streamTracer1, Fout2, DIA )
        Fout2.close()

        Fout3 = open('pv_line%04d.dat' % (ii), "w")
        V,Vruts = GetPloOverLine( time, cellDatatoPointData1, Fout3, DIA )
        Fout3.close()

        for p in prop: 
          for x in p: print>> Fout[0], x,
        print>> Fout[0], round(1.0,3),       ## T/T0  
        for x in Vruts: print>> Fout[0], x,  ## Vxroot 
        print>> Fout[0], Ksf,                ## ks/kf   
        print>> Fout[0]

        prop[1].pop(0) # remove time from ClCd  
        prop.pop(0)    # remove time 

        Prop075 = REeff( Re100, D, U, T2, T1, C=0.75 )
        Prop050 = REeff( Re100, D, U, T2, T1, C=0.50 )
        Prop028 = REeff( Re100, D, U, T2, T1, C=0.28 )

        prop0   = [[ Re100, Prop028[2], Prop050[2], NsiDic["Tr"] ]] 
        _Props.append( [[Ksf]] + prop0 + prop + [Vruts.tolist()] )

        _NUs.append(  [[Ksf]] + [Prop050] + [prop[1]]  )
#        _NUs.append(  [[Ksf]] + [Prop075] + [prop[1]]  )

      Fout[0].close()

      Props.append( sum(_Props[-1],[]) ) 
      NUs.append( sum(_NUs[-1],[]) )

      import shutil
      shutil.copy2("pv_props01.dat", "pv_props02.dat")
 
  #----------------------------------------------------------------# 

os.chdir(FULL_PATH)
Fout4 = open("%s_Reff.dat"% mySelf, "w")
print>> Fout4, "## 1      2      3     4    5  6  7  8  9  10     11     12     13     "
print>> Fout4, "## ks/kf  Re100  R050  R028 Tr Cl Cd Nu Nu Theta1 Theta2 Theta3 Vxroot "
for p in Props:
  for x in p: print>> Fout4, x,
  print>> Fout4 
Fout4.close() 

Fout5 = open("%s_NUeff.dat"% mySelf, "w")
print>> Fout5, "## 1      2    3      4     5      6        7      8   9      10     11     12     "
print>> Fout5, "## ks/kf  Re0  Tw/T0  Reff  Tf/T0  nu0/nuf  kw/kf  Nu  Nu \n##" 
print>> Fout5, "## (kw/kf) Nu  (Tf/T0)**-0.17 = A1 + B1 ( nu0/nuf Re )**n1 \n##" 
print>> Fout5, "##   ( $6 * $2 ):( $7 * $9 * $5**-0.17 )    \n##"
for p in NUs:
  for x in p: print>> Fout5, x,
  print>> Fout5
Fout5.close() 

#----------------------------------------------------------------# 
print "|_\t-->'python plotFrmPvSeparationPoint01.py'<--"
print "|_OK!"

#----------------------------------------------------------------# 
"""
SftpMarenostrum.py  /home/bsc21/bsc21704/z2016/REPOSITORY/TOOLs
"""
