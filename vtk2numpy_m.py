#
#  VTK To Numpy Miguel (vtk_to_numpy_m.py) 
#
#  converts data from vtk to numpy objects
#  
#  Author: Miguel Zavala Ake
#

from paraview.simple import *
import numpy as np
import vtk

#----------------------------------------------------------------# 

def getPtsData( _Obj, _Prop=None, _fname=None, _T=-1.0):

    from paraview.numpy_support import vtk_to_numpy

    if(_Prop==None):
      print _Obj.GetPointDataInformation().keys()
      sys.exit()

    if _T==None: 
      _Obj.SMProxy.UpdatePipeline(    )
    else:
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
