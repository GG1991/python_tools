#!/usr/bin/env python

from paraview.simple import *
import numpy as np
import scipy as sc
import scipy
import os
import vtk
import sys
import vtk2numpy_m as vn

#----------------------------------------------------------------# 

fin   = "macro_t_1.pvtu"
PVTU  = XMLPartitionedUnstructuredGridReader(FileName=fin)
KEYs  = PVTU.GetPointDataInformation().keys()
print KEYs
KEYs  = PVTU.GetCellDataInformation().keys()
print KEYs
Times = np.array(PVTU.TimestepValues)

PlotOverLine = PlotOverLine( Input=PVTU, guiName="PlotOverLine", Source="High Resolution Line Source" )
PlotOverLine.Source.Point2 = [30.0, 30.0, 0.0]
PlotOverLine.Source.Point1 = [30.0, 00.0, 0.0]
PlotOverLine.Source.Resolution = 1000
PlotOverLine.UpdatePipeline()

key   = "stress"
stress = vn.getPtsData( PlotOverLine, key )

key   = "residual"
Res   = vn.getPtsData( PlotOverLine, key )

key   = "displ"
Displ = vn.getPtsData( PlotOverLine, key )

key   = "energy"
Ene   = vn.getPtsData( PlotOverLine, key )

key   = "arc_length"
leng  = vn.getPtsData( PlotOverLine, key )

aux = np.zeros( (leng.shape[0],5) )
aux[:,3] = Ene[:]
aux[:,2] = stress[:,0]
aux[:,1] = Displ[:,0]
aux[:,0] = leng 

np.savetxt("data_overline.dat", aux )
force = np.trapz(stress[:,0],leng)

print "force = ", force ,"K = ", force / Displ[0,0]

#----------------------------------------------------------------# 
