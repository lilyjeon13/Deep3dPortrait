# from PyQt5.QtWidgets import *
# from PyQt5.QtCore import Qt
# from PyQt5.QtGui import QPixmap
# from PyQt5 import QtGui
# import pyqtgraph.opengl as gl
import numpy as np
import os
import sys
# from pywavefront import visualization, Wavefront
from ipygany import Scene, PolyMesh
from pygel3d import hmesh, gl_display as gl

# m = hmesh.load('./1_123456.obj')
# viewer = gl.Viewer()
# viewer.display(m)

# class MyApp(QWidget):
    
#     def __init__(self):
#         self.rotation = 0
#         self.d = gl.GLViewWidget()

#         super().__init__()
#         self.initUI()

#     def initUI(self):
#         sld = QSlider(Qt.Horizontal)
#         sld.setRange(-90,90)
#         sld.setValue(0)
#         sld.valueChanged[int].connect(self.changeValue)
#         objectBox = QVBoxLayout()
#         controlBox = QVBoxLayout()
#         objectBox.addWidget(sld)
#         root_path = os.path.dirname(__file__)
#         box = Wavefront(os.path.join(root_path, './1_123456.obj'))
#         verts = numpy.empty((len(box.mesh_list[0].materials[0].vertices)//18,3,3),dtype=numpy.float32)        
#         colors = numpy.ones((verts.shape[0], 3, 4))
#         for i in range(0,len(box.mesh_list[0].materials[0].vertices)//18):
#             verts[i][0]=box.mesh_list[0].materials[0].vertices[18*i+3:18*i+6]
#             verts[i][1]=box.mesh_list[0].materials[0].vertices[18*i+9:18*i+12]
#             verts[i][2]=box.mesh_list[0].materials[0].vertices[18*i+15:18*i+18]
#             a=box.mesh_list[0].materials[0].vertices[18*i:18*i+3]
#             a.append(1)
#             # print(a)
#             colors[i][0]=a
#             a[0:3]=box.mesh_list[0].materials[0].vertices[18*i+6:18*i+9]
#             colors[i][1]=a
#             a[0:3]=box.mesh_list[0].materials[0].vertices[18*i+12:18*i+15]
#             colors[i][2]=a

#         m2 = gl.GLMeshItem(vertexes=verts,vertexColors=colors,smooth=False,shader='balloon',drawEdges=False)
#         # m2.translate(-5, 5, 0)
#         # fooMd = gl.MeshData.sphere(rows=10, cols=10)
#         # print(fooMd.faceCount)
#         # c = gl.GLMeshItem(meshdata=fooMd, smooth=False, shader='shaded', glOptions='opaque')
#         # print(c.faceCount())
#         # d.pan(10,10,10,relative='global')
#         # print(d.getViewport)
#         # self.d.orbit(0,self.rotation)
#         # d.setCameraPosition(rotation=90)
#         self.d.addItem(m2)
#         self.d.setCameraPosition(QtGui.QVector3D(0,0,0),10,0,0,0)
#         # self.d.orbit(220,90)
#         objectBox.addWidget(self.d)
#         mainBox = QHBoxLayout()
#         mainBox.addLayout(objectBox)
#         mainBox.addLayout(controlBox)
#         self.setLayout(mainBox)
#         self.setWindowTitle('Box Layout')
#         self.setGeometry(1000,1000,1000,1000)
#         self.show()


#     def changeValue(self, value):
#         # self.rotation=value
#         print(value)
#         # print(self.d.opts)


# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     ex = MyApp()
#     sys.exit(app.exec_())