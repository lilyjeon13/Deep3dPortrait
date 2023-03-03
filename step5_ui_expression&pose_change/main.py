from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap,QQuaternion
from PyQt5 import QtGui
import pyqtgraph.opengl as gl
import numpy
import os
import requests

import PyQt5
import sys
from pywavefront import visualization, Wavefront
    
vertss=[]
colorss = []
class MyApp(QWidget):
    def __init__(self):
        self.rotation = 0
        self.d = gl.GLViewWidget()
        self.bars = []
        self.m2 = gl.GLMeshItem()
        super().__init__()
        self.initUI()

    def initUI(self):
        mainBox = QHBoxLayout(self)
        mainBox.addStretch(1)
        # mainBox.addStretch(2)

        objectBox = QVBoxLayout()
        controlBox = QVBoxLayout()
        mainBox.addChildLayout(controlBox)
        mainBox.addLayout(objectBox)
        labels = ["Head Position", "Anger", "Contempt", "Disgust", "Fear", "Joy", "Sadness", "Surprise"]
        for i in range(8):
            hbox = QHBoxLayout()
            label = QLabel(labels[i])
            label.setFixedWidth(100)
            # label.setMinimumWidth(90)
            sld = QSlider(Qt.Horizontal)
            sld.setFixedWidth(420)
            # sld.set
            if i== 0:
                sld.setRange(-45,45)
                sld.valueChanged[int].connect(self.rotateChange)
                sld.setValue(0)
            else:
                sld.setRange(11*i-11,11*i-1)
                sld.valueChanged[int].connect(self.expressChanged)
                # print(sld.objectName())
                self.bars.append(sld)
            hbox.addWidget(label)
            hbox.addWidget(sld)
            objectBox.addLayout(hbox)

        imgs_path = "result"
        sorted_list = sorted(os.listdir(imgs_path))
        # change the sequence on files in imgs_path
        for i in range(int(len(sorted_list) / 11)) :
            select_list = sorted_list[11*i: 11*(i+1)]
            tempa = select_list[2]
            select_list[2:10] = select_list[3:11]
            select_list[10] = tempa
            sorted_list[11*i: 11*(i+1)] = select_list

        for filename in sorted_list:
            print(imgs_path + "/" + filename)
            box = Wavefront(os.path.join(imgs_path, filename))
            verts = numpy.empty((len(box.mesh_list[0].materials[0].vertices)//18,3,3),dtype=numpy.float32)        
            colors = numpy.ones((verts.shape[0], 3, 4))
            for i in range(0,len(box.mesh_list[0].materials[0].vertices)//18):
                verts[i][0]=box.mesh_list[0].materials[0].vertices[18*i+3:18*i+6]
                verts[i][1]=box.mesh_list[0].materials[0].vertices[18*i+9:18*i+12]
                verts[i][2]=box.mesh_list[0].materials[0].vertices[18*i+15:18*i+18]
                a=box.mesh_list[0].materials[0].vertices[18*i:18*i+3]
                a.append(1)
                colors[i][0]=a
                a[0:3]=box.mesh_list[0].materials[0].vertices[18*i+6:18*i+9]
                colors[i][1]=a
                a[0:3]=box.mesh_list[0].materials[0].vertices[18*i+12:18*i+15]
                colors[i][2]=a
            vertss.append(verts)
            colorss.append(colors)
            if len(vertss) == 1:
                self.m2 = gl.GLMeshItem(vertexes=verts,vertexColors=colors,smooth=False)
                # break
        self.d.addItem(self.m2)
        self.d.setFixedWidth(1000)
        self.d.setFixedHeight(1000)
        self.d.setCameraPosition(distance=5)
        self.d.opts["rotationMethod"]="quaternion"
        controlBox.addWidget(self.d)
        # objectBox
        self.setLayout(mainBox)
        self.setWindowTitle('My App')
        self.setGeometry(0,0,1600,1300)
        self.show()

    def rotateChange(self, value):
        q = QtGui.QQuaternion.fromEulerAngles(
                0, value, 0
        )
        self.d.opts['rotation'] = q
        self.d.update()

    def expressChanged(self, value):
        idx = (value//11)
        self.m2.setMeshData(vertexes=vertss[value],vertexColors=colorss[value])
        self.m2.meshDataChanged()
        for i in range(len(self.bars)):
            if i != idx:
                self.bars[i].setValue(i*10)
        self.d.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())

