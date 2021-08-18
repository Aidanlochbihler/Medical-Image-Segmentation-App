import sys
import matplotlib
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import PyQt5.QtWidgets as QtW
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.widgets import RectangleSelector

matplotlib.use('Qt5Agg')
import os

import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from matplotlib import colors

import tensorflow as tf

gpus= tf.config.experimental.list_physical_devices('GPU') #IS NEEDED FOR VRAM OVERLOADS
tf.config.experimental.set_memory_growth(gpus[0], True)

cmap1 = colors.ListedColormap(['none', 'red'])
cmap2 = colors.ListedColormap(['none','green'])

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class PopUp(QMessageBox):
    def __init__(self, text):
        super(PopUp, self).__init__()

        self.setIcon(QMessageBox.Information)
        self.setText(text)
        self.setWindowTitle("")
        self.setStandardButtons(QMessageBox.Close)
        returnValue = self.exec()



class MainWindow(QtW.QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 layout'
        self.left = 700
        self.top = 200
        self.width = 1000
        self.height = 900
        self.file_path = ''
        self.model_path = ''
        self.x_coord_temp = 0 #Variables used to hold the coords of the clicked position
        self.y_coord_temp = 0 
        self.z_coord_temp = 0
        self.x_coord_final = None #Variables used to hold the confirmed coords of the clicked position
        self.y_coord_final = None
        self.z_coord_start_final = None
        self.z_coord_end_final = None
        self.coords_confirmed = False
        self.image_index = 0
        self.volume_index = 0
        self.case = 0 
        self._main = QtW.QWidget()
        self.setCentralWidget(self._main) 
        
        #---------------------Canvas Settings------------------------
        self.fig = matplotlib.figure.Figure(figsize=(5,5), dpi=100, frameon = False)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis('off')
        self.arr_rank = 0 #This is used to determine whether the npy is One 2D, Multi 2D, or Multi 3D
        self.image = 0
        self.image_shape = 0
        self.pred = 0
        self.canvas_image = 0
        self.pred_image = 0
        self.overlay = False
        
        self.model_shape = 0
        self.model = 0
        self.rectangle_shape_x = 0 #Rect W&H are in units pixels  
        self.rectangle_shape_y = 0 
        
        self.bound_on = True
        
        #self.ax.imshow(self.image[1,:,:,0], cmap='gray', interpolation='none')
        self.scale = 'scaled'
        self.ax.axis(self.scale) #USE self.ax.axis([xmin, xmax, ymin, ymax]) For zooming in if ever needed
        #self.canvas.mpl_connect("button_press_event", self.on_press)

        self.canvas.draw()
        
        #This is whats used to select the Coord
        self.rs = RectangleSelector(self.ax, self.select_point,
                                    drawtype='none', 
                                    useblit=True,
                                    button=[1],  #Uses Left Mouse Click
                                    minspanx=0, minspany=0,
                                    maxdist = 0,
                                    spancoords='pixels',
                                    interactive=False)
        
        
        self.button_style = 'QPushButton{ background-color: white; border-style: outset; border-width: 1px; border-radius: 2px; border-color: black; font: bold 12px; min-width: 4em; padding: 2px;} QPushButton:hover:!pressed{ background-color: rgb(240, 240, 240); border-style: outset; border-width: 1px; border-radius: 2px; border-color: black; font: bold 12px; min-width: 4em; padding: 2px;}' 
        self.btn_file_open = QPushButton('Open File', self)
        self.btn_file_open.setStyleSheet(self.button_style)
        self.btn_file_open.clicked.connect(self.open_file)
        self.label_file_open = QLabel('', self)
        self.label_file_open.setWordWrap(True)
        #self.btn_file_open.setMaximumWidth(40) #Limit the width of a button
        
        #-----------------------SIDE BAR---------------------
        self.btn_mode_open = QPushButton('Open Model', self)
        self.btn_mode_open.clicked.connect(self.open_model)
        self.btn_mode_open.setStyleSheet(self.button_style)
        self.label_model_open = QLabel('', self)
        self.label_model_open.setWordWrap(True)


        self.btn_confirm_coords = QPushButton('Confirm Boundary', self)
        self.btn_confirm_coords.setStyleSheet(self.button_style)
        self.btn_confirm_coords.clicked.connect(self.confirm_coords)
        self.label_confirm_coords = QLabel('', self)
        self.label_confirm_coords.setWordWrap(True)
        
        self.btn_clear_coords = QPushButton('Clear Boundary', self)
        self.btn_clear_coords.setStyleSheet(self.button_style)
        self.btn_clear_coords.clicked.connect(self.clear_coords)
        
        self.radio_boundon = QRadioButton("Boundary On")
        self.radio_boundon.toggled.connect(self.boundary_on)
        self.radio_boundon.setChecked(True)
        self.radio_boundon.setStyleSheet('QRadioButton{ background-color: white; border-style: outset; border-width: 1px; border-radius: 2px; border-color: black; font: bold 12px; min-width: 4em; padding: 2px;}')
        
        self.radio_boundoff = QRadioButton("Boundary Off")
        self.radio_boundoff.toggled.connect(self.boundary_off)
        self.radio_boundoff.setStyleSheet('QRadioButton{ background-color: white; border-style: outset; border-width: 1px; border-radius: 2px; border-color: black; font: bold 12px; min-width: 4em; padding: 2px;}')

        self.label_list_box_images = QLabel('Images', self)
        self.list_box_images = QComboBox(self)
        self.list_box_images.setStyleSheet('QComboBox{ background-color: white; border-style: outset; border-width: 1px; border-radius: 0px; border-color: black; font: 12px; min-width: 4em; padding: 2px;}')
        self.list_box_images.activated.connect(self.onclick_list_box_images)
  
        self.label_list_box_volumes = QLabel('Volumes', self)
        self.list_box_volumes = QComboBox(self)
        self.list_box_volumes.setStyleSheet('QComboBox{ background-color: white; border-style: outset; border-width: 1px; border-radius: 0px; border-color: black; font: 12px; min-width: 4em; padding: 2px;}')
        self.list_box_volumes.activated.connect(self.onclick_list_box_volumes)
  
    
  
        self.btn_predict = QPushButton('GO!!!', self)
        self.btn_predict.setStyleSheet(self.button_style)
        self.btn_predict.clicked.connect(self.predict)
        
        #------------------------ARROWS--------------------------------
        self.arrow_right = QToolButton()                                                                      
        self.arrow_right.setArrowType(Qt.RightArrow)
        self.arrow_right.setAutoRaise(True)
        self.arrow_right.setIconSize(QSize(40,80))
        self.arrow_right.clicked.connect(self.press_arrow_right)

        self.arrow_left = QToolButton()                                                                      
        self.arrow_left.setArrowType(Qt.LeftArrow)
        self.arrow_left.setAutoRaise(True)
        self.arrow_left.setIconSize(QSize(40,80))
        self.arrow_left.clicked.connect(self.press_arrow_left)
        
        self.label_z_coord = QLabel('Images', self)
        self.label_z_coord.setAlignment(Qt.AlignCenter)
        self.label_z_coord.setFont(QFont("Arial", 10, QFont.Bold))
        self.label_z_coord.setMargin(0)
        #self.label_z_coord.setStyleSheet("border :0px solid black;padding :0px")
        self.label_z_coord.setWordWrap(True)
        
        '''
        #For Fun Can Add App Icon
        
        app_icon = QtGui.QIcon()
        app_icon.addFile(r'path\img.jpg', QtCore.QSize(32,32))
        app.setWindowIcon(app_icon)
        '''
        self.initUI()

        
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.ax.axis(self.scale)
        #-------------------SIDEBAR-------------------
        self.sidebar_layout = QVBoxLayout()

        
        self.sidebar_layout.addWidget(self.btn_file_open)
        self.sidebar_layout.addWidget(self.label_file_open)
        
        self.sidebar_layout.addWidget(self.btn_mode_open)
        self.sidebar_layout.addWidget(self.label_model_open)
        
        self.sidebar_layout.addSpacerItem(QSpacerItem(0, 15, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.sidebar_layout.addWidget(QHLine())
        self.sidebar_layout.addSpacerItem(QSpacerItem(0, 80, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        self.sidebar_layout.addWidget(self.radio_boundon)
        self.sidebar_layout.addSpacerItem(QSpacerItem(0, 100, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.sidebar_layout.addWidget(self.radio_boundoff)
        
        self.sidebar_layout.addSpacerItem(QSpacerItem(0, 100, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.sidebar_layout.addWidget(self.btn_confirm_coords)
        self.sidebar_layout.addWidget(self.label_confirm_coords)
        
        self.sidebar_layout.addWidget(self.btn_clear_coords)
        self.sidebar_layout.addSpacerItem(QSpacerItem(0, 90, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.sidebar_layout.addWidget(QHLine())
        self.sidebar_layout.addSpacerItem(QSpacerItem(0, 90, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        self.sidebar_layout.addWidget(self.label_list_box_images)
        self.sidebar_layout.addWidget(self.list_box_images)
        self.label_list_box_images.hide()
        self.list_box_images.hide()
        
        
        self.sidebar_layout.addWidget(self.label_list_box_volumes)
        self.sidebar_layout.addWidget(self.list_box_volumes)
        self.label_list_box_volumes.hide()
        self.list_box_volumes.hide()

        
        
        self.sidebar_layout.addSpacerItem(QSpacerItem(0, 500, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.sidebar_layout.addWidget(self.btn_predict)


        self.sidebar_widget = QWidget()
        self.sidebar_widget.setLayout(self.sidebar_layout)
        
        #-------------------Center Canvas Plot-------------------

        self.canvasLayout = QtW.QHBoxLayout(self._main)
        self.canvasLayout.addWidget(self.arrow_left, 1)
        self.canvasLayout.addWidget(self.canvas, 3)
        self.canvasLayout.addWidget(self.arrow_right, 1)
        self.arrow_right.show()
        self.arrow_left.show() 
        
        self.canvas_widget = QWidget()
        self.canvas_widget.setLayout(self.canvasLayout)
        
        self.mainPlotLayout = QtW.QVBoxLayout(self._main)
        self.mainPlotLayout.addWidget(self.label_z_coord, 0.5)
        self.mainPlotLayout.addWidget(self.canvas_widget, 1000)
        self.mainPlot_widget = QWidget()
        self.mainPlot_widget.setLayout(self.mainPlotLayout)
        
        #-------------------Combining Side + Canvas-------------------
        self.centerLayout = QtW.QHBoxLayout(self._main)
        self.centerLayout.addWidget(self.sidebar_widget, 1)
        self.centerLayout.addWidget(self.mainPlot_widget, 3)

        self.center_widget = QWidget()
        self.center_widget.setLayout(self.centerLayout)
        
        #--------------------MAIN-------------------
        self.mainLayout = QtW.QGridLayout(self._main)
        self.mainLayout.addWidget(self.center_widget, 0,0)
        
        
        self.setLayout(self.mainLayout)
        self.show()

    def select_point(self, eclick, erelease):
        if self.bound_on:
            
            x, y = eclick.xdata, eclick.ydata
            x = int(x)
            y = int(y)
            #self.canvas_image[y][x]= 255 #COORDS TRANSFER TO NUMPY
            
            self.ax.clear()
            self.ax.imshow(self.canvas_image, cmap='gray', interpolation='none') 
            
            if self.overlay: 
                self.ax.imshow(self.pred_image, cmap=cmap1, alpha = 0.4) 
                
            self.ax.axis('off')
            self.ax.axis(self.scale)
            self.ax.scatter(x,y, c='r', s=40)
            
            self.rectangle_shape_x = 60
            self.rectangle_shape_y = 60
            
            '''
            #USE THIS FOR ACTUAL VERSION******
            try:
                self.rectangle_shape_x = int(self.model_shape[2])
                self.rectangle_shape_y = int(self.model_shape[1])
            except:
                PopUp('Open A Model First ')
            '''
    
            self.rectangle = plt.Rectangle((x-(self.rectangle_shape_x/2),y-(self.rectangle_shape_y /2)), self.rectangle_shape_x, self.rectangle_shape_y , fill=False, color = "red") 
            self.ax.add_patch(self.rectangle)
            
            print("(%3.2f, %3.2f)" % (x, y))
            
            self.x_coord_temp = x #Variables used to hold the coords of the clicked position
            self.y_coord_temp = y
            self.z_coord_temp = self.image_index
            
            self.canvas.draw()
        else:
            pass
        
    def confirm_coords(self):
        self.x_coord_final = self.x_coord_temp #Variables used to hold the confirmed coords of the clicked position
        self.y_coord_final = self.y_coord_temp 
        
        
        self.model_shape = (None, 2, 256, 256, 1)

        if len(self.model_shape) <=4: #If the model is a rank less that or equal to 4 then it has no Z axis (ie its 2D images)
            self.label_confirm_coords.setText("Boundary confirmed:\nX,Y Coords: (%3.2f, %3.2f)" % (self.x_coord_final, self.y_coord_final))
        else: #Else we need to predict 3D Volumes so a Z range equal to the size from the model will be used
            middle = self.z_coord_temp 
            z_leng = self.model_shape[1]
            
            if (z_leng % 2):
                self.z_coord_start_final = middle-int(z_leng/2)
                self.z_coord_end_final = middle+int(z_leng/2)
                print('bottom: ', self.z_coord_start_final, ' top: ', self.z_coord_end_final)
                
            else:
                self.z_coord_start_final = middle-int(z_leng/2)
                self.z_coord_end_final = middle+int(z_leng/2)-1
                print('bottom: ', self.z_coord_start_final, ' top: ', self.z_coord_end_final)

            self.label_confirm_coords.setText("Boundary confirmed: \nZ-Range: (%3.2f, %3.2f) \nX,Y Coords: (%3.2f, %3.2f)" % (self.z_coord_start_final, self.z_coord_end_final, self.x_coord_final, self.y_coord_final))

    

        self.coords_confirmed = True
        self.label_confirm_coords.adjustSize()
        
    def clear_coords(self):
        self.x_coord_final = None
        self.y_coord_final = None
        
        self.ax.clear()
        self.ax.imshow(self.canvas_image, cmap='gray', interpolation='none') 
        
        if self.overlay: 
            self.pred_image = self.pred[self.image_index,:,:,0]
            self.ax.imshow(self.pred_image, cmap=cmap1, alpha = 0.4) 
            
        self.ax.axis('off')
        self.ax.axis(self.scale)
        self.canvas.draw()
        
        self.label_confirm_coords.setText("")
        self.label_confirm_coords.adjustSize()
    

    def update_canvas_image(self):
        #Case: (256,256,1) One 2D images
        self.ax.clear()
        if (self.arr_rank == 3): 
            print('Case 0:', self.image_shape)
            if self.image_shape[-1] == 1:
                self.slices = np.s_[:,:,0] 
            else:
                self.slices = np.s_[:,:,:]
            self.arrow_right.hide()
            self.arrow_left.hide()

        #Case: (n,256,256,1) Multiple 2D images
        elif (self.arr_rank == 4): 
            print('Case 1:', self.image_shape)
            if self.image_shape[-1] == 1:
                self.slices = np.s_[self.image_index,:,:,0]
            else:
                self.slices = np.s_[self.image_index,:,:,:] 
            
        #Case: (n, p, 256,256,1) Multiple 3D images SPECIAL CASE
        elif (self.arr_rank == 5): 
            print('Case 2:', self.image_shape)

            if self.image_shape[-1] == 1:
                self.slices = np.s_[self.volume_index,self.image_index,:,:,0]
            else:
                self.slices = np.s_[self.volume_index,self.image_index,:,:,:]
                
        self.canvas_image = self.image[self.slices]
        self.ax.imshow(self.canvas_image, cmap='gray', interpolation='none') 

        if self.overlay: 
            self.pred_image = self.pred[self.slices]
            self.ax.imshow(self.pred_image, cmap=cmap1, alpha = 0.4) 
        
        try:
            self.ax.add_patch(self.rectangle)
        except:
            pass
        
        self.ax.axis('off')
        self.ax.axis(self.scale)
        self.canvas.draw()             

    def open_file(self):
        filename = QFileDialog.getOpenFileName()
        path = filename[0]
        self.file_path = path
        self.image = np.load(self.file_path)
        #self.image = np.random.rand(5,4,256,256) #**TEST MULTI 3D IMAGES***
        
        self.image_shape = self.image.shape

        if self.image_shape[-1] > 3: #This is used to account for the case of an image not having a color depth ex: (256,256) --> (256,256,1)
            self.image = np.expand_dims(self.image, axis=-1)
            self.image_shape = self.image.shape

        if self.image_shape[0] == 1: #if npy has (1,256,256,1) --> (256,256,1)
            self.image = np.squeeze(self.image, axis=0)
            self.image_shape = self.image.shape
        
        self.arr_rank = len(self.image_shape)
        
        print(self.file_path)
        self.overlay = False
        self.image_index = 0
        self.update_canvas_image()
        #-------------------Populates the List boxes and shows them if the case is right-------------------------
        if (self.arr_rank == 4): 
            list_items = [str(x) for x in range(self.image_shape[0])]
            self.list_box_images.addItems(list_items)
            self.label_list_box_images.show()
            self.list_box_images.show()
            self.case = 0
        #Case: (n, p, 256,256,1) Multiple 3D images SPECIAL CASE
        elif (self.arr_rank == 5): 
            list_items = [str(x) for x in range(self.image_shape[1])]
            self.list_box_images.addItems(list_items)
            self.label_list_box_images.show()
            self.list_box_images.show()
            
            list_items = [str(x) for x in range(self.image_shape[0])]
            self.list_box_volumes.addItems(list_items)
            self.label_list_box_volumes.show()
            self.list_box_volumes.show()
            self.case = 1
        #---------------------------------------------------------------------------------------------------------    
        
        self.label_file_open.setText('Image File Path: '+ self.file_path)
        self.label_file_open.adjustSize()
        self.label_z_coord.setText('Volume ' + str(self.volume_index)+', Image ' + str(self.image_index)) 
        self.label_z_coord.adjustSize()
        
        
        
    def open_model(self):
        filename = QFileDialog.getOpenFileName()
        path = filename[0]
        self.model_path = path
        self.label_model_open.setText('Model File Path: '+ self.model_path)
        self.label_model_open.adjustSize()
        print(self.model_path)
        
        obj = {'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef}
        #self.model = load_model(self.model_path, custom_objects= obj)
        self.model = tf.keras.models.load_model(self.model_path, custom_objects= obj)
        
        self.model_shape = self.model.layers[0].get_output_at(0).get_shape().as_list()
        print('Model Input Shape:', self.model_shape)

    def boundary_on(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            #print(radioButton.text())
            self.bound_on = True
            self.btn_confirm_coords.show()
            self.label_confirm_coords.show()
            self.btn_clear_coords.show()
            
    def boundary_off(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            #print(radioButton.text())
            self.bound_on = False
            self.btn_confirm_coords.hide()
            self.label_confirm_coords.hide()
            self.btn_clear_coords.hide()
            


    def press_arrow_right(self):
        print('Right', self.image.shape[self.arr_rank-4])
        if self.image_index < (self.image.shape[self.arr_rank-4]-1):
            self.image_index += 1
            self.label_z_coord.setText('Image '+ str(self.image_index)) 
            self.label_z_coord.adjustSize()
            self.label_z_coord.setAlignment(Qt.AlignCenter)
        
        self.update_canvas_image()
        
    def press_arrow_left(self):
        print('Left')
        if self.image_index > 0:
            self.image_index -= 1
            self.label_z_coord.setText('Image '+ str(self.image_index)) 
            self.label_z_coord.adjustSize()
            self.label_z_coord.setAlignment(Qt.AlignCenter)
            
            
        self.update_canvas_image()
        
    def onclick_list_box_images(self):
        print(int(self.list_box_images.currentText()))

        self.image_index = int(self.list_box_images.currentText())
        self.label_z_coord.setText('Volume ' + str(self.volume_index)+', Image ' + str(self.image_index)) 
        self.label_z_coord.adjustSize()
        self.label_z_coord.setAlignment(Qt.AlignCenter)
        
        self.update_canvas_image()
        
    def onclick_list_box_volumes(self):
        print(int(self.list_box_volumes.currentText()))
        self.volume_index = int(self.list_box_volumes.currentText())
        self.label_z_coord.setText('Volume ' + str(self.volume_index)+', Image ' + str(self.image_index)) 
        self.update_canvas_image()
        
        
    def predict(self):
        '''
        if self.bound_on = True:
            if self.coords_confirmed = True:
            Use the confirmed X,Y,Z to crop the numpy
            **NOTE: will need self.z_coord_end_final+1 when actually croping
            else:
                PopUp('Please Confirm Coords First')
        else:
            pass
                
        '''
        print('Predicting in Progress')
        dir_path = os.path.dirname(self.file_path)
        
        imgs = np.load(self.file_path)
        imgs = imgs.astype('float32')
        print('shape', imgs.shape)
        print('max pred', np.max(imgs))
        imgs /= 255
        


        #pre = np.ndarray((total, image_rows, image_cols, channels), dtype=np.float32)
        self.pred = self.model.predict(imgs, batch_size=1)
        self.pred = (self.pred > 0.1).astype('float32')
        
        
        print('Prediction Complete', self.pred.shape)
        np.save(dir_path + '/prediction.npy', self.pred)
            
        self.overlay = True
        self.pred_image = self.pred[self.image_index,:,:,0]
            
        self.ax.imshow(self.canvas_image, cmap='gray', interpolation='none') 
        self.ax.imshow(self.pred_image, cmap=cmap1, alpha = 0.4)
        self.ax.axis('off')
        self.ax.axis(self.scale)
        self.canvas.draw()
        
        PopUp('Prediction Complete')
            
            
        
        
if __name__ == '__main__':
    app = QtCore.QCoreApplication.instance()
    if app is None: app = QtW.QApplication(sys.argv)
    win = MainWindow()
    app.aboutToQuit.connect(app.deleteLater)
    app.exec_()
    


'''
Can you this to catch errors and Print in message box instead of crashing the whole app
import traceback

try:
    1/0
except Exception:
    traceback.print_exc()

'''






