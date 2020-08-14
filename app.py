# -*- coding: utf-8 -*-
"""
Created on Mon May 13 04:10:01 2019
#all you need to install
pip install kivy
pip install docutils
pip install pygmentspypiwin32
pip install kivy.deps.sdl2
pip install kivy.deps.glew
pip install kivy.deps.gstreamer
pip install graden
garden install matplotlib
"""
import matplotlib
matplotlib.use("module://kivy.garden.matplotlib.backend_kivy")
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageFile
import numpy as np
from scipy import misc
import keras as kr

from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas,FigureCanvasKivyAgg

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label


from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
import os
#%%
fig,ax=plt.subplots()
fig2,ax2=plt.subplots()
ImageFile.LOAD_TRUNCATED_IMAGES = True

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    cwdir=ObjectProperty(None)
    
class MyApp(App):
    
    
    def build(self):
        self.title='Math Tutor'
        box = GridLayout(cols=2)
        
        label1=Label(text='[b]Your Chart[/b]',
                     font_size='20sp',size_hint_y=None, height=50,markup=True)
        label2=Label(text='[b]Chart Classifier[/b]',
                     font_size='20sp',size_hint_y=None, height=50,markup=True)
        

        box.add_widget(label1)
        box.add_widget(label2)
        
        
        box.add_widget(FigureCanvasKivyAgg(fig))
        box.add_widget(FigureCanvasKivyAgg(fig2))
        
        openBtn=Button(text='Open',size_hint_y=None, height=50)
        openBtn.bind(on_release=self.show_load)
        box.add_widget(openBtn)
          
        
        
        classify=Button(text='Ask Tutor',size_hint_y=None, height=50)
        classify.bind(on_release=self.class_image)
        box.add_widget(classify)
        
        self.imgG=None
        return box
    
    

    def show_load(self,obj):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup,cwdir=os.getcwd())        
        self._popup = Popup(title="Load file", content=content,size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, filename):
        if filename:
            img=Image.open(os.path.join(path, filename[0]))
            ax.imshow(np.array(img))
            fig.canvas.draw()
            self.dismiss_popup()
            self.imgG=img         

    def dismiss_popup(self):
        self._popup.dismiss()
        
    def class_image(self,obj):
        if self.imgG!=None:
            X_test = []
            img=self.imgG.convert("L")
            img = misc.imresize(img , (200,200))
            img = img/255
            X_test.append(img)
            X_test = np.array(X_test)
            X_test =X_test.reshape(len(X_test),200,200,1)
            model = kr.models.load_model('D:/Anaconda3/mycode/chart7/model_cnn7_90.h5')
            pred = model.predict_classes(X_test) 
            answer = pred[0]
            for i in range(7):
                if i == answer:
                    predict =  Image.open('D:/Anaconda3/mycode/chart6/picture/answer ('+str(i)+').jpg')
            ax2.imshow(np.array(predict))
            fig2.canvas.draw()            
            #d = {0:"bar",1:"histogram",2:"pie",3:"bubble",4:"scatter",5:"line",6:"another picture "}
            
        

      
            
Factory.register('LoadDialog', cls=LoadDialog)
MyApp().run()
