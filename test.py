#necessary libraries

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.widget import Widget
from kivy.base import EventLoop
from kivy.config import Config
from kivy.clock import Clock
from kivy.uix.popup import Popup

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, recall_score, precision_score

from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.app import runTouchApp

import pickle as cPickle
import pandas as pd
import numpy as np

#importing the model 
with open('model.pkl', 'rb') as fid:
    model = cPickle.load(fid)

# The features that are identified as most important by various feature selection methods

need = ['Medical_History_15', 'Medical_History_1', 'Medical_Keyword_15',
       'Medical_Keyword_3', 'Medical_History_2', 'Medical_History_23',
       'Employment_Info_2', 'Medical_History_4', 'Medical_Keyword_23',
       'Medical_Keyword_25', 'Product_Info_4', 'Medical_Keyword_38',
       'Product_Info_2', 'Product_Info_3', 'Medical_History_16',
       'Medical_Keyword_42', 'Medical_Keyword_1', 'BMI',
       'Employment_Info_3', 'InsuredInfo_6', 'Medical_Keyword_37',
       'Medical_Keyword_24', 'Ins_Age', 'Medical_History_39', 'Wt',
       'Medical_Keyword_40', 'Medical_Keyword_22', 'Medical_Keyword_47',
       'Medical_Keyword_10', 'Medical_Keyword_43', 'Medical_Keyword_46',
       'Medical_Keyword_33', 'Insurance_History_4', 'Medical_Keyword_35',
       'Insurance_History_3', 'Medical_Keyword_16', 'Employment_Info_6',
       'Medical_History_13', 'Medical_Keyword_27', 'Medical_Keyword_31',
       'Medical_History_33', 'Medical_History_29', 'Medical_History_41',
       'Medical_History_6', 'Medical_Keyword_30', 'Medical_Keyword_9',
       'InsuredInfo_1', 'Insurance_History_2', 'Medical_Keyword_4',
       'Medical_Keyword_18', 'Medical_Keyword_19', 'Medical_Keyword_36',
       'Medical_Keyword_12', 'Medical_Keyword_34', 'Medical_Keyword_13',
       'Medical_Keyword_28', 'Family_Hist_1', 'Medical_Keyword_14',
       'Medical_Keyword_11', 'Medical_Keyword_7', 'InsuredInfo_7',
       'Medical_Keyword_21', 'Medical_History_28', 'Medical_History_9',
       'Medical_Keyword_41', 'Medical_Keyword_17', 'InsuredInfo_5',
       'Family_Hist_4', 'Medical_Keyword_29', 'Medical_Keyword_8',
       'Insurance_History_1', 'Employment_Info_1', 'Medical_Keyword_5',
       'Medical_Keyword_26', 'Medical_Keyword_44', 'Medical_History_34',
       'Medical_History_18', 'Employment_Info_5', 'Medical_History_40',
       'Medical_Keyword_39', 'Product_Info_6', 'Medical_History_30',
       'Medical_Keyword_6', 'Medical_Keyword_2', 'Medical_History_21',
       'Employment_Info_4', 'Insurance_History_8', 'Medical_Keyword_20',
       'Medical_Keyword_32', 'Medical_History_25', 'Family_Hist_2',
       'InsuredInfo_3', 'Ht', 'Medical_History_12', 'Medical_History_3',
       'Medical_History_27', 'Medical_History_36', 'Medical_History_19',
       'Medical_History_26', 'Product_Info_1', "Response"]


def ml_part(data, flag):
    '''
    arguments :

    data - data that is given to the mmodel for testing
    Flag - To detemine the kind of operation needed

    This part will scale the input data and will predict the ouput by rendering it against the model

    '''
    scaler = MinMaxScaler()
    if flag ==1:
        f=1
        if data != '':
            data = pd.read_csv(data, index_col = 'Id') 
            for col in data.columns:
                data[col] = data[col].fillna(0)
                if col not in need:
                    data = data.drop(col, axis = 1)
            data['Product_Info_2'] = pd.factorize(data['Product_Info_2'])[0]
            x_test = data.drop('Response', axis = 1).iloc[:, :]
            y_test = data['Response'].iloc[:]
    elif flag ==2:
        f=2
        data = data.split(' ')
        data[2] = pd.factorize(data[2])[0]
        x_test = data.drop('Response', axis = 1).iloc[:, :]
        y_test = data['Response'].iloc[:]
        print (data)
    else:
        f=3
        data = data[:]

    for col in data.columns:
        data[col] = data[col].fillna(0)
        if col not in need:
            data = data.drop(col, axis = 1)
    x_test = pd.DataFrame(scaler.fit_transform(x_test))
    pred = model.predict(x_test)

    # Below part is to compute the evaluation metrics and to present them in a user interface
    metric1 = f1_score(pred, y_test, average="weighted")*100
    metric2 = precision_score(pred, y_test, average="weighted")*100
    metric3 = recall_score(pred, y_test, average="weighted")*100
    if flag == 1:
        btn3=Button(text='F1 Accuracy :  '+str(metric1)+ '\nPrecision      :  ' + str(metric2)+'\nRecall            :  '+str(metric3),font_size=20)
    elif flag ==2:
        btn3=Button(text='Response is '+str(metric),font_size=20)

    pop=Popup(content=btn3, title='output',size_hint=(None, None), size=(400, 300))
    pop.open()
    btn3.bind(on_press=pop.dismiss)

with open('name.txt', 'r') as f:
    names = f.readlines()[0].split('\t')
    
# main class that contains code for GUI creation
class Widget(Widget):   
    def __init__(self, **kwargs):
        super(Widget, self).__init__()
        EventLoop.ensure_window()
        Window.size = (1280,700)
        Window.clearcolor = (1, 1, 1, 1)
        EventLoop.window.title = self.title = 'Insurance Prediction'
        layout = GridLayout(cols=8, spacing=10,  size_hint_y=None,   row_default_height=40)
        layout.bind(minimum_height=layout.setter('height'))
        txt = []
        btn1 = Button(text="Submit",italic=True, color =(1,1,1,1))
        btn2 = Button(text="Submit",italic=True, color =(1,1,1,1))
        layout.add_widget(Label(text=" ",italic=True, bold=True))
        layout.add_widget(Label(text=" ",italic=True, bold=True))
        layout.add_widget(Label(text=" ",italic=True, bold=True))
        layout.add_widget(Label(text=" ",italic=True, bold=True))
        layout.add_widget(Label(text=" ",italic=True, bold=True)) 
        layout.add_widget(Label(text=" ",italic=True, bold=True))
        layout.add_widget(Label(text=" ",italic=True, bold=True))
        layout.add_widget(Label(text=" ",italic=True, bold=True)) 
        layout.add_widget(Label(text=" ",italic=True, bold=True)) 
        layout.add_widget(Label(text=" ",italic=True, bold=True))
        layout.add_widget(Label(text="path for test csv : ",italic=True, bold=True, font_size = 15, color =(0,0,0,0)))
        self.txt1 = TextInput(multiline=False, font_size=20)    
        layout.add_widget(self.txt1)
        layout.add_widget(btn1)
        btn1.bind(on_press=lambda *a:ml_part(self.txt1.text, 1))
        layout.add_widget(Label(text=" ",italic=True, bold=True, color =(0,0,0,0)))
        lb1 = Label(text= ' ',italic=True, bold=True, color =(0,0,0,0))
        layout.add_widget(lb1) 
        layout.add_widget(Label(text=" ",italic=True, bold=True)) 
        layout.add_widget(Label(text=" ",italic=True, bold=True))
        layout.add_widget(Label(text=" ",italic=True, bold=True))
        layout.add_widget(Label(text=" ",italic=True, bold=True))
        layout.add_widget(Label(text="or ",italic=True, bold=True,color = (0,0,0,0)))
        layout.add_widget(Label(text=" ",italic=True, bold=True))
        layout.add_widget(Label(text=" ",italic=True, bold=True))
        layout.add_widget(Label(text=" ",italic=True, bold=True))
        layout.add_widget(Label(text=" ",italic=True, bold=True)) 
        layout.add_widget(Label(text=" ",italic=True, bold=True)) 
        ctr = 0
        for col in names:
            dummy1=Label(text=" ",italic=True, bold=True)
            dummy2=Label(text=" ",italic=True, bold=True)
            lbl=Label(text=col,italic=True, bold=True,color =(0,0,0,0))
            txt1=TextInput(multiline=False, font_size=20,size_hint=(.2, .6) )
            txt.append(txt1.text)
            layout.add_widget(lbl)
            layout.add_widget(txt1)
            ctr+=1
            if ctr == 3:
                ctr = 0
                layout.add_widget(dummy1)
                layout.add_widget(dummy2)
        layout.add_widget(Label(text=" ",italic=True, bold=True)) 
        btn3 = Button(text="Submit",italic=True)
        layout.add_widget(btn3)
        layout.add_widget(Label(text="Response : ",italic=True, bold=True,color =(0,0,0,0)))
        layout.add_widget(Label(text=" ",italic=True, bold=True))
        layout.add_widget(Label(text=" ",italic=True, bold=True)) 
        layout.add_widget(Label(text=" ",italic=True, bold=True))
        layout.add_widget(Label(text=" ",italic=True, bold=True))
        layout.add_widget(Label(text=" ",italic=True, bold=True))
        layout.add_widget(Label(text=" ",italic=True, bold=True,color = (0,0,0,0)))
        layout.add_widget(Label(text=" ",italic=True, bold=True))
        layout.add_widget(Label(text=" ",italic=True, bold=True))
        layout.add_widget(Label(text=" ",italic=True, bold=True))
        layout.add_widget(Label(text=" ",italic=True, bold=True)) 
        layout.add_widget(Label(text=" ",italic=True, bold=True)) 
        btn3.bind(on_press=lambda *a:ml_part(txt, 3))
        root = ScrollView(size_hint=(1, None), size=(Window.width, Window.height))
        root.add_widget(layout)
        runTouchApp(root)

class App(App):
    def build(self):
        return Widget()

if __name__ == '__main__':
    App().run()


