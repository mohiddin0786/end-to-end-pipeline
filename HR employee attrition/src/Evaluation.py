import pandas as pd   
from sklearn.metrics import classification_report,confusion_matrix

# Here we evaluate the model using classification report and confusion matrix.

class Evaluation:
    def __init__(self,y_test,y_pred):
        self.y_test = y_test
        self.y_pred = y_pred
    def classification_report(self):
        return classification_report(self.y_test,self.y_pred)
    def confusion_matrix(self):
        return confusion_matrix(self.y_test,self.y_pred)
    
