import pandas as pd   
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Fitting training dataset into Logistic regression and Random forest model.

class Train:
    def __init__(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
    def logistic_regression(self):
        model = LogisticRegression(max_iter=1000).fit(self.x_train,self.y_train)
        return model
    def random_forest(self):
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            class_weight='balanced'
        ).fit(self.x_train,self.y_train)
        return model
    
