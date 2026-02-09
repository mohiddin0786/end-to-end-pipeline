import pandas as pd   
from sklearn.model_selection import train_test_split

# For HR attrition dataset, we have to encode data to fit in logistic or randomforest model. So we use get_dummies to encode the data 
# We also split the data into train and test set in this class.


class Preprocess:
    def __init__(self,data):
        self.data = data
    def encode(self):
        data_encoded = pd.get_dummies(self.data,drop_first=True)
        return data_encoded
    def split(self,data_encoded,target = "left",test_size=0.2):
        features = data_encoded.columns.drop(target) 
        x = data_encoded[features]
        y = data_encoded[target]
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_size,random_state=100)
        return x_train,x_test,y_train,y_test
    
