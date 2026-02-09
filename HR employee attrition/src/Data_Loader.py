import warnings
import pandas as pd  
warnings.filterwarnings('ignore')

# Here we load data using pandas

class Data_Loader:
    def __init__(self,path):
        self.path = path
    def load_data(self):
        data = pd.read_csv(self.path)
        return data
