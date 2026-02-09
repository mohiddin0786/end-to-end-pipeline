from Data_Loader import Data_Loader
from preprocesser import Preprocess
from Train import Train 
from Evaluation import Evaluation

file_path = 'data/HR_attrition_data.csv'

def run_pipeline(file_path):
    #loading data 

    data_loader = Data_Loader(file_path)
    data = data_loader.load_data()

    #preprocessing data

    processer = Preprocess(data)
    data_encoded = processer.encode()

    #spliting data into train and test set

    x_train,x_test,y_train,y_test = processer.split(data_encoded)

    # training data into Logistic regression.

    train = Train(x_train,y_train)
    logistic_model = train.logistic_regression()

    # training data into Random forest model.

    random_forest_model = train.random_forest()

    # evaluating the both models 
    eval_logistic_train = Evaluation(y_train,logistic_model.predict(x_train))
    eval_logistic_test= Evaluation(y_test,logistic_model.predict(x_test))

    eval_random_forest_train = Evaluation(y_train,random_forest_model.predict(x_train))
    eval_random_forest_test = Evaluation(y_test,random_forest_model.predict(x_test))

    print("Logistic Regression - Training Data Report:")
    print(eval_logistic_train.classification_report())
    print("\nLogistic Regression - Test Data Report:")
    print(eval_logistic_test.classification_report())

    print("\nRandom Forest - Training Data Report:")
    print(eval_random_forest_train.classification_report())
    print("\nRandom Forest - Test Data Report:")
    print(eval_random_forest_test.classification_report())
 
if __name__ == "__main__":
    run_pipeline(file_path)
