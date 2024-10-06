from cmath import log
import importlib
from pyexpat import model
import numpy as np
import yaml
from logger import logger as logging
import os
import sys
from collections import namedtuple
from typing import List
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
from config import MetricInfoArtifact

def evaluate_classification_model(model_list: list, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, base_accuracy:float=0.6)->MetricInfoArtifact:
    """
    Description:
    This function compare multiple regression model return best model
    Params:
    model_list: List of model
    X_train: Training dataset input feature
    y_train: Training dataset target feature
    X_test: Testing dataset input feature
    y_test: Testing dataset input feature
    return
    It retured a frozen class 
    MetricInfoArtifact
    """
    try:
        
    
        index_number = 0
        metric_info_artifact = None
        for model in model_list:
            model_name = str(model)  #getting model name based on model object
            logging.info(f"{'>>'*30}Started evaluating model: [{type(model).__name__}] {'<<'*30}")
            
            #Getting prediction for training and testing dataset
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            test_acc = accuracy_score(y_test, y_test_pred)
            train_acc = accuracy_score(y_train,y_train_pred)


            #Calculating haarrmonic mean for train data
            # precision_train = precision_score(y_train, y_train_pred)
            # recall_train = recall_score(y_train, y_train_pred)
            # f1_train = f1_score(y_train, y_train_pred)

            #ncludes all the entries in a series using harmonic mean
            # Calculating harmonic mean of train_accuracy and test_accuracy in order to test with base accuracy
            model_accuracy = (2 * (train_acc * test_acc)) / (train_acc + test_acc)
            diff_test_train_acc = abs(test_acc - train_acc)

            #Calculating haarrmonic mean for test data
            precision_test = precision_score(y_test, y_test_pred)
            recall_test = recall_score(y_test, y_test_pred)
            f1_test = f1_score(y_test, y_test_pred)

            
            #logging all important metric
            logging.info(f"{'>>'*30} Score {'<<'*30}")
            logging.info(f"Train Score\t\t Test Score\t\t Average Score")
            logging.info(f"{train_acc}\t\t {test_acc}\t\t{model_accuracy}")

            logging.info(f"{'>>'*30} Loss {'<<'*30}")
            logging.info(f"Diff test train accuracy: [{diff_test_train_acc}].") 
            logging.info(f"Harmonic mean of test data: [{f1_test}].")


            #if model accuracy is greater than base accuracy and train and test score is within certain thershold
            #we will accept that model as accepted model
            
            if model_accuracy >= base_accuracy and diff_test_train_acc < 0.20:
                base_accuracy = model_accuracy
                metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                                                        model_object=model,
                                                        test_precision=precision_test,
                                                        test_recall = recall_test,
                                                        test_f1 = f1_test,
                                                        train_accuracy=train_acc,
                                                        test_accuracy=test_acc,
                                                        model_accuracy=model_accuracy,
                                                        index_number=index_number)

                logging.info(f"Acceptable model found {metric_info_artifact}. ")
            index_number += 1
        if metric_info_artifact is None:
            logging.info(f"No model found with higher accuracy than base accuracy")
        return metric_info_artifact
    except Exception as e:
        raise e

