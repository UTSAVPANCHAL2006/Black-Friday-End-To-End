import os 
import sys

import pandas as pd 
import pickle

from src.custom_exception import CustomException
from src.logger import get_logger
from config.path_config import *

import xgboost 
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining():
    
    def __init__(self):
        self.model_output_path = MODEL_OUTPUT_PATH
        self.train_path = PROCESSED_TRAIN_FILE_PATH
        self.test_path = PROCESSED_TEST_FILE_PATH
        
        self.model_params = {
            "n_estimators" : 500,
            "learning_rate":0.05,
            "max_depth":8,
            "subsample":0.8,
            "colsample_bytree":0.8,
            "random_state":42,
            "n_jobs":-1
        }
        
    def prepare_data(self):
        
        try:
            
            logger.info("Loading Preprocessing data...")
            
            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)
            
            X_train = train_df.drop(columns=["Purchase"])
            y_train = train_df["Purchase"]
            X_test  = test_df.drop(columns=["Purchase"])
            y_test  = test_df["Purchase"]            
            
            logger.info(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
            return X_train, y_train, X_test, y_test
    
        except Exception as e:
            logger.error(f"Failed While during prepare data{e}")
            raise CustomException(e, sys)
        
    def train_model(self,X_train, y_train):
        
        try:
            logger.info("Training XBG Model...")
            
            model = XGBRegressor(**self.model_params)
            model.fit(X_train,y_train)
            
            logger.info("Model Training Completed")
            return model
        except Exception as e:
            logger.error(f"failed while during model training : {e}")
            raise CustomException(e, sys)
        
    def evaluate_model(self , model , X_train , y_train , X_test , y_test):
        
        try:
            
            logger.info("Evaluating model")
            
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_r2 = r2_score(y_train , y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            metrics = {
                "MAE"      : mae,
                "MSE"      : mse,
                "r2"       : r2,
                "train_r2" : train_r2,
                "test_r2"  : test_r2
            }
            
            logger.info(f"MAE  : {mae:.4f}")
            logger.info(f"MSE  : {mse:.4f}")
            logger.info(f"R2   : {r2:.4f}")
            logger.info(f"Train R2 : {train_r2:.4f}")
            logger.info(f"Test  R2 : {test_r2:.4f}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error while during evaluating..{e}")
            raise CustomException(e , sys)
        
    def save_model(self, model):
        
        try:
            
            os.makedirs(self.model_output_path , exist_ok=True)
            
            model_path = os.path.join(self.model_output_path, "model.pkl")
            pickle.dump(model, open(model_path, "wb"))
            
            return model_path
        
        except Exception as e:
            logger.error(f"Error while during saving model {e}")
            raise CustomException(e , sys)
        
    def initiate_model_training(self ):
        
        try:
            
            logger.info("Statring Model Training Pipeline...")
            
            X_train, y_train, X_test, y_test = self.prepare_data()
            
            # Use the local SQLite database for the MLflow Model Registry
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            
            with mlflow.start_run():
                
                model = self.train_model(X_train,y_train)
                
                metrics = self.evaluate_model(model,X_train,y_train,X_test,y_test)
                
                mlflow.log_params(self.model_params)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(
                    model, 
                    "Xgboost_model", 
                    registered_model_name="UserPurchasePredictionModel"
                )
                
                # Log the encoder artifacts
                mlflow.log_artifacts("artifacts/encoders", artifact_path="encoders")

                logger.info("MLflow logging completed.")
                
                model_path = self.save_model(model)
            
            logger.info("Model Training Pipeline completed successfully.")
            return model_path
        
        except Exception as e:
            logger.error(f"Error in model training pipeline: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    trainer = ModelTraining()
    model_path = trainer.initiate_model_training()
    print(f"Model saved at: {model_path}")