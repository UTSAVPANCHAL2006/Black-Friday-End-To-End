import os
import sys
import numpy as np 
import pandas as pd 
import pickle

from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import * 

import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder

logger = get_logger(__name__)


class DataPreProcessing():
    
    def __init__(self):
        self.processed_dir        = PROCESSED_DIR
        self.processed_train_path = PROCESSED_TRAIN_FILE_PATH
        self.processed_test_path  = PROCESSED_TEST_FILE_PATH
        self.train_path           = TRAIN_FILE_PATH
        self.test_path            = TEST_FILE_PATH
        
        self.le_gender = LabelEncoder()
        self.le_city = LabelEncoder()
        self.product_freq = None
        
        
        
    def feature_engineering(self, df):
        
        try:
            
            logger.info("Feature Engineering...")
            
            df["Product_Category_2"] = df["Product_Category_2"].fillna(0)
            df["Product_Category_3"] = df["Product_Category_3"].fillna(0)
            df = df[df["Purchase"].notnull()]
            
            df["Stay_In_Current_City_Years"] = df["Stay_In_Current_City_Years"].replace("4+", 4)
            df["Stay_In_Current_City_Years"] = df["Stay_In_Current_City_Years"].astype(int)
            
            age_map = {
            "0-17": 0,
            "18-25": 1,
            "26-35": 2,
            "36-45": 3,
            "46-50": 4,
            "51-55": 5,
            "55+": 6
            }

            df["Age"] = df["Age"].map(age_map)
            
            
            logger.info("Completed.....")
            
            return df
            
        except Exception as e:
            logger.error(f"Error While During Feature Engineering..{e}")
            raise CustomException(f"Failed while During Feature Eng",e)
        
    def preprocessing(self, df):
        
        try:
            
            product_id = df['Product_ID']
            
            X = df.drop(columns=['Purchase', 'User_ID', 'Product_ID'])
            y = df['Purchase']
            
            logger.info(f"Create A Product_id varible {product_id.value_counts().head()}")
            
            return X, y, product_id  
            
        except Exception as e:
            logger.error(f"Error during Feature Engineering: {e}")
            raise CustomException(e, sys)
        
    def labelencoder(self, train_df, test_df):
        
        try:
            
            logger.info("Label Encoding start..")
            
            train_df['Gender'] = self.le_gender.fit_transform(train_df['Gender'])
            test_df['Gender'] = self.le_gender.transform(test_df['Gender'])
            
            train_df['City_Category'] = self.le_city.fit_transform(train_df['City_Category'])
            test_df['City_Category'] = self.le_city.transform(test_df['City_Category'])
            
            
            logger.info("Label Encoding completed.")
            return train_df, test_df
        
        except Exception as e:
            logger.error(f"Error during encoding: {e}")
            raise CustomException(e, sys)
            
        
    def frequency_encode(self, train_df, test_df, product_id_train, product_id_test):
        try:
            
            logger.info("Frequency Encoding Product ID....")
            
            freq = product_id_train.value_counts()
            self.product_freq = freq.to_dict()
            
            train_df["Product_ID_FE"] = product_id_train.map(freq).values
            test_df["Product_ID_FE"]  = product_id_test.map(freq).fillna(0).values

            logger.info("Frequency Encoding completed.")
            return train_df, test_df
        
        except Exception as e:
            logger.error(f"Error during frequency encoding: {e}")
            raise CustomException(e, sys)
    
    def save_artifacts(self , X_train , y_train , X_test , y_test):
        try:
            
            os.makedirs(self.processed_dir, exist_ok=True)
            
            train_final = pd.DataFrame(X_train)
            train_final["Purchase"] = y_train.values
            train_final.to_csv(self.processed_train_path, index=False)
            
            test_final = pd.DataFrame(X_test)
            test_final["Purchase"] = y_test.values
            test_final.to_csv(self.processed_test_path, index=False)

            logger.info(f"Processed train saved to {self.processed_train_path}")
            logger.info(f"Processed test  saved to {self.processed_test_path}")

        except Exception as e:
            logger.error(f"Error saving artifacts: {e}")
            raise CustomException(e, sys)
        
    def save_encoders(self):
        try:
            encoder_dir = "artifacts/encoders"
            os.makedirs(encoder_dir, exist_ok=True)

            pickle.dump(self.le_gender, open(os.path.join(encoder_dir, "le_gender.pkl"), "wb"))
            pickle.dump(self.le_city,   open(os.path.join(encoder_dir, "le_city.pkl"),   "wb"))
            pickle.dump(self.product_freq, open(os.path.join(encoder_dir, "product_freq.pkl"), "wb"))

            logger.info("Encoders saved to artifacts/encoders/")

        except Exception as e:
            logger.error(f"Error saving encoders: {e}")
            raise CustomException(e, sys)
        
    def initiate_data_preprocessing(self):
        logger.info("Starting Data Preprocessing Pipeline...")

        try:
            train_df = pd.read_csv(self.train_path)
            test_df  = pd.read_csv(self.test_path)
            logger.info(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")

            
            train_df = self.feature_engineering(train_df)
            test_df  = self.feature_engineering(test_df)

            product_id_train = train_df["Product_ID"]
            product_id_test  = test_df["Product_ID"]

            drop_cols = ["Purchase", "User_ID", "Product_ID"]
            X_train = train_df.drop(columns=drop_cols)
            y_train = train_df["Purchase"]
            X_test  = test_df.drop(columns=drop_cols)
            y_test  = test_df["Purchase"]

            X_train, X_test = self.labelencoder(X_train, X_test)

            X_train, X_test = self.frequency_encode(X_train, X_test, product_id_train, product_id_test)

            self.save_artifacts(X_train, y_train, X_test, y_test)
            self.save_encoders()

            logger.info("Data Preprocessing Pipeline completed successfully.")
            return self.processed_train_path, self.processed_test_path

        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    preprocessing = DataPreProcessing()
    train_path, test_path = preprocessing.initiate_data_preprocessing()
    print(f"Processed Train : {train_path}")
    print(f"Processed Test  : {test_path}")