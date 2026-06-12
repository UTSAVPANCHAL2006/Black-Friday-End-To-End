import os

from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
from utils.comman_function import load_data

import pandas as pd 
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)

class DataIngestion():
    
    
    def __init__(self):
        self.raw_dir         = RAW_DIR
        self.raw_file_path   = RAW_FILE_PATH
        self.train_file_path = TRAIN_FILE_PATH
        self.test_file_path  = TEST_FILE_PATH
        
    def initiate_data_ingestion(self):
        
        logger.info("Starting Data Ingestion Process")
        
        try:
            
            # Using relative path so it works everywhere (like in Docker or another PC)
            data_path = os.path.join("Data", "data.csv")
            df = load_data(data_path)
            logger.info(f"Data Loaded {df.shape[0]} Row & {df.shape[1]} Columns")
            
            os.makedirs(self.raw_dir , exist_ok=True)
            logger.info(f"Created Directory :{self.raw_dir}")
            
            df.to_csv(self.raw_file_path ,index=False)
            logger.info(f"Raw Data Saved To :{self.raw_file_path}")
            
            train_df , test_df = train_test_split(df ,test_size=0.2, random_state=42)
            
            train_df.to_csv(self.train_file_path, index=False)
            test_df.to_csv(self.test_file_path, index=False)
            
            logger.info(f"Train Size {train_df.shape} | Test Size {test_df.shape}")
            logger.info("Data Ingestion Completed")
            
            return self.train_file_path , self.test_file_path
            
        except Exception as e:
            logger.error(f" Error During Data Ingestion : {e}")
            raise CustomException(f"failed while during data ingestion", e)
        
if __name__=="__main__":
    
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    print(f"Train : {train_path}")
    print(f"Test  : {test_path}")    