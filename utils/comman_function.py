import os
import pandas as pd 

from src.custom_exception import CustomException
from src.logger import get_logger


logger =  get_logger(__name__)

def load_data(path):
    try:
        logger.info("Data Loading")
        
        df = pd.read_csv(path)
        
        return df
    except Exception as e:
        logger.error(f"error while loading data{e}")
        raise CustomException("failed to load data",e)