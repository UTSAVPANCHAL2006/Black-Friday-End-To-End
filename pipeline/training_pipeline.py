from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreProcessing
from src.model_training import ModelTraining

def run_pipeline():
    
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    
    preprocessing = DataPreProcessing()
    train_path, test_path = preprocessing.initiate_data_preprocessing()
    
    training = ModelTraining()
    model_path = training.initiate_model_training()
    
    print(f"Pipeline completed. Model at: {model_path}")

if __name__ == "__main__":
    run_pipeline()