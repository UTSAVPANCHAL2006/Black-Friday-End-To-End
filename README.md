# 🛒 User Purchase Prediction - End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2)

An end-to-end Machine Learning pipeline that predicts the purchase amount of a user based on their demographics and product details. This project implements modern **MLOps practices**, including a modular code structure, experiment tracking, API deployment, and an interactive web user interface.

---

## 🌟 Key Features

- **Modular Architecture:** Cleanly separated code for data ingestion, preprocessing, and model training.
- **Experiment Tracking:** Integrated with **MLflow** to track models, hyperparameters, and training metrics.
- **REST API:** A high-performance inference API built using **FastAPI**.
- **Interactive UI:** A web-based frontend application built using **Streamlit**.
- **CI/CD & Docker:** Includes a `Dockerfile` and GitHub Actions workflows for automated deployment.

---

## 📂 Repository Structure

```text
.
├── src/                    # Source code for the ML pipeline
│   ├── data_ingestion.py   # Data loading and splitting
│   ├── data_preprocessing.py # Feature engineering & encoding
│   ├── model_training.py   # Model training logic with MLflow
│   ├── logger.py           # Custom logging configuration
│   └── custom_exception.py # Custom error handling
├── artifacts/              # Contains saved datasets, models, and encoders
├── notebooks/              # Jupyter notebooks for exploratory data analysis
├── app.py                  # FastAPI backend application
├── main.py                 # Streamlit frontend application
├── Dockerfile              # Docker image configuration
└── .github/workflows/      # CI/CD deployment pipelines
```

---

## 🚀 Getting Started

Follow these steps to set up the project locally on your machine.

### 1. Clone the Repository
```bash
git clone https://github.com/UTSAVPANCHAL2006/user-purchase-prediction.git
cd user-purchase-prediction
```

### 2. Setup Virtual Environment
Create a virtual environment and install the required dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Start the Backend API (FastAPI)
Launch the REST API server that loads the trained model and serves predictions:
```bash
uvicorn app:app --reload
```
The API will be available at [http://localhost:8000](http://localhost:8000). You can test the endpoints via the Swagger UI at [http://localhost:8000/docs](http://localhost:8000/docs).

### 4. Start the Frontend App (Streamlit)
Open a new terminal window, activate your virtual environment, and run:
```bash
streamlit run main.py
```
The UI will automatically open in your browser at [http://localhost:8501](http://localhost:8501).

---

## 🧠 Retraining the Model

If you want to modify the data or hyperparameters and retrain the model, run the pipeline scripts:

```bash
# 1. Preprocess data and generate new encoders
python -m src.data_preprocessing

# 2. Train the model (Tracked automatically by MLflow)
python -m src.model_training
```

---

## 🐳 Docker Deployment

To build and run the project using Docker without installing Python dependencies locally:

```bash
# Build the Docker image
docker build -t purchase-prediction-app .

# Run the container
docker run -p 8000:8000 purchase-prediction-app
```

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/UTSAVPANCHAL2006/user-purchase-prediction/issues).

## 📝 License
This project is open source and available under the [MIT License](LICENSE).
