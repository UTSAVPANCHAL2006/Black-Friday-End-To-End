# 🏗️ System & MLOps Architecture

This document outlines the architecture of the **User Purchase Prediction** project. The system is designed following modern **MLOps (Machine Learning Operations)** best practices, ensuring clear separation of concerns between model training, tracking, deployment, and serving.

---

## 📊 High-Level Architecture Diagram

```mermaid
graph TD
    %% Styling
    classDef primary fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef storage fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef action fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;

    subgraph "1. Model Training Pipeline"
        MLE([ML Engineer]) -->|Executes| Ingest[Data Ingestion]
        Ingest -->|Raw Data| Preprocess[Data Preprocessing <br> Feature Engineering]
        Preprocess -->|Processed Data| Train[Model Training <br> Scikit-Learn]
        class Ingest,Preprocess,Train action;
    end

    subgraph "2. Tracking & Model Registry"
        Train -->|Logs Params, Metrics <br> & Encoders| MLflow[(MLflow Server <br> sqlite:///mlflow.db)]
        Train -->|Registers Model| Registry[(MLflow Model Registry)]
        Preprocess -->|Passes Encoders <br> to Training| Train
        class MLflow,Registry storage;
    end

    subgraph "3. CI/CD & Version Control"
        MLE -->|Git Push / PR| Repo[GitHub Repository]
        Repo -->|Triggers| Actions[GitHub Actions CI]
        Actions -->|Builds Image| Docker[(Docker Image)]
    end

    subgraph "4. Inference & Serving"
        Docker -.->|Deploys to| Env[Production Server]
        Registry -.->|Loads Model & Encoders| API
        Env --- API[FastAPI Backend]
        API <-->|JSON Payload| UI[Streamlit Frontend]
        User([End User]) <-->|Interacts| UI
        class API,UI primary;
    end
```

---

## ⚙️ Component Breakdown

### 1. Model Training Pipeline (`src/`)
This is the core data science pipeline responsible for creating the model.
- **Data Ingestion:** Reads the raw dataset (`Data/`) and performs initial validation.
- **Preprocessing:** Handles missing values, performs `LabelEncoding`, and maps categorical IDs using `Frequency Encoding`. 
- **Model Training:** Fits the machine learning algorithms to the processed data.

### 2. Tracking & Model Registry (`mlflow.db` & `mlruns/`)
Ensures absolute reproducibility and strict versioning of the machine learning models.
- **MLflow Tracking:** Backed by a local SQLite database (`mlflow.db`), it logs all experiment parameters, evaluation metrics (like MAE, MSE, R2), and training timestamps.
- **Model Registry:** The finalized XGBoost model is formally registered under the name `UserPurchasePredictionModel`. Crucially, the **data encoders** (`le_city.pkl`, `le_gender.pkl`, `product_freq.pkl`) are logged as MLflow artifacts alongside the registered model, ensuring that the exact transformations used during training are perfectly coupled with the inference model to prevent feature mismatch.

### 3. CI/CD & Version Control (`.github/workflows/`)
Automates the software engineering aspects of the ML lifecycle.
- Every push to the `main` branch triggers **GitHub Actions**, which automatically builds the project into an isolated **Docker Image**. 
- This guarantees that the code running on the developer's laptop behaves identically in production.

### 4. Inference & Serving (`app.py` & `main.py`)
The production applications that serve the model to end-users.
- **FastAPI:** Acts as the high-performance inference engine. It loads the artifacts in memory and exposes a RESTful endpoint (`/predict`).
- **Streamlit:** A clean, user-friendly frontend that gathers user input, formats it as JSON, and queries the FastAPI backend.
