# Titanic ML Prediction API

This project is a machine learning API that predicts whether a Titanic passenger would survive based on passenger data.

The application is built using Python and Flask, uses a PostgreSQL database, and runs in Docker containers.

---

## About

This project was built as part of my learning in backend development, machine learning and containerized applications.

The focus has been on:
- building a full ML pipeline
- experimenting with features
- integrating a trained model into an API

---

## Demo

Runs locally using Docker.

---

## Features

- Train a machine learning model on Titanic dataset
- Predict survival via REST API
- Store training runs in a PostgreSQL database
- Fully containerized with Docker and docker-compose

---

## Tech Stack

- Python
- Flask
- Scikit-learn
- PostgreSQL
- Docker

---

## Project Structure

- app/ → Flask API (routes, models, config)
- ml/ → Machine learning logic (training & prediction)
- migrations/ → Database migrations
- Dockerfile → Container setup
- docker-compose.yml → Multi-container setup

---

## Model

The model is an MLPClassifier (neural network) trained on ~800 passengers.

### Features used:
- pclass (passenger class)
- sex (encoded: female=0, male=1)
- age (filled with mean)
- sibsp (siblings/spouses aboard)
- parch (parents/children aboard)
- fare (ticket price)

### Preprocessing:
- Missing age values are filled with mean
- StandardScaler is used to normalize all features
- Data is split into 80% training / 20% testing

---

## How to run

1. Clone the repository

```bash
git clone https://github.com/Redhoudini/Titanic-ML-Prediction-API.git
cd Titanic-ML-Prediction-API

```

2. Start the application

```bash
docker-compose up --build
```

3. The API will be available at:

```bash
http://localhost:5000
```

## API Usage

### POST /api/predict

Required fields:
```JSON
{
  "pclass": 3,
  "sex": "male",
  "age": 25
}
```

Optional fields (defaults used if omitted):
```JSON
{
  "sibsp": 0,
  "parch": 0,
  "fare": 32
}
```


### Example Response



```JSON
{
  "prediction": 0,
  "survival_probability": 0.23
}
```

## Training Runs
All model runs are stored in the database and can be viewed at:

http://localhost:5000/runs

Each run includes:
- model name
- dataset
- accuracy
- loss



## What I learned

- Building a REST API with Flask
- Training and evaluating ML models
- Feature engineering and preprocessing
- Importance of scaling (StandardScaler)
- Managing experiments and results
- Running full-stack applications with Docker


## Future improvements

- Hyperparameter tuning (layers, neurons)
- Cross-validation instead of fixed split
- Better feature engineering
- Frontend interface for predictions
