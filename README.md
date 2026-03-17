# Titanic ML Prediction API

This project is a machine learning API that predicts whether a Titanic passenger would survive based on features such as passenger class and age.

The application is built using Python and Flask, uses a PostgreSQL database, and runs in Docker containers.

---

## About

This project was built as part of my learning in backend development, machine learning and containerized applications.

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

The model is trained using the Titanic dataset and predicts survival based on:

- Passenger class (pclass)
- Age

The model is a simple machine learning model built using scikit-learn.

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

## Example API request

POST request:

```JSON
{
  "pclass": 3,
  "age": 25
}
```


## What I learned

- Building a REST API using Flask

- Training and using a machine learning model

- Integrating a PostgreSQL database

- Using Docker to run multi-container applications

- Structuring a backend project


## Future improvements

- Improve model accuracy
- Add input validation
- Add frontend interface

