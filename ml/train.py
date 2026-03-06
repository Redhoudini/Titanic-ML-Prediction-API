import psycopg2
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
from datetime import datetime

DB_CONFIG = {
    "dbname": "appdb",
    "user": "appuser",
    "password": "apppassword",
    "host": "db",
    "port": "5432"
}


def load_data():
    conn = psycopg2.connect(**DB_CONFIG)

    query = """
        SELECT pclass, age, survived
        FROM titanic_passengers
        WHERE age IS NOT NULL
    """

    df = pd.read_sql(query, conn)
    conn.close()


    X = df[["pclass", "age"]]
    y = df["survived"]

    return X, y




def save_training_run(model_name, dataset, accuracy, loss):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO training_runs (model_name, dataset, accuracy, loss, created_at)
        VALUES (%s, %s, %s, %s, %s)
    """, (model_name, dataset, accuracy, loss, datetime.utcnow()))

    conn.commit()
    cur.close()
    conn.close()


def train():
    X, y = load_data()

    import matplotlib.pyplot as plt

    plt.figure()

    dead = y == 0
    survived = y == 1

    plt.scatter(X["pclass"][dead], X["age"][dead], label="Died", alpha=0.5)
    plt.scatter(X["pclass"][survived], X["age"][survived], label="Survived", alpha=0.5)

    plt.xlabel("Passenger Class")
    plt.ylabel("Age")
    plt.title("Titanic Survival")
    plt.legend()

    plt.savefig("app/static/titanic_plot.png")
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MLPClassifier(
        hidden_layer_sizes=(8, 4),
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "ml/models/titanic_mlp_pclass_age_drop_missing.pkl")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob)

    save_training_run(
        model_name="MLPClassifier_mean_age_by_class_when_NULL",
        dataset="titanic_train_500_age_passengerclass.csv",
        accuracy=float(accuracy),
        loss=float(loss)
    )

    print("Training færdig")
    print("Accuracy:", accuracy)
    print("Loss:", loss)






if __name__ == "__main__":
    train()