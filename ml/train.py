import psycopg2
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report
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
        SELECT "passenger_id", pclass, age, survived
        FROM titanic_passengers
        ORDER BY "passenger_id"
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df


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
    data = load_data()

    print("Rows loaded:", len(data))
    print(data.head())
    print(data.describe(include="all"))

    # y = target
    yvalues = data[["survived"]].copy()

    # x = features
    xdata = data.drop("survived", axis=1).copy()

    # passenger_id giver normalt ikke mening som feature
    xdata.drop("passenger_id", axis=1, inplace=True)

    # Fill missing age med gennemsnit inden for samme pclass
    xdata["age"] = xdata.groupby("pclass")["age"].transform(
        lambda col: col.fillna(col.mean())
    )

    # Plot 1: age vs pclass
    plt.figure()
    plt.scatter(xdata["age"], xdata["pclass"], alpha=0.5)
    plt.xlabel("Age")
    plt.ylabel("Pclass")
    plt.title("Age vs Pclass")
    plt.savefig("app/static/titanic_plot_age_pclass.png")
    plt.close()

    # Plot 2: age vs survived
    plt.figure()
    plt.scatter(xdata["age"], yvalues["survived"], alpha=0.5)
    plt.xlabel("Age")
    plt.ylabel("Survived")
    plt.title("Age vs Survived")
    plt.savefig("app/static/titanic_plot_age_survived.png")
    plt.close()

    # Plot 3: pclass vs survived
    plt.figure()
    plt.scatter(xdata["pclass"], yvalues["survived"], alpha=0.5)
    plt.xlabel("Pclass")
    plt.ylabel("Survived")
    plt.title("Pclass vs Survived")
    plt.savefig("app/static/titanic_plot_pclass_survived.png")
    plt.close()

    print("Rows after fillna by class:", len(xdata))

    xtrain = xdata.iloc[:400]
    xtest = xdata.iloc[400:]
    ytrain = yvalues.iloc[:400]
    ytest = yvalues.iloc[400:]

    print("xtrain shape:", xtrain.shape)
    print("xtest shape:", xtest.shape)
    print("ytrain shape:", ytrain.shape)
    print("ytest shape:", ytest.shape)

    # Scale kun X
    scaler = StandardScaler()
    scaler.fit(xtrain)

    xtrain_scaled = scaler.transform(xtrain)
    xtest_scaled = scaler.transform(xtest)

    model = MLPClassifier(
        hidden_layer_sizes=(8, 8),
        max_iter=1000,
        random_state=0
    )

    model.fit(xtrain_scaled, ytrain.values.ravel())

    joblib.dump(model, "ml/models/titanic_mlp_mean_age.pkl")

    predictions = model.predict(xtest_scaled)
    probabilities = model.predict_proba(xtest_scaled)

    matrix = confusion_matrix(ytest, predictions)
    print("Confusion matrix:")
    print(matrix)

    print("Classification report:")
    print(classification_report(ytest, predictions))

    tn, fp, fn, tp = matrix.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    loss = log_loss(ytest, probabilities)

    save_training_run(
        model_name="MLPClassifier_mean_age_by_class_400_rest_scaled",
        dataset="titanic_passengers",
        accuracy=float(accuracy),
        loss=float(loss)
    )

    print("Training færdig")
    print("TN:", tn, "FP:", fp, "FN:", fn, "TP:", tp)
    print("Accuracy:", accuracy)
    print("Loss:", loss)


if __name__ == "__main__":
    train()