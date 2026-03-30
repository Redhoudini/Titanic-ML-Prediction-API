import csv
import psycopg2

DB_CONFIG = {
    "dbname": "appdb",
    "user": "appuser",
    "password": "apppassword",
    "host": "db",
    "port": "5432"
}

CSV_PATH = "titanic_800.csv"


def import_titanic_data():
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS titanic_passengers (
                id SERIAL PRIMARY KEY,
                passenger_id INTEGER,
                survived INTEGER,
                pclass INTEGER,
                sex TEXT,
                age FLOAT,
                sibsp INTEGER,
                parch INTEGER,
                fare FLOAT
            );
        """)

        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            rows = []
            for row in reader:
                rows.append((
                    int(row["PassengerId"]),
                    int(row["Survived"]),
                    int(row["Pclass"]),
                    (row["Sex"]),
                    float(row["Age"]) if row["Age"] else None,
                    int(row["SibSp"]),
                    int(row["Parch"]),
                    float(row["Fare"])
                ))

        cur.execute("DELETE FROM titanic_passengers;")

        cur.executemany("""
            INSERT INTO titanic_passengers (passenger_id, survived, pclass, sex, age, sibsp, parch, fare)
            VALUES (%s, %s, %s, %s, %s, %s ,%s, %s);
        """, rows)

        conn.commit()
        print(f"Importerede {len(rows)} rækker til titanic_passengers")

    except Exception as e:
        print(f"Fejl: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            cur.close()
            conn.close()


if __name__ == "__main__":
    import_titanic_data()