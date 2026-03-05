# ANSVAR: Entry point. Starter Flask ved at kalde create_app() og køre serveren på port 5000.

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
