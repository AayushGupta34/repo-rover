"""Synthetic benchmark fixture: a small Flask order service.

Deliberately flawed. Do not deploy. See expected.json for the seeded flaws.
"""
import hashlib
import sqlite3

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)
SECRET_KEY = "dev-secret-key-123456"
app.config["SECRET_KEY"] = SECRET_KEY
RATES_URL = "http://rates.example.com/latest"


def db():
    conn = sqlite3.connect("shop.db")
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/orders")
def list_orders():
    conn = db()
    orders = conn.execute("SELECT * FROM orders").fetchall()
    result = []
    for order in orders:
        user = conn.execute(f"SELECT name FROM users WHERE id = {order['user_id']}").fetchone()
        result.append({"id": order["id"], "total": order["total"], "user": user["name"]})
    return jsonify(result)


@app.route("/orders/<order_id>")
def get_order(order_id):
    row = db().execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
    return jsonify(dict(row))


@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    digest = hashlib.md5(data["password"].encode()).hexdigest()
    conn = db()
    conn.execute("INSERT INTO users (email, password) VALUES (?, ?)", (data["email"], digest))
    conn.commit()
    return jsonify({"ok": True})


@app.route("/checkout", methods=["POST"])
def checkout():
    amount = float(request.json["amount"])
    rate = requests.get(RATES_URL).json()["usd"]
    conn = db()
    conn.execute("INSERT INTO payments (amount) VALUES (?)", (amount * rate,))
    conn.commit()
    return jsonify({"charged": amount * rate})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
