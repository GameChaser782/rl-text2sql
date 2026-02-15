"""
Create placeholder SQLite databases referenced by a Spider-style JSON file.

Usage:
    python scripts/create_placeholder_dbs.py --input data/spider/train_spider.json --db_root data/spider/database

This will create minimal SQLite files for any db_id referenced in the JSON.
"""

import argparse
import json
import os
import sqlite3


def create_placeholder_db(db_root, db_id):
    os.makedirs(os.path.join(db_root, db_id), exist_ok=True)
    db_path = os.path.join(db_root, db_id, f"{db_id}.sqlite")
    if os.path.exists(db_path):
        print(f"DB already exists: {db_path}")
        return db_path
    # Create a simple table so basic SELECTs can run
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS placeholder (id INTEGER PRIMARY KEY, text TEXT)"
    )
    cur.executemany(
        "INSERT INTO placeholder (id, text) VALUES (?,?)", [(1, "alpha"), (2, "beta")]
    )
    conn.commit()
    conn.close()
    print(f"Created placeholder DB: {db_path}")
    return db_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--db_root", type=str, required=True)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    db_ids = set()
    for ex in data:
        db_id = ex.get("db_id")
        if db_id:
            db_ids.add(db_id)

    for db_id in db_ids:
        create_placeholder_db(args.db_root, db_id)


if __name__ == "__main__":
    main()
