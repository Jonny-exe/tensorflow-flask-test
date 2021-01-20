# db.py
import sys
import mariadb

# Connect to MariaDB Platform
try:
    conn = mariadb.connect(
        user="code_share",
        password="password",
        host="192.0.2.1",
        port=3306,
        database="code_share",
    )
except mariadb.Error as e:
    print(f"Error connecting to MariaDB Platform: {e}")
    sys.exit(1)

# Get Cursor
cur = conn.cursor()