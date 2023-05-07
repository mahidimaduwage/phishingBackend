import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('url_database.db')
c = conn.cursor()

# Query the data from the table
c.execute("SELECT * FROM url_data")

# Fetch and print the data
rows = c.fetchall()
for row in rows:
    print(row)

# Close the connection
conn.close()