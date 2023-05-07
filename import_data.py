import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('url_database.db')
c = conn.cursor()

# Create the table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS url_data (id INTEGER PRIMARY KEY AUTOINCREMENT, url_type TEXT, url TEXT)''')

# Read the dataset from the txt file
# with open('test.txt', 'r') as f:
#     for line in f:
#         url_type, url = line.strip().split('\t', maxsplit=1)
#
#         # Insert the data into the table
#         c.execute("INSERT INTO url_data (url_type, url) VALUES (?, ?)", (url_type, url))

url_type="legitimate"
url="chrome://extensions/"
c.execute("INSERT INTO url_data (url_type, url) VALUES (?, ?)", (url_type, url))

# Commit the changes and close the connection
conn.commit()
conn.close()

#inserted train.txt,val.txt,test.txt