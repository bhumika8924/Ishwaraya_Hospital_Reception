import mysql.connector

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="@bhumi1234",
    database="hospital_monitor"
)

# Create cursor
cursor = conn.cursor()


# Function to insert data
def insert_log(entries, exits, receptionists):

    query = """
    INSERT INTO reception_logs
    (timestamp, entries_count, exits_count, receptionists_count)
    VALUES (NOW(), %s, %s, %s)
    """

    values = (entries, exits, receptionists)

    cursor.execute(query, values)

    conn.commit()

    print(
        f"Saved -> Entries: {entries}, Exits: {exits}, Receptionists: {receptionists}"
    )