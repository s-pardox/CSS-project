from mastodon.utils.Database import Database

def create_main_tables():
    """
    This function creates the 'Accounts', 'Toots', and 'AccountStats' tables in your database.
    It assumes that the database connection is handled by the Database singleton class.
    """
    cnx = Database.getInstance().get_connection()
    cursor = cnx.cursor()

    # Create 'Accounts' table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Accounts (
            account_id BIGINT PRIMARY KEY,
            username VARCHAR(255),
            instance VARCHAR(255),
            created_at TIMESTAMP
        ) ENGINE=InnoDB;
    """)
    cnx.commit()

    # Create 'Toots' table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Toots (
            toot_id BIGINT PRIMARY KEY,
            created_at TIMESTAMP,
            account_id BIGINT,
            FOREIGN KEY(account_id) REFERENCES Accounts(account_id)
        ) ENGINE=InnoDB;
    """)
    cnx.commit()

    # Create 'AccountStats' table
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS AccountStats (
                stats_id INT AUTO_INCREMENT PRIMARY KEY,
                account_id BIGINT,
                followers_count BIGINT,
                following_count BIGINT,
                statuses_count BIGINT,
                last_status_at DATE,
                FOREIGN KEY(account_id) REFERENCES Accounts(account_id)
            ) ENGINE=InnoDB;
        """)
    cnx.commit()

    print("Tables created successfully!")
    cursor.close()
    Database.getInstance().close_connection()

def create_instance_info_table():
    """
    This function creates the 'InstanceInfo' table in your database.
    It assumes that the database connection is handled by the Database singleton class.
    """
    cnx = Database.getInstance().get_connection()
    cursor = cnx.cursor()

    # Create 'InstanceInfo' table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS InstanceInfo (
            instances_id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255),
            users BIGINT,
            statuses BIGINT,
            connections BIGINT,
            created_at TIMESTAMP
        ) ENGINE=InnoDB;
    """)
    cnx.commit()

    print("InstanceInfo table created successfully!")
    cursor.close()
    Database.getInstance().close_connection()

def create_top_10_instance_table():
    """
    This function creates the 'DailyTop10' table in your database.
    It assumes that the database connection is handled by the Database singleton class.
    """
    cnx = Database.getInstance().get_connection()
    cursor = cnx.cursor()

    # Create 'DailyTop10' table
    cursor.execute("""
      CREATE TABLE DailyTop10 (
        date DATE NOT NULL,
        name VARCHAR(255) NOT NULL,
        users BIGINT(20) DEFAULT NULL,
        PRIMARY KEY (date, name)
      )
    """)
    cnx.commit()

    print("DailyTop10 table created successfully!")
    cursor.close()
    Database.getInstance().close_connection()

def create_insert_daily_top_10_procedure():
    """
    This function generates and executes the 'InsertDailyTop10' stored procedure.
    It assumes that the database connection is handled by the Database singleton class.

    The 'InsertDailyTop10' stored procedure retrieves the top 10 instances from the 'InstanceInfo' table
    for each day within the date range '2022-11-20' to '2023-05-15', based on the 'users' field.
    It then inserts the selected instances into the 'DailyTop10' table, which includes the columns:
    - date: The date for which the top 10 instances are calculated.
    - name: The name of the instance.
    - users: The number of users for the instance.

    The stored procedure deletes any existing records in the 'DailyTop10' table for the specific date
    and inserts the new top 10 instances.

    Example usage:
    create_insert_daily_top_10_procedure()

    After executing the stored procedure, the 'DailyTop10' table will contain the top 10 instances
    for each day within the specified date range.
    """
    cnx = Database.getInstance().get_connection()
    cursor = cnx.cursor()

    # Generate 'InsertDailyTop10' stored procedure
    create_procedure_query = """
        CREATE PROCEDURE InsertDailyTop10()
        BEGIN
          DECLARE currentDate DATE;

          SET currentDate = '2022-11-20';

          WHILE currentDate <= '2023-05-15' DO
            DELETE FROM DailyTop10 WHERE date = currentDate;

            INSERT INTO DailyTop10 (date, name, users)
            SELECT currentDate, name, users
            FROM InstanceInfo
            WHERE date(created_at) = currentDate
            ORDER BY users DESC
            LIMIT 10;

            SET currentDate = currentDate + INTERVAL 1 DAY;
          END WHILE;

        END;
    """
    cursor.execute(create_procedure_query)
    cnx.commit()

    print("InsertDailyTop10 stored procedure created successfully!")
    cursor.close()
    Database.getInstance().close_connection()

def call_insert_daily_top_10_procedure():
    """
    This function calls the 'InsertDailyTop10' stored procedure.
    It assumes that the database connection is handled by the Database singleton class.
    """
    cnx = Database.getInstance().get_connection()
    cursor = cnx.cursor()

    # Call 'InsertDailyTop10' stored procedure
    call_procedure_query = "CALL InsertDailyTop10"
    cursor.execute(call_procedure_query)
    cnx.commit()

    print("InsertDailyTop10 stored procedure called successfully!")
    cursor.close()

def delete_insert_daily_top_10_procedure():
    """
    This function deletes the 'InsertDailyTop10' stored procedure.
    It assumes that the database connection is handled by the Database singleton class.
    """
    cnx = Database.getInstance().get_connection()
    cursor = cnx.cursor()

    # Delete 'InsertDailyTop10' stored procedure
    delete_procedure_query = "DROP PROCEDURE IF EXISTS InsertDailyTop10"
    cursor.execute(delete_procedure_query)
    cnx.commit()

    print("InsertDailyTop10 stored procedure deleted successfully!")
    cursor.close()
    Database.getInstance().close_connection()

def drop_main_tables():
    """
    This function deletes the 'Accounts', 'Toots', and 'AccountStats' tables from your database.
    It assumes that the database connection is handled by the Database singleton class.
    """

    cnx = Database.getInstance().get_connection()
    cursor = cnx.cursor()

    # Delete 'AccountStats' table
    cursor.execute("""
        DROP TABLE IF EXISTS AccountStats;
    """)
    cnx.commit()

    # Delete 'Toots' table
    cursor.execute("""
        DROP TABLE IF EXISTS Toots;
    """)
    cnx.commit()

    # Delete 'Accounts' table
    cursor.execute("""
        DROP TABLE IF EXISTS Accounts;
    """)
    cnx.commit()

    print("Tables deleted successfully!")
    cursor.close()
    Database.getInstance().close_connection()
