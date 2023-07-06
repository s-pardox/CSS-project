import mysql.connector
from mastodon.settings import sql_credentials

class Database:
    """
    Singleton class that manages a MySQL connection.

    This class uses the singleton pattern to ensure that only one MySQL connection exists at a time.
    The connection is established when the class is first instantiated and can be accessed via the getInstance method.
    """

    __instance = None  # Stores the singleton instance of this class
    cnx = None  # Stores the MySQL connection

    @staticmethod
    def getInstance():
        """
        Static method to get the current instance of this class.

        If an instance does not yet exist, it creates one.
        Returns the instance of this class.
        """
        if Database.__instance == None:
            Database()
        return Database.__instance

    def __init__(self):
        """
        Constructor for the Database class.

        This is made 'virtually private' by raising an exception if an attempt is made to instantiate an additional instance.
        If an instance does not exist, it creates the instance and a MySQL connection.
        """
        if Database.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Database.__instance = self
            self.cnx = mysql.connector.connect(
                user=sql_credentials.USER,
                password=sql_credentials.PASSWORD,
                host=sql_credentials.HOST,
                database=sql_credentials.DATABASE
            )

    def get_connection(self):
        """
        Method to get the current MySQL connection.

        If a connection does not exist or has been closed, it creates a new one.
        Returns the MySQL connection.
        """
        if not self.cnx.is_connected():
            self.cnx = mysql.connector.connect(
                user=sql_credentials.USER,
                password=sql_credentials.PASSWORD,
                host=sql_credentials.HOST,
                database=sql_credentials.DATABASE
            )
        return self.cnx

    def close_connection(self):
        """
        Method to close the current MySQL connection.

        If a connection exists and is open, it closes the connection.
        """
        if self.cnx.is_connected():
            self.cnx.close()
