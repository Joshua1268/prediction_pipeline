import os
from dotenv import load_dotenv
import pymysql
from sshtunnel import SSHTunnelForwarder
import logging

logger = logging.getLogger(__name__)
load_dotenv()

class DatabaseConnector:
    def __init__(self):
        self.ssh_user = os.getenv('SSH_USERNAME')
        self.ssh_pass = os.getenv('SSH_PASSWORD')
        self.ssh_host = os.getenv('SSH_HOST')
        
        self.db_host = os.getenv('DB_HOST')
        self.db_user = os.getenv('DB_USERNAME')
        self.db_pass = os.getenv('DB_PASSWORD')
        self.db_name = os.getenv('DB_NAME')
        
        self.tunnel = None
        self.conn = None

    def connect(self):
        try:
            self.tunnel = SSHTunnelForwarder(
                (self.ssh_host, 22),
                ssh_username=self.ssh_user,
                ssh_password=self.ssh_pass,
                remote_bind_address=(self.db_host, 3306),
                set_keepalive=True
            )
            self.tunnel.start()
            logger.info("Connexion SSH tunnel établie avec succès.")

            self.conn = pymysql.connect(
                host='127.0.0.1',
                user=self.db_user,
                passwd=self.db_pass,
                database=self.db_name,
                port=self.tunnel.local_bind_port
            )
            logger.info("Connexion à la base de données établie avec succès.")
            return self.conn

        except Exception as e:
            logger.error(f"Erreur lors de la connexion à la base de données: {e}")
            raise

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("Connexion à la base fermée.")
        if self.tunnel:
            self.tunnel.stop()
            logger.info("Tunnel SSH fermé.")
