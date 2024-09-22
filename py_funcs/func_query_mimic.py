import psycopg2
import pandas as pd
import json
import os
from .func_project_dir import *

class mimicdb:
    def __init__(self, schema: str = None):
        # Set up the path to the configuration file
        project_dir = project_path()
        self.config_path = os.path.join(project_dir, "files", "mimic-iii-db.txt")
        self._load_config()

        self.schema = schema if schema else self.config["SCHEMA"][0]
        self.connection = self.connect()  # Automatically connect to the database on initialization

    def _load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            raise Exception(f"Configuration file {self.config_path} not found.")
        except json.JSONDecodeError:
            raise Exception("Error decoding JSON from the configuration file.")

        self.server = self.config["SERVER"][0]
        self.database = self.config["DATABASE"][0]
        self.username = self.config["USERNAME"][0]
        self.password = self.config["PASSWORD"][0]

    def connect(self):
        config = {
            "host": self.server,
            "database": self.database,
            "user": self.username,
            "password": self.password,
            "options": f"-c search_path={self.schema}"
        }
        try:
            conn = psycopg2.connect(**config)
            return conn
        except psycopg2.Error as e:
            raise Exception(f"Error connecting to the database: {e}")

    def get_database_objects(self):
        query = """
            SELECT table_name, 'table' AS type 
            FROM information_schema.tables 
            WHERE table_schema = %s AND table_type = 'BASE TABLE'
            UNION
            SELECT table_name, 'view' AS type
            FROM information_schema.views
            WHERE table_schema = %s
            UNION
            SELECT matviewname AS table_name, 'materialized view' AS type 
            FROM pg_matviews
            WHERE schemaname = %s;
        """
        return self._execute_query(query, params=(self.schema, self.schema, self.schema), fetchall=True)

    def get_column_names(self, table_name):
        query = f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = %s 
            AND table_name = %s;
        """
        columns = self._execute_query(query, params=(self.schema, table_name), fetchall=True)
        return [col[0].upper() for col in columns]

    def make_query(self, sql):
        df = self._execute_query(sql, as_dataframe=True)
        df.columns = [col.upper() for col in df.columns]
        return df

    def _execute_query(self, query, params=None, fetchall=False, as_dataframe=False):
        with self.connection as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                if as_dataframe:
                    rows = cursor.fetchall()
                    colnames = [desc[0] for desc in cursor.description]
                    return pd.DataFrame(rows, columns=colnames)
                if fetchall:
                    return cursor.fetchall()
                return cursor.fetchone()
