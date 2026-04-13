import psycopg2
import json
import logging as log

class PostgresDB:
    def __init__(self, connection_string=None, host="localhost", port=5432, database="rag", user="postgres", password=""):
        self.connection_string = connection_string
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.conn = None
        self.cursor = None
        
    def connect(self):
        try:
            if self.connection_string:
                self.conn = psycopg2.connect(self.connection_string)
            else:
                self.conn = psycopg2.connect(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password
                )
            self.cursor = self.conn.cursor()
            log.info(f"Connected to PostgreSQL at {self.host}:{self.port}/{self.database}")
            return True
        except Exception as e:
            log.error(f"Failed to connect to PostgreSQL: {e}")
            return False

    def disconnect(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            log.info("Disconnected from PostgreSQL")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type:
                self.conn.rollback()
            else:
                self.conn.commit()
        self.disconnect()

    def create_chunks_table(self, table_name="chunks"):
        try:
            query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                chunk_id VARCHAR(255) UNIQUE NOT NULL,
                parent_id VARCHAR(255),
                source VARCHAR(1024),
                chunk_number INTEGER,
                text TEXT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_{table_name}_source ON {table_name}(source);
            CREATE INDEX IF NOT EXISTS idx_{table_name}_parent_id ON {table_name}(parent_id);
            """
            self.cursor.execute(query)
            log.info(f"Created table: {table_name}")
            return True
        except Exception as e:
            log.error(f"Error creating table: {e}")
            return False

    def insert_chunks(self, table_name, chunks_data):
        try:
            for chunk in chunks_data:
                query = f"""
                INSERT INTO {table_name} (chunk_id, parent_id, source, chunk_number, text, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE
                SET text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata
                """
                self.cursor.execute(query, (
                    chunk.get("chunk_id"),
                    chunk.get("parent_id"),
                    chunk.get("source"),
                    chunk.get("chunk_number"),
                    chunk.get("text"),
                    json.dumps(chunk.get("metadata", {}))
                ))
            log.info(f"Inserted {len(chunks_data)} chunks")
            return True
        except Exception as e:
            log.error(f"Error inserting chunks: {e}")
            return False

    def get_chunks_by_parent_id(self, table_name, parent_id):
        try:
            query = f"SELECT chunk_id, chunk_number, text FROM {table_name} WHERE parent_id = %s ORDER BY chunk_number"
            self.cursor.execute(query, (parent_id,))
            return self.cursor.fetchall()
        except Exception as e:
            log.error(f"Error getting chunks: {e}")
            return None

    def get_chunk_by_id(self, table_name, chunk_id):
        try:
            query = f"SELECT chunk_id, parent_id, source, chunk_number, text, metadata FROM {table_name} WHERE chunk_id = %s"
            self.cursor.execute(query, (chunk_id,))
            result = self.cursor.fetchone()
            if result:
                return {
                    "chunk_id": result[0],
                    "parent_id": result[1],
                    "source": result[2],
                    "chunk_number": result[3],
                    "text": result[4],
                    "metadata": result[5]
                }
            return None
        except Exception as e:
            log.error(f"Error getting chunk: {e}")
            return None