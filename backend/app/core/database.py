import logging
from motor.motor_asyncio import AsyncIOMotorClient
from neo4j import GraphDatabase
from app.core.config import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    client_mongo: AsyncIOMotorClient = None
    client_neo4j = None
    db_mongo = None

    @classmethod
    async def connect_dbs(cls):
        # 1. Connect to MongoDB (Document Storage for Evidence/Cases)
        logger.info("Connecting to MongoDB...")
        cls.client_mongo = AsyncIOMotorClient(settings.MONGO_URI)
        cls.db_mongo = cls.client_mongo[settings.MONGO_DB_NAME]
        logger.info("MongoDB connected successfully.")

        # 2. Connect to Neo4j (Knowledge Graph for Contradictions)
        logger.info("Connecting to Neo4j...")
        cls.client_neo4j = GraphDatabase.driver(
            settings.NEO4J_URI, 
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        logger.info("Neo4j connected successfully.")

    @classmethod
    async def close_dbs(cls):
        if cls.client_mongo:
            cls.client_mongo.close()
            logger.info("MongoDB connection closed.")
        if cls.client_neo4j:
            cls.client_neo4j.close()
            logger.info("Neo4j connection closed.")

db_manager = DatabaseManager()