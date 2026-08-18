from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "CrimeVision AI Backend"
    API_V1_STR: str = "/api/v1"
    
    # MongoDB Configuration
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "crimevision"
    
    # Neo4j Configuration
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "crimevision_password"
    
    # JWT Security
    JWT_SECRET_KEY: str = "your_super_secret_jwt_key_here"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 120

    class Config:
        env_file = ".env"

settings = Settings()