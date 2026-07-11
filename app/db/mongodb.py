from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
from datetime import datetime, timezone

class MongoDBClient:
    client: AsyncIOMotorClient = None
    db = None

    @classmethod
    def connect(cls):
        if cls.client is None:
            cls.client = AsyncIOMotorClient(settings.mongodb_url)
            cls.db = cls.client[settings.mongodb_database]

    @classmethod
    def close(cls):
        if cls.client is not None:
            cls.client.close()

    @classmethod
    async def get_collection(cls, name: str):
        if cls.db is None:
            cls.connect()
        return cls.db[name]

    @classmethod
    async def create_job(cls, job_data: dict) -> str:
        coll = await cls.get_collection("ocr_jobs")
        job_data["created_at"] = datetime.now(timezone.utc)
        job_data["updated_at"] = datetime.now(timezone.utc)
        result = await coll.insert_one(job_data)
        return str(result.inserted_id)
        
    @classmethod
    async def get_job(cls, job_id: str) -> dict:
        coll = await cls.get_collection("ocr_jobs")
        return await coll.find_one({"job_id": job_id})
        
    @classmethod
    async def update_job_status(cls, job_id: str, status: str, extra_fields: dict = None):
        coll = await cls.get_collection("ocr_jobs")
        update_data = {"status": status, "updated_at": datetime.now(timezone.utc)}
        if extra_fields:
            update_data.update(extra_fields)
        await coll.update_one({"job_id": job_id}, {"$set": update_data})
