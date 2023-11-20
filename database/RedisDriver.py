import redis.asyncio as redis

class RedisDriver:
    def __init__(self, url):

        self.redis_url = f'redis://{url}'
        self.redis_client = redis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)

    def getRedisClient(self):
        return self.redis_client

    async def setKey(self, key, val, ttl=60):
        await self.redis_client.set(key, val)
        if ttl:
            await self.redis_client.expire(key, ttl)
        return True

    async def getKey(self, key):
        return await self.redis_client.get(key)

    async def deleteKey(self, key):
        return await self.redis_client.delete(key)

    async def deleteKeyWithPrefix(self, prefix):
        keys = await self.redis_client.keys(prefix + "*")
        for key in keys:
            await self.redis_client.delete(key)
        return True

    async def getContentsWithCodeAndIndex(self, prefix):
        keys = await self.redis_client.keys(prefix + "*")
        # keys 리스트에서 첫 번째 요소를 가져옴
        if keys:
            content = keys[0].split("_")[2]
            return content
        else:
            return None  # 키가 없는 경우
    async def flushAll(self):
        return await self.redis_client.flushall()