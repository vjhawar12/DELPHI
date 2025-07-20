import aiofiles
import aiohttp
from gcloud.aio.storage import Storage
import os

async def upload(bucket_name, blob_name, path_to_file):
	async with aiohttp.ClientSession() as session:
		storage_client = Storage(service_file=os.getenv("GCS_KEY"), session=session)
		async with aiofiles.open(path_to_file, mode="rb") as f:
			output = await f.read()
			status = await storage_client.upload(
				bucket_name,
				blob_name,
				output,
			)
		print(f"https://storage.googleapis.com/{bucket_name}/{blob_name}")
	return f"https://storage.googleapis.com/{bucket_name}/{blob_name}"