from cloud import upload
from DELPHI.helper.utils import cam
import os
from datetime import datetime

""" Async function that takes photo with picam and temporarily stores it locally before uploading it to GCP """
async def take_photo():
	file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
	photo_path = os.path.join(os.environ["PHOTOS"], file_name)
	cam.capture_file(photo_path) # overwrites the old photo which was uploaded to cloud 
	bucket_name = "nav-sight_photos"
	blob_name = "image_" + file_name

	return await upload(bucket_name, blob_name, photo_path)
