from cloud import cam, upload
import os
from datetime import datetime
from PIL import Image, ImageEnhance
import requests
from model_user import predict

async def take_photo():
	file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
	photo_path = os.path.join(os.environ["PHOTOS"], file_name)
	cam.capture_file(photo_path)
	bucket_name = "nav-sight_photos"
	blob_name = "image_" + file_name

	return await upload("nav-sight-photos", blob_name, photo_path)


def identify_quality(cloud_image_url):
	try:
		os.system("python /home/vedant/NavSight/model_user.py")
		response = requests.post(
			'https://api.remove.bg/v1.0/removebg',
			data={'image_url' : cloud_image_url, 'size': 'auto'},
			headers={'X-Api-Key': os.getenv("REMOVE_BG_KEY")},
		)

		if response.status_code == requests.codes.ok:
			with open('/home/vedant/NavSight/no-bg.png', 'wb') as out:
				out.write(response.content)
		else:
			print("Error:", response.status_code, response.text)

		output_image = Image.open('/home/vedant/NavSight/no-bg.png')
		output_image = output_image.convert("RGB")
		enhancer = ImageEnhance.Sharpness(output_image)
		img_enhanced = enhancer.enhance(2.0)
		img_enhanced.save("/home/vedant/NavSight/enhanced.jpg")
		output_image = Image.open("/home/vedant/NavSight/enhanced.jpg")
		res = predict(output_image)

		if 'fresh' in res.lower():
			return 'From the image, this looks fresh'
		else:
			return 'From the image, this looks rotten'

	except Exception as e:
		print(e)