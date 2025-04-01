from picamera2 import Picamera2, Preview
import os
from openai import AsyncOpenAI
from google.cloud import storage, texttospeech
from google.oauth2 import service_account
from google.cloud import vision
from datetime import datetime
import RPi.GPIO as gpio
from time import sleep
import sys
from model_user import predict
import requests
import asyncio
import aiofiles
import aiohttp
from gcloud.aio.storage import Storage
from removebg import RemoveBg
from PIL import Image, ImageEnhance

button_pin = 16
switch_pin = 36
mode = "Object Recognition"
gpio.setmode(gpio.BOARD)
gpio.setup(button_pin, gpio.IN)
gpio.setup(switch_pin, gpio.IN, pull_up_down=gpio.PUD_UP)

cam = Picamera2()
client = AsyncOpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))

vision_client = vision.ImageAnnotatorClient.from_service_account_file(
			os.getenv("OCR_KEY")
	)


tts_client = texttospeech.TextToSpeechClient(
		credentials = service_account.Credentials.from_service_account_file(os.getenv("TTS_KEY"))
	)


async def take_photo():
	file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
	photo_path = os.path.join(os.environ["PHOTOS"], file_name)
	cam.capture_file(photo_path)
	bucket_name = "nav-sight_photos"
	blob_name = "image_" + file_name

	return await upload("nav-sight-photos", blob_name, photo_path)

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

def detect_text_uri(uri):
	image = vision.Image()
	image.source.image_uri = uri
	response = vision_client.text_detection(image=image)
	texts = response.text_annotations

	if texts:
		print(texts[0].description)
		return texts[0].description

async def analyze_image(image_url):
	_response = client.chat.completions.create(
		model="gpt-4o-mini",
   		messages=[
			{
	   			"role": "user",
				"content": [
					{"type" : "text", "text" : "Ignoring the background color, use 2 sentences to explain what are the main objects in this image. Do not start with 'In this image'. Use the first sentence to mention the object and the second to give important context. Do not read the text in the photo at all."},
					{ "type" : "image_url","image_url" : { "url": image_url, } }
				],
			},
		],
		stream = False,
	)


	_response2 = client.chat.completions.create(
		model="gpt-4o-mini",
   		messages=[
			{
	   			"role": "user",
				"content": [
					{"type" : "text", "text" : "Based on the attatched image, answer the following question with one word, yes or no: Was there a food or drink item in the photo?"},
					{ "type" : "image_url","image_url" : { "url": image_url, } }
				]
			},
		], stream = False,
	)

	response, response2 = await asyncio.gather(_response, _response2)
	# change to return if not working
	return response.choices[0].message.content, response2.choices[0].message.content

async def tts(text):
	ssml = f"<speak> {text} </speak>"

	input_text = texttospeech.SynthesisInput(ssml=ssml)

	voice = texttospeech.VoiceSelectionParams(
		language_code = 'en-US',
		name = 'en-US-Standard-C',
		ssml_gender = texttospeech.SsmlVoiceGender.FEMALE,
	)

	audio_config = texttospeech.AudioConfig(
		audio_encoding = texttospeech.AudioEncoding.MP3
	)

	response = tts_client.synthesize_speech(
		input = input_text, voice = voice, audio_config = audio_config
	)

	player = await asyncio.create_subprocess_exec(
		"mpg321", "-",
		stdin=asyncio.subprocess.PIPE,
		stdout=asyncio.subprocess.DEVNULL,
		stderr=asyncio.subprocess.DEVNULL
	)
	try:
		player.stdin.write(response.audio_content)
		await player.stdin.drain()
	finally:
		player.stdin.close()
		await player.wait()

async def button_callback(mode):
	image_url = await take_photo()
	if mode == "Object Recognition":
		print("detecting image")
		analysis, food_or_drink = await analyze_image(image_url)
		await tts(analysis)
		print(food_or_drink)
		if 'yes' in food_or_drink.lower():
			await tts("Since I detected a consumable in this image, I'll check if its fresh or not.")
			await tts(identify_quality(image_url))
	elif mode == "Text Recognition":
		print("recongizing text")
		await tts(detect_text_uri(image_url))
	else:
		raise ValueError(f"Invalid mode '{mode}'")

async def load_ml():
	asyncio.create_task(tts('Loading ML model. Do not press the button right now.'))
	process = await asyncio.create_subprocess_exec("python", "/home/vedant/NavSight/model_user.py")
	await process.wait()


async def run():
	warmup = asyncio.create_task(tts(" "))

	await load_ml()
	await tts('Object recognition mode on.')
	await tts('Camera enabled. Press the button anytime.')

	cam.start_preview(Preview.NULL)
	preview_config = cam.create_preview_configuration(main={"size": (800, 600)})
	cam.configure(preview_config)
	cam.start()

	try:
		while True:
			if gpio.input(switch_pin) == 1:
				mode = "Object Recognition"
			if gpio.input(switch_pin) == 0:
				mode = "Text Recognition"
			if gpio.input(button_pin) == 1:
				print("button pressed")
				await tts(f'Press detected. Active mode {mode} I\'m taking a look')
				await button_callback(mode)
			await asyncio.sleep(0.2)
	except KeyboardInterrupt:
		await tts("Terminating program")
	finally:
		cam.stop()
		cam.close()
		gpio.cleanup()
		sys.exit()
		await client.close()
		await vision_client.close()
		await tts_client.close()

asyncio.run(run())


