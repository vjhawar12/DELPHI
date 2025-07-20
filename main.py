from picamera2 import Preview
from google.cloud import vision
import RPi.GPIO as gpio
import sys
from tts import tts
import asyncio
from init import client, vision_client, tts_client
from cloud import upload
from camera import take_photo, identify_quality, cam
import json

with open("params.json", "r") as file:
	params = json.load(file)

button_pin = params["button_pin"]
switch_pin = params["switch_pin"]

mode = "Object Recognition"
gpio.setmode(gpio.BOARD)
gpio.setup(button_pin, gpio.IN)
gpio.setup(switch_pin, gpio.IN, pull_up_down=gpio.PUD_UP)

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


