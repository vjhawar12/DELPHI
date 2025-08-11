from picamera2 import Preview
from google.cloud import vision
import RPi.GPIO as gpio
import sys
from DELPHI.helper.tts import tts
import asyncio
from DELPHI.helper.utils import client, vision_client, tts_client, cam
from DELPHI.helper.camera import take_photo
import json

# JSON file for variables
with open("params.json", "r") as file:
	params = json.load(file)

button_pin = params["button_pin"]
switch_pin = params["switch_pin"]

# Obj recognition mode by default
mode = "Object Recognition"
gpio.setmode(gpio.BOARD)
gpio.setup(button_pin, gpio.IN)

# pull up resistor to ensure no floating states
gpio.setup(switch_pin, gpio.IN, pull_up_down=gpio.PUD_UP) 

""" Accepts an image url (from cloud) and uses the Google Vision API to detect text """
def detect_text_uri(uri):
	image = vision.Image() 
	image.source.image_uri = uri
	response = vision_client.text_detection(image=image)
	texts = response.text_annotations

	if texts:
		print(texts[0].description)
		return texts[0].description
	
""" Accepts an image url (from cloud) and uses the OpenAI API to detect objects in the image """
async def analyze_image(image_url):
	_response = client.chat.completions.create(
		model="gpt-4o-mini", # a lightweight model that performs quite well at image recognition
   		messages=[
			{
	   			"role": "user",
				"content": [
					{"type" : "text", "text" : "Ignoring the background color, use 2 sentences to explain what are the main objects in this image. Do not start with 'In this image'. Use the first sentence to mention the object and the second to give important context. Do not read the text in the photo at all."},
					{ "type" : "image_url","image_url" : { "url": image_url, } }
				],
			},
		],
		stream = True, # stream mode leads to lower latency (~3 seconds) between button press and hearing the first word   
	)

	await asyncio.gather(_response)

""" On button press run either object detection or text recognition mode depending on the switch value """
async def button_callback(mode):
	image_url = await take_photo()
	if mode == "Object Recognition":
		print("detecting image")
		analysis = await analyze_image(image_url)
		await tts(analysis)

	elif mode == "Text Recognition":
		print("recongizing text")
		await tts(detect_text_uri(image_url))
	else:
		raise ValueError(f"Invalid mode '{mode}'")

""" Main method. Called once pi is booted up. """
async def run():
	# warming up asyncio with a dummy task slightly improved performance
	warmup = asyncio.create_task(tts(" "))
	
	await tts('Object recognition mode on.')
	await tts('Camera enabled. Press the button anytime.')
	
	# configuring camera
	cam.start_preview(Preview.NULL)
	preview_config = cam.create_preview_configuration(main={"size": (800, 600)})
	cam.configure(preview_config)
	cam.start()
	
	# button press logic
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
	except KeyboardInterrupt: # for testing only
		await tts("Terminating program")
	finally: # cleaning up resources
		cam.stop()
		cam.close()
		gpio.cleanup()
		sys.exit()
		await client.close()
		await vision_client.close()
		await tts_client.close()

asyncio.run(run())


