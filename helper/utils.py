from openai import AsyncOpenAI
from google.cloud import vision, texttospeech
import os
from google.oauth2 import service_account
from picamera2 import Picamera2

""" This file just initializes resources used elsewhere """

cam = Picamera2() 

client = AsyncOpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))

vision_client = vision.ImageAnnotatorClient.from_service_account_file(
			os.getenv("OCR_KEY")
	)


tts_client = texttospeech.TextToSpeechClient(
		credentials = service_account.Credentials.from_service_account_file(os.getenv("TTS_KEY"))
	)