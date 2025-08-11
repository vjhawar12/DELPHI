from google.cloud import texttospeech
import asyncio
from DELPHI.helper.utils import tts_client

""" Text-to-speech function using Google's texttospeech API"""
async def tts(text):
	ssml = f"<speak> {text} </speak>" # these tokens indicate the beginning and end of spoken sentence

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
		stdin=asyncio.subprocess.PIPE, # pipping to STDIN avoids having to store audio on disk before playing it
		stdout=asyncio.subprocess.DEVNULL,
		stderr=asyncio.subprocess.DEVNULL
	)
	try:
		player.stdin.write(response.audio_content)
		await player.stdin.drain()
	finally:
		player.stdin.close()
		await player.wait()