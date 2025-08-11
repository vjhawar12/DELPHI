from google.cloud import texttospeech
import asyncio
from DELPHI.helper.utils import tts_client

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