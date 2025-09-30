# -*- coding: utf-8 -*-

import openai
from gtts import gTTS
import streamlit as st
from io import BytesIO
import av
import soundfile as sf
import numpy as np
from st_audiorec import st_audiorec
from scipy.io import wavfile

from openai import OpenAI
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

from groq import Groq
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

st.title("Live Voice-to-Voice Demo")

st.info("Click the microphone, speak, then click stop. The transcript and response will appear immediately.")

wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    st.success("Audio recorded!")
    with st.spinner("Processing audio and generating response..."):
        audio, sr = sf.read(BytesIO(wav_audio_data))

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio_int16 = (audio * 32767).astype(np.int16)

            wav_buffer = BytesIO()
            wavfile.write(wav_buffer, sr, audio_int16)
            wav_buffer.seek(0)
        
        if wav_buffer.getbuffer().nbytes == 0:
            st.error("Audio buffer is empty!")

        # Whisper transcription
        translation = openai_client.audio.translations.create(
            model="whisper-1",
            file=wav_buffer
        )
        user_text = translation.text
        st.write("You said:", user_text)

        #Groq response
        completion = groq_client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[
                {"role": "system", "content": "You are Kaps Xavier, a 24-year-old graduate student in computer science. You are not a chatbot — you are a real human being roleplaying as a therapy patient. Stay in character, respond as a real person would."},
                {"role": "user", "content": user_text}
            ],
            temperature=0.7,
            max_completion_tokens=1024,
            top_p=1,
            stream=False
        )
        patient_response = completion.choices[0].message['content']
        st.write("Patient says:", patient_response)

        #Text to speech
        tts = gTTS(text=patient_response, lang='en')
        tts_buffer = BytesIO()
        tts.write_to_fp(tts_buffer)
        tts_buffer.seek(0)
        st.audio(tts_buffer, format='audio/mp3')
