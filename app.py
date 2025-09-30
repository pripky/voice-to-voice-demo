# -*- coding: utf-8 -*-

import openai
from gtts import gTTS
import streamlit as st
from io import BytesIO
import av
import soundfile as sf
import numpy as np
from st_audiorec import st_audiorec
import tempfile

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
            # Save WAV to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_wav:
            tmp_wav.write(wav_audio_data)
            tmp_wav.flush()

            # Convert to MP3
            audio = gTTS(text="dummy", lang="en") 

            # Send to Whisper
            with open(tmp_wav.name, "rb") as f:
                translation = openai_client.audio.translations.create(
                    model="whisper-1",
                    file=f
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
        
        patient_response = completion.choices[0].message.content
        st.write("Patient says:", patient_response)


        #Text to speech
        tts = gTTS(text=patient_response, lang='en')
        tts_buffer = BytesIO()
        tts.write_to_fp(tts_buffer)
        tts_buffer.seek(0)
        st.audio(tts_buffer, format='audio/mp3')
