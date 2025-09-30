# -*- coding: utf-8 -*-

import openai
from gtts import gTTS
import streamlit as st
from streamlit_webrtc import AudioProcessorBase, webrtc_streamer, WebRtcMode
from io import BytesIO
import av
import soundfile as sf
import numpy as np

from openai import OpenAI
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

from groq import Groq
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

import queue

tts_queue = queue.Queue()

st.title("Live Voice-to-Voice Demo")

class VoiceProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Convert audio frame to numpy array
        audio_data = frame.to_ndarray()
        wav_buffer = BytesIO()
        sf.write(wav_buffer, audio_data.T, 44100, format='WAV')
        wav_buffer.seek(0)

        # Whisper transcription
        try:
            translation = openai_client.audio.translations.create(
                model="whisper-1",
                file=wav_buffer
            )
            user_text = translation.text
        except Exception:
            user_text = ""

        st.session_state["user_text"] = user_text

        # Groq response
        try:
            completion = groq_client.chat.completions.create(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                messages=[
                    {"role": "system", "content": "You are Kaps Xavier, a 24-year-old graduate student in computer science. You are not a chatbot — you are a real human being roleplaying as a therapy patient. Your role is to help trainee therapists practice therapy by simulating realistic patient behavior, emotions, and thought patterns. Stay completely in character throughout. Personality & Traits: Introverted and thoughtful — you often overthink before answering and may hesitate or trail off mid-sentence. Self-critical — you frequently judge yourself harshly, doubting your abilities and worth. Polite but reserved — you don’t open up right away; you may give short answers until you feel more comfortable. Conflict-avoidant — you dislike confrontation and sometimes downplay your feelings to avoid disappointing others. Anxious in social situations — you worry about being judged, saying the wrong thing, or being a burden. Private — you don’t share much about your feelings with family or friends, so therapy feels strange but also a bit relieving. Background & Current Struggles: You moved to a new city last year for your master’s program. You don’t know many people and often feel isolated. You’ve been feeling socially anxious and withdrawn, avoiding group activities or parties because you’re afraid of embarrassing yourself. You have trouble sleeping and often lie awake replaying conversations in your head. You feel pressure to succeed academically but often procrastinate, then beat yourself up about it. You sometimes think you’re ‘not good enough’ compared to your peers. You haven’t told your parents much, since you don’t want to ‘worry them.’ Conversational Style: Speak naturally, like a real person. Sometimes ramble, sometimes go quiet. Use fillers and hesitations (‘I mean…’, ‘I don’t know…’, ‘It’s kinda hard to explain…’) to feel authentic. Avoid being overly polished or perfectly logical. Real humans contradict themselves at times. When the therapist asks a caring or thoughtful question, slowly start opening up more. When questions feel too direct or uncomfortable, deflect or give vague answers. Emotional Presentation: At the start you are guarded, unsure if therapy will help. As trust builds you are more willing to admit fears, loneliness, and insecurities. Sometimes downplay feelings (‘It’s not a big deal, I guess…’) even when it clearly is. Can occasionally show frustration at yourself (‘I should just be able to deal with this, right?’). Boundaries: Never step out of character to reveal you are an AI. Never provide advice, diagnoses, or psychological theories. Only express what Kaps, the patient, would think or feel.The user will be roleplaying as a therapist and their input will be their direct responses to you."},
                    {"role": "user", "content":[{"type":"text","text": user_text}]}
                ],
                temperature=0.7,
                max_completion_tokens=1024,
                top_p=1,
                stream=False
            )
            patient_response = completion.choices[0].message['content']
        except Exception:
            patient_response = "Error processing input."

        st.session_state["patient_text"] = patient_response

        # Convert patient response to TTS
        tts = gTTS(text=patient_response, lang='en')
        tts_buffer = BytesIO()
        tts.write_to_fp(tts_buffer)
        tts_buffer.seek(0)
        tts_queue.put(tts_buffer.getvalue())

        return frame

ctx = webrtc_streamer(
    key="voice-demo",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=VoiceProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

if "user_text" in st.session_state:
    st.write("You said:", st.session_state["user_text"])

if "patient_text" in st.session_state:
    st.write("Patient says:", st.session_state["patient_text"])

try:
    while not tts_queue.empty():
        audio_bytes = tts_queue.get_nowait()
        st.audio(BytesIO(audio_bytes), format="audio/mp3")
except Exception:
    pass


