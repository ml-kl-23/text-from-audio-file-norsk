# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 10:13:43 2022

!pip3 install pyaudio

(If pyaudio is not installing please use the installation direct from 
wheel based on your python version)

!pip3 install SpeechRecognition
!pip3 install gradio

@author: manish
"""
from transformers import pipeline

import gradio as gr

asr = pipeline("automatic-speech-recognition", "NbAiLab/nb-wav2vec2-1b-bokmaal")
classifier = pipeline("text-classification")

def speech_to_text(speech):
    text = asr(speech)["text"]
    return text

def text_to_sentiment(text):
    return classifier(text)[0]["label"]

demo = gr.Blocks()

with demo:
    audio_file = gr.Audio(type="filepath")
    text = gr.Textbox()
    label = gr.Label()

    b1 = gr.Button("Recognize Speech")
    b2 = gr.Button("Classify Sentiment")

    b1.click(speech_to_text, inputs=audio_file, outputs=text)
    b2.click(text_to_sentiment, inputs=text, outputs=label)

demo.launch()