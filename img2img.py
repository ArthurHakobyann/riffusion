from riffusion.streamlit import util_ as streamlit_util
import io
import typing as T
from pathlib import Path
import os
from scipy.io import wavfile
import numpy as np
import pydub
import random
import streamlit as st
from PIL import Image
from func import *
from riffusion.datatypes import InferenceInput, PromptInput
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit.tasks.interpolation import run_interpolation
from riffusion.util import audio_util

import sounddevice as sd
from playsound import playsound
from scipy.io.wavfile import write
from pynput import keyboard
import queue
import soundfile as sf

#audio_file = open("../ESC-50-master/audio/1-21896-A-35.wav","rb").read()
def real_time():
    segment = streamlit_util.load_audio_file("./save.wav")
    if segment.frame_rate != 44100:
            segment = segment.set_frame_rate(44100)
    device = "cuda"
    extension = "wav"
    checkpoint = "riffusion/riffusion-model-v1"
    start_time_s = 0.0
    clip_duration_s = 3
    overlap_duration_s = 0.1
    duration_s = min(100, segment.duration_seconds - start_time_s)
    increment_s = clip_duration_s - overlap_duration_s
    clip_start_times = start_time_s + np.arange(0, duration_s - clip_duration_s, increment_s)
    params = SpectrogramParams(
        min_frequency=0,
        max_frequency=10000,
        stereo=False,)
    interpolate = False
    use_magic_mix = False
    prompt = "rock and roll electro guitar solo"
    denoising = 0.60
    negative_prompt="acoustic , classic "
    seed = random.randint(1,10000)
    num_inference_steps=70
    guidance=7
    scheduler = "DPMSolverMultistepScheduler"


    magic_mix_kmin=0.3
    magic_mix_kmax=0.5
    magic_mix_mix_factor=0.5

    clip_segments = slice_audio_into_clips(
            segment=segment,
            clip_start_times=clip_start_times,
            clip_duration_s=clip_duration_s,)
    #print(len(clip_segments),clip_segments[0].shape)
    result_images: T.List[Image.Image] = []
    result_segments: T.List[pydub.AudioSegment] = []
    for i, clip_segment in enumerate(clip_segments):
        audio_bytes = io.BytesIO()
        clip_segment.export(audio_bytes, format="wav")
        init_image = streamlit_util.spectrogram_image_from_audio(
            clip_segment,
            params=params,
            device=device,)
        print(init_image)
        init_image_resized = scale_image_to_32_stride(init_image)
        progress_callback = None
        if interpolate:
            alphas = list(np.linspace(0, 1, len(clip_segments)))
            inputs = InferenceInput(
                    alpha=float(alphas[i]),
                    num_inference_steps=num_inference_steps,
                    seed_image_id="og_beat",
                    start="guitar",
                    end="jazz",)
            image, audio_bytes = run_interpolation(
                    inputs=inputs,
                    init_image=init_image_resized,
                    device=device,
                    checkpoint=checkpoint,)

        elif use_magic_mix:
            image = streamlit_util.run_img2img_magic_mix(
                    prompt=prompt,
                    init_image=init_image_resized,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance,
                    seed=seed,
                    kmin=magic_mix_kmin,
                    kmax=magic_mix_kmax,
                    mix_factor=magic_mix_mix_factor,
                    device=device,
                    scheduler=scheduler,
                    checkpoint=checkpoint,
                )
        else:
            image = streamlit_util.run_img2img(
                    prompt=prompt,
                    init_image=init_image_resized,
                    denoising_strength=denoising,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    progress_callback=progress_callback,
                    device=device,
                    scheduler=scheduler,
                    checkpoint=checkpoint,)
            # Resize back to original size
        image = image.resize(init_image.size, Image.BICUBIC)
        result_images.append(image)
        riffed_segment = streamlit_util.audio_segment_from_spectrogram_image(
                image=image,
                params=params,
                device=device,)
        result_segments.append(riffed_segment)
        audio_bytes = io.BytesIO()
        riffed_segment.export(audio_bytes, format="wav")
        '''
        diff_image = Image.fromarray(255 - diff_np.astype(np.uint8))
        diff_segment = streamlit_util.audio_segment_from_spectrogram_image(
                    image=diff_image,
                    params=params,
                    device=device,)
        audio_bytes = io.BytesIO()
        diff_segment.export(audio_bytes, format=extension)
        '''
    combined_segment = audio_util.stitch_segments(result_segments, crossfade_s=overlap_duration_s)
    print(combined_segment)
    audio_bytes = io.BytesIO()
    combined_segment.export("play.wav", format="wav")
    #print(audio_bytes)
    #wavfile.write("lll.wav",44100,audio_bytes)
    print(f"#### Final Audio ({combined_segment.duration_seconds}s)")
    
    
    

    


q = queue.Queue()
s_r = False
def callback(indata, frames, time, status):
    global s_r
    if s_r:
        q.put(indata.copy())

def on_press(key):
    return key != keyboard.Key.space

def on_release(key):
    global s_r
    if key == keyboard.Key.space:
        s_r = not s_r
        listener.stop()

listener = keyboard.Listener(on_release=on_release)
listener.start()
print('press Space to start the recording')

filename = "save.wav"
with sf.SoundFile(filename, samplerate=44100, channels=2, mode='w') as file:
    with sd.InputStream(samplerate=44100, channels=2, callback=callback):
        listener.join()
        while not q.empty():
            file.write(q.get())
            

print("processing...")
real_time()
playsound("play.wav")
