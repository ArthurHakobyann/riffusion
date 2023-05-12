
import io
import typing as T
from pathlib import Path

import numpy as np
import pydub
import streamlit as st
from PIL import Image
def scale_image_to_32_stride(image: Image.Image) -> Image.Image:
    """
    Scale an image to a size that is a multiple of 32.
    """
    closest_width = int(np.ceil(image.width / 32) * 32)
    closest_height = int(np.ceil(image.height / 32) * 32)
    return image.resize((closest_width, closest_height), Image.BICUBIC)
def slice_audio_into_clips(
    segment: pydub.AudioSegment, clip_start_times: T.Sequence[float], clip_duration_s: float
) -> T.List[pydub.AudioSegment]:
    """
    Slice an audio segment into a list of clips of a given duration at the given start times.
    """
    clip_segments: T.List[pydub.AudioSegment] = []
    for i, clip_start_time_s in enumerate(clip_start_times):
        clip_start_time_ms = int(clip_start_time_s * 1000)
        clip_duration_ms = int(clip_duration_s * 1000)
        clip_segment = segment[clip_start_time_ms : clip_start_time_ms + clip_duration_ms]

        # TODO(hayk): I don't think this is working properly
        if i == len(clip_start_times) - 1:
            silence_ms = clip_duration_ms - int(clip_segment.duration_seconds * 1000)
            if silence_ms > 0:
                clip_segment = clip_segment.append(pydub.AudioSegment.silent(duration=silence_ms))

        clip_segments.append(clip_segment)

    return clip_segments
def get_clip_params(advanced: bool = False) -> T.Dict[str, T.Any]:
    """
    Render the parameters of slicing audio into clips.
    """
    p: T.Dict[str, T.Any] = {}

    cols = st.columns(4)

    p["start_time_s"] = cols[0].number_input(
        "Start Time [s]",
        min_value=0.0,
        value=0.0,
    )
    p["duration_s"] = cols[1].number_input(
        "Duration [s]",
        min_value=0.0,
        value=20.0,
    )

    if advanced:
        p["clip_duration_s"] = cols[2].number_input(
            "Clip Duration [s]",
            min_value=3.0,
            max_value=10.0,
            value=5.0,
        )
    else:
        p["clip_duration_s"] = 5.0

    if advanced:
        p["overlap_duration_s"] = cols[3].number_input(
            "Overlap Duration [s]",
            min_value=0.0,
            max_value=10.0,
            value=0.2,
        )
    else:
        p["overlap_duration_s"] = 0.2

    return p

