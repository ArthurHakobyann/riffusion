import dataclasses
import io
import typing as T
from pathlib import Path

import numpy as np
import pydub
import streamlit as st
from PIL import Image

from riffusion.datatypes import InferenceInput, PromptInput
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util_ as streamlit_util


def render() -> None:
     

    device = streamlit_util.select_device(st.sidebar)
    extension = streamlit_util.select_audio_extension(st.sidebar)

    num_interpolation_steps = T.cast(
        int,
        st.sidebar.number_input(
            "Interpolation steps",
            value=12,
            min_value=1,
            max_value=20,
            help="Number of model generations between the two prompts. Controls the duration.",
        ),
    )

    num_inference_steps = T.cast(
        int,
        st.sidebar.number_input(
            "Steps per sample", value=50, help="Number of denoising steps per model run"
        ),
    )

    guidance = st.sidebar.number_input(
        "Guidance",
        value=7.0,
        help="How much the model listens to the text prompt",
    )

    init_image_name = st.sidebar.selectbox(
        "Seed image",
        # TODO(hayk): Read from directory
        options=["og_beat", "agile", "marim", "motorway", "vibes", "custom"],
        index=0,
        help="Which seed image to use for img2img. Custom allows uploading your own.",
    )
    assert init_image_name is not None
    if init_image_name == "custom":
        init_image_file = st.sidebar.file_uploader(
            "Upload a custom seed image",
            type=streamlit_util.IMAGE_EXTENSIONS,
            label_visibility="collapsed",
        )
        if init_image_file:
            st.sidebar.image(init_image_file)

    alpha_power = st.sidebar.number_input("Alpha Power", value=1.0)

    show_individual_outputs = st.sidebar.checkbox(
        "Show individual outputs",
        value=False,
        help="Show each model output",
    )
    show_images = st.sidebar.checkbox(
        "Show individual images",
        value=False,
        help="Show each generated image",
    )

    alphas = np.linspace(0, 1, num_interpolation_steps)

    # Apply power scaling to alphas to customize the interpolation curve
    alphas_shifted = alphas * 2 - 1
    alphas_shifted = (np.abs(alphas_shifted) ** alpha_power * np.sign(alphas_shifted) + 1) / 2
    alphas = alphas_shifted

    alphas_str = ", ".join([f"{alpha:.2f}" for alpha in alphas])
    st.write(f"**Alphas** : [{alphas_str}]")

    # Prompt inputs A and B in two columns

    with st.form(key="interpolation_form"):
        left, right = st.columns(2)

        with left:
            st.write("##### Prompt A")
            prompt_input_a = PromptInput(
                guidance=guidance, **get_prompt_inputs(key="a", denoising_default=0.75)
            )

        with right:
            st.write("##### Prompt B")
            prompt_input_b = PromptInput(
                guidance=guidance, **get_prompt_inputs(key="b", denoising_default=0.75)
            )

        st.form_submit_button("Generate", type="primary")

    if not prompt_input_a.prompt or not prompt_input_b.prompt:
        st.info("Enter both prompts to interpolate between them")
        return

    if init_image_name == "custom":
        if not init_image_file:
            st.info("Upload a custom seed image")
            return
        init_image = Image.open(init_image_file).convert("RGB")
    else:
        init_image_path = (
            Path(__file__).parent.parent.parent.parent / "seed_images" / f"{init_image_name}.png"
        )
        init_image = Image.open(str(init_image_path)).convert("RGB")

    # TODO(hayk): Move this code into a shared place and add to riffusion.cli
    image_list: T.List[Image.Image] = []
    audio_bytes_list: T.List[io.BytesIO] = []
    for i, alpha in enumerate(alphas):
        inputs = InferenceInput(
            alpha=float(alpha),
            num_inference_steps=num_inference_steps,
            seed_image_id="og_beat",
            start=prompt_input_a,
            end=prompt_input_b,
        )

        if i == 0:
            with st.expander("Example input JSON", expanded=False):
                st.json(dataclasses.asdict(inputs))

        image, audio_bytes = run_interpolation(
            inputs=inputs,
            init_image=init_image,
            device=device,
            extension=extension,
        )

        if show_individual_outputs:
            st.write(f"#### ({i + 1} / {len(alphas)}) Alpha={alpha:.2f}")
            if show_images:
                st.image(image)
            st.audio(audio_bytes)

        image_list.append(image)
        audio_bytes_list.append(audio_bytes)

    st.write("#### Final Output")

    # TODO(hayk): Concatenate with overlap and better blending like in audio to audio
    audio_segments = [pydub.AudioSegment.from_file(audio_bytes) for audio_bytes in audio_bytes_list]
    concat_segment = audio_segments[0]
    for segment in audio_segments[1:]:
        concat_segment = concat_segment.append(segment, crossfade=0)

    audio_bytes = io.BytesIO()
    concat_segment.export(audio_bytes, format=extension)
    audio_bytes.seek(0)

    st.write(f"Duration: {concat_segment.duration_seconds:.3f} seconds")
    st.audio(audio_bytes)

    output_name = (
        f"{prompt_input_a.prompt.replace(' ', '_')}_"
        f"{prompt_input_b.prompt.replace(' ', '_')}.{extension}"
    )
    st.download_button(
        output_name,
        data=audio_bytes,
        file_name=output_name,
        mime=f"audio/{extension}",
    )


def get_prompt_inputs(
    key: str,
    include_negative_prompt: bool = False,
    cols: bool = False,
    denoising_default: float = 0.5,
) -> T.Dict[str, T.Any]:
    """
    Compute prompt inputs from widgets.
    """
    p: T.Dict[str, T.Any] = {}

    # Optionally use columns
    left, right = T.cast(T.Any, st.columns(2) if cols else (st, st))

    visibility = "visible" if cols else "collapsed"
    p["prompt"] = left.text_input("Prompt", label_visibility=visibility, key=f"prompt_{key}")

    if include_negative_prompt:
        p["negative_prompt"] = right.text_input("Negative Prompt", key=f"negative_prompt_{key}")

    p["seed"] = T.cast(
        int,
        left.number_input(
            "Seed",
            value=42,
            key=f"seed_{key}",
            help="Integer used to generate a random result. Vary this to explore alternatives.",
        ),
    )

    p["denoising"] = right.number_input(
        "Denoising",
        value=denoising_default,
        key=f"denoising_{key}",
        help="How much to modify the seed image",
    )

    return p


def run_interpolation(
    inputs: InferenceInput,
    init_image: Image.Image,
    checkpoint: str = streamlit_util.DEFAULT_CHECKPOINT,
    device: str = "cuda",
    extension: str = "mp3",
) -> T.Tuple[Image.Image, io.BytesIO]:
    """
    Cached function for riffusion interpolation.
    """
    pipeline = streamlit_util.load_riffusion_checkpoint(
        device=device,
        checkpoint=checkpoint,
        # No trace so we can have variable width
        no_traced_unet=True,
    )

    image = pipeline.riffuse(
        inputs,
        init_image=init_image,
        mask_image=None,
    )

    # TODO(hayk): Change the frequency range to [20, 20k] once the model is retrained
    params = SpectrogramParams(
        min_frequency=0,
        max_frequency=10000,
    )

    # Reconstruct from image to audio
    audio_bytes = streamlit_util.audio_bytes_from_spectrogram_image(
        image=image,
        params=params,
        device=device,
        output_format=extension,
    )

    return image, audio_bytes
