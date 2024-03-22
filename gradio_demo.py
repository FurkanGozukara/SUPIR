import argparse
import datetime
import os
import shutil
import tempfile
import threading
import time
import traceback
from datetime import datetime
from typing import Tuple, List, Any, Dict
import gc
import einops
import gradio as gr
import numpy as np
import requests
import torch
from PIL import Image
from PIL import PngImagePlugin
from gradio_imageslider import ImageSlider
from SUPIR.utils.rename_meta import rename_meta_key
from omegaconf import OmegaConf
import ui_helpers
from SUPIR.models.SUPIR_model import SUPIRModel
from SUPIR.util import HWC3, upscale_image, fix_resize, convert_dtype
from SUPIR.util import create_model, load_supir_weights
from SUPIR.utils.compare import create_comparison_video
from SUPIR.utils.face_restoration_helper import FaceRestoreHelper
from SUPIR.utils.model_fetch import get_model
from SUPIR.utils.status_container import StatusContainer, MediaData
from SUPIR.utils import shared, devices
from llava.llava_agent import LLavaAgent
import CKPT_PTH
from ui_helpers import is_video, extract_video, compile_video, is_image, get_video_params, printt

SUPIR_REVISION = "v42"

parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default='127.0.0.1', help="IP address for the server to listen on.")
parser.add_argument("--share", type=str, default=False, help="Set to True to share the app publicly.")
parser.add_argument("--port", type=int, help="Port number for the server to listen on.")
parser.add_argument("--log_history", action='store_true', default=False, help="Enable logging of request history.")
parser.add_argument("--lowvram", action='store_true', default=False, help="Enable loading model parameters in FP16 precision to reduce memory usage.")
parser.add_argument("--lowvram_fp8", action='store_true', default=False, help="Enable loading model parameters in FP8 precision to reduce memory usage.")
parser.add_argument("--force_vae_upcasting", action='store_true', default=False, help="Forces VAE conversion from FP16 to FP32 to prevent black background image error.")
parser.add_argument("--use_tile_vae", action='store_true', default=True,
                    help="Enable tiling for the VAE to handle larger images with limited memory.")
parser.add_argument("--use_fast_tile", action='store_true', default=False,
                    help="Use a faster tile encoding/decoding, may impact quality.")
parser.add_argument("--outputs_folder_button", type=str, default=False, help="Outputs Folder Button Will Be Enabled")
parser.add_argument("--encoder_tile_size", type=int, default=512,
                    help="Tile size for the encoder. Larger sizes may improve quality but require more memory.")
parser.add_argument("--decoder_tile_size", type=int, default=64,
                    help="Tile size for the decoder. Larger sizes may improve quality but require more memory.")
parser.add_argument("--load_8bit_llava", action='store_true', default=False,
                    help="Load the LLAMA model in 8-bit precision to save memory.")
parser.add_argument("--load_4bit_llava", action='store_true', default=True,
                    help="Load the LLAMA model in 4-bit precision to significantly reduce memory usage.")
parser.add_argument("--ckpt_browser", action='store_true', default=True, help="Enable a checkpoint selection dropdown.")
parser.add_argument("--ckpt_dir", type=str, default='models',
                    help="Directory where models and checkpoints are stored.")
parser.add_argument("--ckpt_dir2", type=str, default=None, help="Alternative directory for big models will be stored.")
parser.add_argument("--theme", type=str, default='default',
                    help="Theme for the UI. Use 'default' or specify a custom theme.")
parser.add_argument("--open_browser", action='store_true', default=True,
                    help="Automatically open the web browser when the server starts.")
parser.add_argument("--outputs_folder", type=str, default='outputs', help="Folder where output files will be saved.")
parser.add_argument("--debug", action='store_true', default=False,
                    help="Enable debug mode, disables open_browser, and adds ui buttons for testing elements.")
parser.add_argument("--dont_move_cpu", action='store_true', default=False,
                    help="Disables moving models to the CPU after completed. If you have sufficient VRAM enable this.")

args = parser.parse_args()
ui_helpers.ui_args = args
current_video_fps = 0
total_video_frames = 0
video_start = 0
video_end = 0
last_input_path = None
last_video_params = None
meta_upload = False

total_vram = 100000
auto_unload = False
if torch.cuda.is_available():
    # Get total GPU memory
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print("Total VRAM: ", total_vram, "GB")
    # If total VRAM <= 12GB, set auto_unload to True
    auto_unload = total_vram <= 12

    if total_vram <= 24:
        args.lowvram = True
        args.use_tile_vae = True
        print("Loading low vram params")

shared.opts.half_mode = args.lowvram

if args.lowvram_fp8:
    shared.opts.half_mode = args.lowvram_fp8
    shared.opts.fp8_storage = args.lowvram_fp8

server_ip = args.ip
model_path = None
model_path2 = None

if args.debug:
    args.open_browser = False

if args.ckpt_dir is not None:
    args.ckpt_dir = os.path.join(os.path.dirname(__file__), args.ckpt_dir, 'checkpoints')
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir, exist_ok=True)

model_path = os.path.split(args.ckpt_dir)[0]
model_path2 = model_path if args.ckpt_dir2 is None else args.ckpt_dir2
CKPT_PTH.setModelsPath(model_path, model_path2)

#get default from repo if not exists yet
default_sdxl_model = get_model(CKPT_PTH.DEFAULT_SDXL_PATH)

if torch.cuda.device_count() >= 2:
    SUPIR_device = 'cuda:0'
    LLaVA_device = 'cuda:1'
elif torch.cuda.device_count() == 1:
    SUPIR_device = 'cuda:0'
    LLaVA_device = 'cuda:0'
else:
    SUPIR_device = 'cpu'
    LLaVA_device = 'cpu'

face_helper = None
model: SUPIRModel = None
config: OmegaConf = None
llava_agent = None
models_loaded = False
unique_counter = 0
status_container = StatusContainer()

# Store this globally so we can update variables more easily
elements_dict = {}

single_process = False
is_processing = False
last_used_checkpoint = None

slider_html = """
<div id="keyframeSlider" class="keyframe-slider">
  <div id="frameSlider"></div>

  <!-- Labels for start and end times -->
  <div class="labels">
    <span id="startTimeLabel">0:00:00</span>
    <span id="nowTimeLabel">0:00:30</span>
    <span id="endTimeLabel">0:01:00</span>
  </div>
</div>
"""


def refresh_models_click():
    new_model_list = list_models()
    return gr.update(choices=new_model_list)


def refresh_styles_click():
    new_style_list = list_styles()
    style_list = list(new_style_list.keys())
    return gr.update(choices=style_list)


def update_start_time(src_file, upscale_size, start_time):
    global video_start
    video_start = start_time
    target_res_text = update_target_resolution(src_file, upscale_size)
    return gr.update(value=target_res_text, visible=target_res_text != "")


def update_end_time(src_file, upscale_size, end_time):
    global video_end
    video_end = end_time
    target_res_text = update_target_resolution(src_file, upscale_size)
    return gr.update(value=target_res_text, visible=target_res_text != "")


def select_style(style_name, current_prompt=None, values=False):
    style_list = list_styles()

    if style_name in style_list.keys():
        style_pos, style_neg, style_llava = style_list[style_name]
        if values:
            return style_pos, style_neg, style_llava
        return gr.update(value=style_pos), gr.update(value=style_neg), gr.update(value=style_llava)
    if values:
        return "", "", ""
    return gr.update(value=""), gr.update(value=""), gr.update(value="")


import platform


def open_folder():
    open_folder_path = os.path.abspath(args.outputs_folder)
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')


def set_info_attributes(elements_to_set: Dict[str, Any]):
    output = {}
    for key, value in elements_to_set.items():
        if not getattr(value, 'elem_id', None):
            setattr(value, 'elem_id', key)
        classes = getattr(value, 'elem_classes', None)
        if isinstance(classes, list):
            if "info-btn" not in classes:
                classes.append("info-button")
                setattr(value, 'elem_classes', classes)
        output[key] = value
    return output


def list_models():
    model_dir = args.ckpt_dir
    output = []
    output.append(CKPT_PTH.DEFAULT_SDXL_PATH.split('/')[-1])

    if os.path.exists(model_dir):
        output = [f for f in os.listdir(model_dir) if ((f.endswith('.safetensors') or f.endswith('.ckpt')) and 'SUPIR-v0F.ckpt' not in f and 'SUPIR-v0Q.ckpt' not in f)]
    else:
        local_model_dir = os.path.join(os.path.dirname(__file__), args.ckpt_dir)
        if os.path.exists(local_model_dir):
            output = [f for f in os.listdir(local_model_dir) if ((f.endswith('.safetensors') or f.endswith('.ckpt')) and 'SUPIR-v0F.ckpt' not in f and 'SUPIR-v0Q.ckpt' not in f)]
   
    # Sort the models
    output = [os.path.basename(f) for f in output]
    # Ensure the values are unique
    output = list(set(output))
    output.sort()
    return output


def get_ckpt_path(ckpt_path):
    if os.path.exists(ckpt_path):
        return ckpt_path
    else:
        if os.path.exists(args.ckpt_dir):
            return os.path.join(args.ckpt_dir, ckpt_path)
        local_model_dir = os.path.join(os.path.dirname(__file__), args.ckpt_dir)
        if os.path.exists(local_model_dir):
            return os.path.join(local_model_dir, ckpt_path)
    return None


def list_styles():
    styles_path = os.path.join(os.path.dirname(__file__), 'styles')
    output = {}
    style_files = []
    llava_prompt = default_llava_prompt
    for root, dirs, files in os.walk(styles_path):
        for file in files:
            if file.endswith('.csv'):
                style_files.append(os.path.join(root, file))
    for style_file in style_files:
        with open(style_file, 'r') as f:
            lines = f.readlines()
            # Parse lines, skipping the first line
            for line in lines[1:]:
                line = line.strip()
                if len(line) > 0:
                    name = line.split(',')[0]
                    cap_line = line.replace(name + ',', '')
                    captions = cap_line.split('","')
                    if len(captions) >= 2:
                        positive_prompt = captions[0].replace('"', '')
                        negative_prompt = captions[1].replace('"', '')
                        if "{prompt}" in positive_prompt:
                            positive_prompt = positive_prompt.replace("{prompt}", "")

                        if "{prompt}" in negative_prompt:
                            negative_prompt = negative_prompt.replace("{prompt}", "")

                        if len(captions) == 3:
                            llava_prompt = captions[2].replace('"', "")

                        output[name] = (positive_prompt, negative_prompt, llava_prompt)

    return output


def selected_model():
    models = list_models()
    target_model = CKPT_PTH.DEFAULT_SDXL_PATH.split('/')[-1]
    if target_model in models:
        return target_model
    else:
        if len(models) > 0:
            return models[0]
    return None


def load_face_helper():
    global face_helper
    if face_helper is None:
        face_helper = FaceRestoreHelper(
            device='cpu',
            upscale_factor=1,
            face_size=1024,
            use_parse=True,
            det_model='retinaface_resnet50'
        )


def load_model(selected_model, diff_dtype, selected_checkpoint, sampler='DPMPP2M', device='cpu', progress=gr.Progress()):
    global model, config, last_used_checkpoint
    checkpoint_use = default_sdxl_model
    if selected_checkpoint:
        checkpoint_use = os.path.join(args.ckpt_dir, selected_checkpoint)
        if last_used_checkpoint is None:
            last_used_checkpoint = checkpoint_use

    if last_used_checkpoint != checkpoint_use:
        model = None
        devices.torch_gc()
        last_used_checkpoint = checkpoint_use
    reload_supir = (selected_model != model.current_model) if hasattr(model, 'current_model') and not model is None else False
    if model is None or reload_supir:  
        if progress is not None:
            progress(1 / 2, desc="Loading SUPIR...")

        if reload_supir == False:
            model, config = create_model(f"options/{'SUPIR_v0_tiled.yaml' if args.use_tile_vae else 'SUPIR_v0.yaml'}", device, sampler)   

        model = load_supir_weights(model, config, diff_dtype, supir_sign=selected_model[-1], reload_supir=reload_supir,
                                   ckpt_dir=args.ckpt_dir, ckpt=checkpoint_use, vae_file=None)
        if args.use_tile_vae:
            model.init_tile_vae(encoder_tile_size=512, decoder_tile_size=64, use_fast=args.use_fast_tile)
       
        #model.first_stage_model.denoise_encoder_s1 = copy.deepcopy(model.first_stage_model.denoise_encoder)
        model.current_model = selected_model

    if progress is not None:
        progress(2 / 2, desc="SUPIR loaded.")

def load_llava():
    global llava_agent
    if llava_agent is None:
        llava_path = get_model(CKPT_PTH.LLAVA_MODEL_PATH)
        llava_agent = LLavaAgent(llava_path, device=LLaVA_device, load_8bit=args.load_8bit_llava, load_4bit=args.load_4bit_llava)

def unload_llava():
    global llava_agent
    if args.load_4bit_llava or args.load_8bit_llava:
        printt("Clearing LLaVA.")
        clear_llava()
        printt("LLaVA cleared.")
    else:
        printt("Unloading LLaVA.")
        llava_agent = llava_agent.to('cpu')
        devices.torch_gc()
        printt("LLaVA unloaded.")

def clear_llava():
    global llava_agent
    del llava_agent
    llava_agent = None
    devices.torch_gc()


def all_to_cpu_background():
    if args.dont_move_cpu:
        return
    global face_helper, model, llava_agent
    printt("Moving all to CPU")
    if face_helper is not None:
        face_helper = face_helper.to('cpu')
        printt("Face helper moved to CPU")
    if model is not None:
        model = model.to('cpu')
        model.move_to('cpu')
        printt("Model moved to CPU")
    if llava_agent is not None:
        unload_llava()
    devices.torch_gc()
    printt("All moved to CPU")


def all_to_cpu():
    if args.dont_move_cpu:
        return
    cpu_thread = threading.Thread(target=all_to_cpu_background)
    cpu_thread.start()


def to_gpu(elem_to_load, device):
    if elem_to_load is not None:
        elem_to_load = elem_to_load.to(device)
        if getattr(elem_to_load, 'move_to', None):
            elem_to_load.move_to(device)
        torch.cuda.set_device(device)
    return elem_to_load


def update_model_settings(model_type, param_setting):
    """
    Returns a series of gr.updates with settings based on the model type.
    If 'model_type' contains 'lightning', it uses the settings for a 'lightning' SDXL model.
    Otherwise, it uses the settings for a normal SDXL model.
    s_cfg_Quality, spt_linear_CFG_Quality, s_cfg_Fidelity, spt_linear_CFG_Fidelity, edm_steps
    """
    # Default settings for a "lightning" SDXL model
    lightning_settings = {
        's_cfg_Quality': 2.0,
        'spt_linear_CFG_Quality': 2.0,
        's_cfg_Fidelity': 1.5,
        'spt_linear_CFG_Fidelity': 1.5,
        'edm_steps': 10
    }

    # Default settings for a normal SDXL model
    normal_settings = {
        's_cfg_Quality': 7.5,
        'spt_linear_CFG_Quality': 4.0,
        's_cfg_Fidelity': 4.0,
        'spt_linear_CFG_Fidelity': 1.0,
        'edm_steps': 50
    }

    # Choose the settings based on the model type
    settings = lightning_settings if 'Lightning' in model_type else normal_settings

    if param_setting == "Quality":
        s_cfg = settings['s_cfg_Quality']
        spt_linear_CFG = settings['spt_linear_CFG_Quality']
    else:
        s_cfg = settings['s_cfg_Fidelity']
        spt_linear_CFG = settings['spt_linear_CFG_Fidelity']

    return gr.update(value=s_cfg), gr.update(value=spt_linear_CFG), gr.update(value=settings['edm_steps'])


def update_inputs(input_file, upscale_amount):
    global current_video_fps, total_video_frames, video_start, video_end
    file_input = gr.update(visible=True)
    image_input = gr.update(visible=False, sources=[])
    video_slider = gr.update(visible=False)
    video_start_time = gr.update(value=0)
    video_end_time = gr.update(value=0)
    video_current_time = gr.update(value=0)
    video_fps = gr.update(value=0)
    video_total_frames = gr.update(value=0)
    current_video_fps = 0
    total_video_frames = 0
    video_start = 0
    video_end = 0
    res_output = gr.update(value="")
    if is_image(input_file):
        image_input = gr.update(visible=True, value=input_file, sources=[], label="Input Image")
        file_input = gr.update(visible=False)
        target_res = update_target_resolution(input_file, upscale_amount)
        res_output = gr.update(value=target_res, visible=target_res != "")
    elif is_video(input_file):
        video_attributes = ui_helpers.get_video_params(input_file)
        video_start = 0
        end_time = video_attributes['frames']
        video_end = end_time
        mid_time = int(end_time / 2)
        current_video_fps = video_attributes['framerate']
        total_video_frames = end_time
        video_end_time = gr.update(value=end_time)
        video_total_frames = gr.update(value=end_time)
        video_current_time = gr.update(value=mid_time)
        video_frame = ui_helpers.get_video_frame(input_file, mid_time)
        video_slider = gr.update(visible=True)
        image_input = gr.update(visible=True, value=video_frame, sources=[], label="Input Video")
        file_input = gr.update(visible=False)
        video_fps = gr.update(value=current_video_fps)
        target_res = update_target_resolution(input_file, upscale_amount)
        res_output = gr.update(value=target_res, visible=target_res != "")
    elif input_file is None:
        file_input = gr.update(visible=True, value=None)
    return file_input, image_input, video_slider, res_output, video_start_time, video_end_time, video_current_time, video_fps, video_total_frames


def update_target_resolution(img, do_upscale):
    global last_input_path, last_video_params
    if img is None:
        last_video_params = None
        last_input_path = None
        return ""
    if is_image(img):
        last_input_path = img
        last_video_params = None
        with Image.open(img) as img:
            width, height = img.size
            width_org, height_org = img.size
    elif is_video(img):
        if img == last_input_path:
            params = last_video_params
        else:
            last_input_path = img
            params = get_video_params(img)
            last_video_params = params
        width, height = params['width'], params['height']
        width_org, height_org = params['width'], params['height']
    else:
        last_input_path = None
        last_video_params = None
        return ""

    width *= do_upscale
    height *= do_upscale

    if min(width, height) < 1024:
        do_upscale_factor = 1024 / min(width, height)
        width *= do_upscale_factor
        height *= do_upscale_factor

    output_lines = [
        f"<td style='padding: 8px; border-bottom: 1px solid #ddd;'>Input: {int(width_org)}x{int(height_org)} px, {width_org * height_org / 1e6:.2f} Megapixels</td>",
        f"<td style='padding: 8px; border-bottom: 1px solid #ddd;'>Estimated Output Resolution: {int(width)}x{int(height)} px, {width * height / 1e6:.2f} Megapixels</td>",
    ]

    if total_video_frames > 0 and is_video(img):
        selected_video_frames = video_end - video_start
        total_video_time = int(selected_video_frames / current_video_fps)
        output_lines += [
            f"<td style='padding: 8px; border-bottom: 1px solid #ddd;'>Selected video frames: {selected_video_frames}</td>",
            f"<td style='padding: 8px; border-bottom: 1px solid #ddd;'>Total video time: {total_video_time} seconds</td>"
        ]

    output = '<table style="width:100%;"><tr>'
    # Fix here: Convert each pair of <td> elements into a single string before joining
    output += ''.join(''.join(output_lines[i:i + 2]) for i in range(0, len(output_lines), 2))
    output += '</tr></table>'
    return output


def read_image_metadata(image_path):
    if image_path is None:
        return
    # Check if the file exists
    if not os.path.exists(image_path):
        return "File does not exist."

    # Get the last modified date and format it
    last_modified_timestamp = os.path.getmtime(image_path)
    last_modified_date = datetime.fromtimestamp(last_modified_timestamp).strftime('%d %B %Y, %H:%M %p - UTC')

    # Open the image and extract metadata
    with Image.open(image_path) as img:
        width, height = img.size
        megapixels = (width * height) / 1e6

        metadata_str = f"Last Modified Date: {last_modified_date}\nMegapixels: {megapixels:.2f}\n"

        # Extract metadata based on image format
        if img.format == 'JPEG':
            exif_data = img._getexif()
            if exif_data:
                for tag, value in exif_data.items():
                    tag_name = Image.ExifTags.TAGS.get(tag, tag)
                    metadata_str += f"{tag_name}: {value}\n"
        else:
            metadata = img.info
            if metadata:
                for key, value in metadata.items():
                    metadata_str += f"{key}: {value}\n"
            else:
                metadata_str += "No additional metadata found."

    return metadata_str


def update_elements(status_label):
    print(f"Label changed: {status_label}")
    prompt_el = gr.update()
    result_gallery_el = gr.update(height=400)
    result_slider_el = gr.update(height=400)
    result_video_el = gr.update(height=400)
    comparison_video_el = gr.update(height=400, visible=False)
    seed = None
    face_gallery_items = []
    evt_id = ""
    if not is_processing:
        output_data = status_container.image_data
        if len(output_data) == 1:
            image_data = output_data[0]
            caption = image_data.caption
            print(f"Caption: {caption}")
            prompt_el = gr.update(value=caption)
            if len(image_data.outputs) > 0:
                outputs = image_data.outputs
                params = image_data.metadata_list
                if len(params) != len(outputs):
                    print("Mismatch in outputs and metadata list, using defaults.")
                    params = [status_container.process_params] * len(outputs)
                first_output = outputs[0]
                first_params = params[0]
                seed = first_params.get('seed', "")
                face_gallery_items = first_params.get('face_gallery', [])
                evt_id = first_params.get('event_id', "")
                if image_data.media_type == "image":
                    if image_data.comparison_video:
                        comparison_video_el = gr.update(value=image_data.comparison_video, visible=True)
                    result_slider_el = gr.update(value=[image_data.media_path, first_output], visible=True)
                    result_gallery_el = gr.update(value=None, visible=False)
                    result_video_el = gr.update(value=None, visible=False)
                elif image_data.media_type == "video":
                    prompt_el = gr.update(value="")
                    result_video_el = gr.update(value=first_output, visible=True)
                    result_gallery_el = gr.update(value=None, visible=False)
                    result_slider_el = gr.update(value=None, visible=False)
        elif len(output_data) > 1:
            first_output_data = output_data[0]
            if len(first_output_data.outputs):
                first_params = first_output_data.metadata_list[
                    0] if first_output_data.metadata_list else status_container.process_params
                seed = first_params.get('seed', "")
                face_gallery_items = first_params.get('face_gallery', [])
                evt_id = first_params.get('event_id', "")
            all_outputs = []
            for output_data in output_data:
                all_outputs.extend(output_data.outputs)
            result_gallery_el = gr.update(value=all_outputs, visible=True)
            result_slider_el = gr.update(value=None, visible=False)
            result_video_el = gr.update(value=None, visible=False)
    seed_el = gr.update(value=seed)
    event_id_el = gr.update(value=evt_id)
    face_gallery_el = gr.update(value=face_gallery_items)

    return prompt_el, result_gallery_el, result_slider_el, result_video_el, comparison_video_el, event_id_el, seed_el, face_gallery_el


def populate_slider_single():
    # Fetch the image at http://www.marketingtool.online/en/face-generator/img/faces/avatar-1151ce9f4b2043de0d2e3b7826127998.jpg
    # and use it as the input image
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_path.write(requests.get(
        "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1151ce9f4b2043de0d2e3b7826127998.jpg").content)
    temp_path.close()
    lowres_path = temp_path.name.replace('.jpg', '_lowres.jpg')
    with Image.open(temp_path.name) as img:
        current_dims = (img.size[0] // 2, img.size[1] // 2)
        resized_dims = (img.size[0] // 4, img.size[1] // 4)
        img = img.resize(current_dims)
        img.save(temp_path.name)
        img = img.resize(resized_dims)
        img.save(lowres_path)
    return (gr.update(value=[lowres_path, temp_path.name], visible=True,
                      elem_classes=["active", "preview_slider", "preview_box"]),
            gr.update(visible=False, value=None, elem_classes=["preview_box"]))


def populate_gallery():
    # Fetch the image at http://www.marketingtool.online/en/face-generator/img/faces/avatar-1151ce9f4b2043de0d2e3b7826127998.jpg
    # and use it as the input image
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_path.write(requests.get(
        "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1151ce9f4b2043de0d2e3b7826127998.jpg").content)
    temp_path.close()
    lowres_path = temp_path.name.replace('.jpg', '_lowres.jpg')
    with Image.open(temp_path.name) as img:
        current_dims = (img.size[0] // 2, img.size[1] // 2)
        resized_dims = (img.size[0] // 4, img.size[1] // 4)
        img = img.resize(current_dims)
        img.save(temp_path.name)
        img = img.resize(resized_dims)
        img.save(lowres_path)
    return gr.update(value=[lowres_path, temp_path.name], visible=True,
                     elem_classes=["preview_box", "active"]), gr.update(visible=False, value=None,
                                                                        elem_classes=["preview_slider", "preview_box"])


def start_single_process(*element_values):
    global status_container, is_processing
    status_container = StatusContainer()
    status_container.is_batch = False
    values_dict = zip(elements_dict.keys(), element_values)
    values_dict = dict(values_dict)
    main_prompt = values_dict['main_prompt']
    # Delete input_image, prompt, batch_process_folder, outputs_folder from values_dict

    input_image = values_dict['src_file']
    if input_image is None:
        return "No input image provided."

    image_files = [input_image]

    # Make a dictionary to store the image data and path
    img_data = []

    if is_video(input_image):
        # Store the original video path for later
        status_container.source_video_path = input_image
        status_container.is_video = True
        extracted_folder = os.path.join(args.outputs_folder, "extracted_frames")
        if os.path.exists(extracted_folder):
            shutil.rmtree(extracted_folder)
        os.makedirs(extracted_folder, exist_ok=True)
        start = values_dict.get('video_start', None)
        end = values_dict.get('video_end', None)
        extract_success, video_params = extract_video(input_image, extracted_folder, video_start=start, video_end=end)
        if extract_success:
            status_container.video_params = video_params
        for file in os.listdir(extracted_folder):
            full_path = os.path.join(extracted_folder, file)
            media_data = MediaData(media_path=full_path)
            media_data.caption = main_prompt
            img_data.append(media_data)
    else:
        for file in image_files:
            try:
                media_data = MediaData(media_path=file)
                img = Image.open(file)
                media_data.media_data = np.array(img)
                media_data.caption = main_prompt
                img_data.append(media_data)
            except:
                pass
    result = "An exception occurred. Please try again."
    # auto_unload_llava, batch_process_folder, main_prompt, output_video_format, output_video_quality, outputs_folder,video_duration, video_fps, video_height, video_width
    keys_to_pop = ['batch_process_folder', 'main_prompt', 'output_video_format',
                   'output_video_quality', 'outputs_folder', 'video_duration', 'video_end', 'video_fps',
                   'video_height', 'video_start', 'video_width', 'src_file']

    values_dict['outputs_folder'] = args.outputs_folder
    status_container.process_params = values_dict
    values_dict = {k: v for k, v in values_dict.items() if k not in keys_to_pop}

    try:
        _, result = batch_process(img_data, **values_dict)
    except Exception as e:
        print(f"An exception occurred: {e} at {traceback.format_exc()}")
        is_processing = False
    return result


def start_batch_process(*element_values):
    global status_container, is_processing
    status_container = StatusContainer()
    status_container.is_batch = True
    values_dict = zip(elements_dict.keys(), element_values)
    values_dict = dict(values_dict)
    batch_process_folder = values_dict['batch_process_folder']
    outputs_folder = values_dict['outputs_folder']
    main_prompt = values_dict['main_prompt']
    if not batch_process_folder:
        return "No input folder provided."
    if not os.path.exists(batch_process_folder):
        return "The input folder does not exist."

    if len(outputs_folder) < 2:
        outputs_folder = args.outputs_folder

    image_files = [file for file in os.listdir(batch_process_folder) if
                   is_image(os.path.join(batch_process_folder, file))]

    # Make a dictionary to store the image data and path
    img_data = []
    for file in image_files:
        media_data = MediaData(media_path=os.path.join(batch_process_folder, file))
        img = Image.open(os.path.join(batch_process_folder, file))
        media_data.media_data = np.array(img)
        media_data.caption = main_prompt
        img_data.append(media_data)

    # Store it globally
    status_container.image_data = img_data
    result = "An exception occurred. Please try again."
    try:
        keys_to_pop = ['batch_process_folder', 'main_prompt', 'output_video_format',
                       'output_video_quality', 'outputs_folder', 'video_duration', 'video_end', 'video_fps',
                       'video_height', 'video_start', 'video_width', 'src_file']

        status_container.outputs_folder = outputs_folder
        values_dict['outputs_folder'] = outputs_folder
        status_container.process_params = values_dict

        values_dict = {k: v for k, v in values_dict.items() if k not in keys_to_pop}
        result, _ = batch_process(img_data, **values_dict)
    except Exception as e:
        print(f"An exception occurred: {e} at {traceback.format_exc()}")
        is_processing = False
    return result

def llava_process(inputs: List[MediaData], temp, p, question=None, save_captions=False, progress=gr.Progress()):
    global llava_agent, status_container
    outputs = []
    total_steps = len(inputs) + 1
    step = 0
    progress(step / total_steps, desc="Loading LLaVA...")
    load_llava()
    step += 1
    printt("Moving LLaVA to GPU.")
    llava_agent = to_gpu(llava_agent, LLaVA_device)
    printt("LLaVA moved to GPU.")
    progress(step / total_steps, desc="LLaVA loaded, captioning images...")
    for md in inputs:
        img = md.media_data
        img_path = md.media_path
        progress(step / total_steps, desc=f"Processing image {step}/{len(inputs)} with LLaVA...")
        if img is None:  ## this is for llava and video
            img = Image.open(img_path)
            img = np.array(img)
        lq = HWC3(img)
        lq = Image.fromarray(lq.astype('uint8'))
        caption = llava_agent.gen_image_caption([lq], temperature=temp, top_p=p, qs=question)
        caption = caption[0]
        md.caption = caption
        outputs.append(md)
        if save_captions:
            cap_path = os.path.splitext(img_path)[0] + ".txt"
            with open(cap_path, 'w') as cf:
                cf.write(caption)
        if not is_processing:  # Check if batch processing has been stopped
            break
        step += 1

    progress(step / total_steps, desc="LLaVA processing completed.")
    status_container.image_data = outputs
    return f"LLaVA Processing Completed: {len(inputs)} images processed at {time.ctime()}."

# video_start_time_number, video_current_time_number, video_end_time_number,
#                      video_fps_number, video_total_frames_number, src_input_file, upscale_slider
def update_video_slider(start_time, current_time, end_time, fps, total_frames, src_file, upscale_size):
    print(f"Updating video slider: {start_time}, {current_time}, {end_time}, {fps}, {src_file}")
    global video_start, video_end
    video_start = start_time
    video_end = end_time
    video_frame = ui_helpers.get_video_frame(src_file, current_time)
    target_res_text = update_target_resolution(src_file, upscale_size)
    return gr.update(value=video_frame), gr.update(value=target_res_text, visible=target_res_text != "")


def supir_process(inputs: List[MediaData], a_prompt, n_prompt, num_samples,
                  upscale, edm_steps,
                  s_stage1, s_stage2, s_cfg, seed, sampler, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype,
                  linear_cfg, linear_s_stage2, spt_linear_cfg, spt_linear_s_stage2, model_select,
                  ckpt_select, num_images, random_seed, apply_llava, face_resolution, apply_bg, apply_face,
                  face_prompt, dont_update_progress=False, unload=True,
                  progress=gr.Progress()):
    global model, status_container, event_id
    main_begin_time = time.time()
    total_images = len(inputs) * num_images
    total_progress = total_images + 1
    if unload:
        total_progress += 1
    counter = 0
    progress(counter / total_progress, desc="Loading SUPIR Model...")
    load_model(model_select, diff_dtype, ckpt_select, sampler, progress=progress)
    to_gpu(model, SUPIR_device)

    counter += 1
    progress(counter / total_progress, desc="Model Loaded, Processing Images...")
    model.ae_dtype = convert_dtype('fp32' if args.force_vae_upcasting else ae_dtype)
    model.model.dtype = convert_dtype(diff_dtype)

    idx = 0
    output_data = []
    processed_images = 0
    params = status_container.process_params

    for image_data in inputs:
        gen_params_list = []
        img_params = params.copy()
        img = image_data.media_data
        image_path = image_data.media_path
        progress(counter / total_progress, desc=f"Processing image {counter}/{total_images}...")
        if img is None:
            printt(f"Image {counter}/{total_images} is None, loading from disk.")
            with Image.open(image_path) as img:
                img = np.array(img)

        printt(f"Processing image {counter}/{total_images}...")

        # Prompt is stored directly in the image data
        img_prompt = image_data.caption
        idx = idx + 1

        # See if there is a caption file
        if not apply_llava:
            cap_path = os.path.join(os.path.splitext(image_path)[0] + ".txt")
            if os.path.exists(cap_path):
                printt(f"Loading caption from {cap_path}...")
                with open(cap_path, 'r') as cf:
                    img_prompt = cf.read()

        img = HWC3(img)
        printt("Upscaling image (pre)...")
        img = upscale_image(img, upscale, unit_resolution=32, min_size=1024)

        lq = np.array(img)
        lq = lq / 255 * 2 - 1
        lq = torch.tensor(lq, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(SUPIR_device)[:, :3, :, :]

        _faces = []
        if not dont_update_progress and progress is not None:
            progress(counter / total_images, desc=f"Upscaling Images {counter}/{total_images}")
        face_captions = [img_prompt]

        if apply_face:
            lq = np.array(img)
            load_face_helper()
            if face_helper is None or not isinstance(face_helper, FaceRestoreHelper):
                raise ValueError('Face helper not loaded')
            face_helper.clean_all()
            face_helper.read_image(lq)
            # get face landmarks for each face
            face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
            face_helper.align_warp_face()

            lq = lq / 255 * 2 - 1
            lq = torch.tensor(lq, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(SUPIR_device)[:, :3, :, :]

            if len(face_prompt) > 1:
                face_captions = [face_prompt]
            to_gpu(face_helper, SUPIR_device)

        results = []

        for _ in range(num_images):
            gen_params = img_params.copy()
            gen_params['evt_id'] = event_id
            result = None
            if random_seed or num_images > 1:
                seed = np.random.randint(0, 2147483647)
            gen_params['seed'] = seed
            start_time = time.time()  # Track the start time

            def process_sample(model, input_data, caption, face_resolution=None, is_face=False):
                samples = model.batchify_sample(input_data, caption, num_steps=edm_steps, restoration_scale=s_stage1,
                                                s_churn=s_churn, s_noise=s_noise, cfg_scale=s_cfg,
                                                control_scale=s_stage2, seed=seed,
                                                num_samples=num_samples, p_p=a_prompt, n_p=n_prompt,
                                                color_fix_type=color_fix_type,
                                                use_linear_cfg=linear_cfg, use_linear_control_scale=linear_s_stage2,
                                                cfg_scale_start=spt_linear_cfg, control_scale_start=spt_linear_s_stage2, sampler_cls=sampler)

                if is_face and face_resolution < 1024:
                    samples = samples[:, :, 512 - face_resolution // 2:512 + face_resolution // 2,
                              512 - face_resolution // 2:512 + face_resolution // 2]
                return samples

            if apply_face:
                faces = []
                restored_faces = []
                for face in face_helper.cropped_faces:
                    restored_faces.append(face)
                    face = np.array(face) / 255 * 2 - 1
                    face = torch.tensor(face, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(SUPIR_device)[:, :3,
                           :, :]
                    faces.append(face)

                for index, face in enumerate(faces):
                    progress(index / len(faces), desc=f"Upscaling Face {index}/{len(faces)}")
                    caption = face_captions[0]  # currently we dont have multiple captions for faces
                    gen_params['face_prompt'] = caption
                    samples = process_sample(model, face, [caption], face_resolution, is_face=True)
                    samples = torch.nn.functional.interpolate(samples, size=face_helper.face_size, mode='bilinear',
                                                              align_corners=False)
                    x_samples = (einops.rearrange(samples,
                                                  'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().round().clip(0,
                                                                                                                    255).astype(
                        np.uint8)
                    face_helper.add_restored_face(x_samples[0])
                    restored_faces.append(x_samples[0])
                gen_params["face_gallery"] = restored_faces

            # Look ma, we can do either now!
            if apply_bg:
                caption = [img_prompt]
                samples = process_sample(model, lq, caption)
                _bg = (einops.rearrange(samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().round().clip(0,
                                                                                                                   255).astype(
                    np.uint8)
                if apply_face:
                    face_helper.get_inverse_affine(None)
                    result = face_helper.paste_faces_to_input_image(upsample_img=_bg[0])
                else:
                    result = _bg[0]

            # I couldnt make this part
            # if not apply_bg and apply_face:
            #    face_helper.get_inverse_affine(None)
            #    result = face_helper.paste_faces_to_input_image(upsample_img=lq)

            if not apply_face and not apply_bg:
                caption = [img_prompt]
                print("Batchifying sample...")
                samples = process_sample(model, lq, caption)
                x_samples = (
                        einops.rearrange(samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().round().clip(
                    0, 255).astype(np.uint8)
                result = x_samples[0]

            gen_params['caption'] = img_prompt
            image_generation_time = time.time() - start_time
            if result is not None:
                results.append(result)
                gen_params_list.append(gen_params)
            desc = f"Image {counter}/{total_images} upscale completed in {image_generation_time:.2f} seconds"
            counter += 1
            progress(counter / total_images, desc=desc)

        # Update outputs
        image_data.outputs = results
        image_data.metadata_list = gen_params_list
        # save_image will return the image data with the outputs as paths
        image_data = save_image(image_data, status_container.is_video)

        # Append the image data to the output data after saving
        output_data.append(image_data)

        progress(counter / total_images, desc=f"Image {counter}/{total_images} processed.")
        processed_images = counter
        # Now break
        if not is_processing:
            break
    # Now we update the status container
    status_container.image_data = output_data
    if not is_processing or unload:
        progress(counter / total_images, desc="Unloading SUPIR...")
        all_to_cpu()
        progress(counter / total_images, desc="SUPIR Unloaded.")
    main_end_time = time.time()
    global unique_counter
    unique_counter = unique_counter + 1
    return f"Image Upscaling Completed: processed {total_images} images at in {main_end_time - main_begin_time:.2f} seconds #{unique_counter}"


def batch_process(img_data,
                  a_prompt, ae_dtype, apply_bg, apply_face, apply_llava, apply_supir, ckpt_select, color_fix_type,
                  diff_dtype, edm_steps, face_prompt, face_resolution, linear_CFG, linear_s_stage2,
                  make_comparison_video, model_select, n_prompt, num_images, num_samples, qs, random_seed,
                  s_cfg, s_churn, s_noise, s_stage1, s_stage2, sampler, save_captions, seed, spt_linear_CFG,
                  spt_linear_s_stage2, temperature, top_p, upscale, auto_unload_llava, progress=gr.Progress()
                  ):
    global is_processing, llava_agent, model, status_container
    ckpt_select = get_ckpt_path(ckpt_select)
    tiled = "TiledRestore" if args.use_tile_vae else "Restore"
    sampler_cls = f"sgm.modules.diffusionmodules.sampling.{tiled}{sampler}Sampler"
    if not ckpt_select:
        msg = "No checkpoint selected. Please select a checkpoint to continue."
        return msg
    start_time = time.time()
    last_result = "Select something to do."
    if not apply_llava and not apply_supir:
        msg = "No processing selected. Please select LLaVA, SUPIR, or both to continue."
        printt(msg)
        return msg, msg
    if is_processing:
        msg = "Batch processing already in progress."
        printt(msg)
        return msg, msg
    if len(img_data) == 0:
        msg = "No images to process."
        printt(msg)
        return msg, msg

    params = status_container.process_params
    is_processing = True
    # Get the list of image files in the folder
    total_images = len(img_data)

    # Total images, times number of images, plus 2 for load/unload
    total_supir_steps = total_images * num_images + 2 if apply_supir else 0

    # Total images, plus 2 for load/unload
    total_llava_steps = total_images + 2 if apply_llava else 0

    # Total steps, plus 1 for saving outputs
    total_steps = total_supir_steps + total_llava_steps + 1
    counter = 0
    # Disable llava for video...because...uh...yeah, video.
    # if status_container.is_video:
    #   apply_llava = False
    progress(0, desc=f"Processing {total_images} images...")
    printt(f"Processing {total_images} images...", reset=True)
    if apply_llava:
        printt('Processing LLaVA')
        last_result = llava_process(img_data, temperature, top_p, qs, save_captions, progress=progress)
        printt('LLaVA processing completed')

        if auto_unload_llava:
            unload_llava()
            printt('LLaVA unloaded')

        # Update the img_data from the captioner
        img_data = status_container.image_data
    # Check for cancellation
    if not is_processing and model is not None:
        progress(total_steps / total_steps, desc="Cancelling SUPIR...")
        all_to_cpu()
        return f"Batch Processing Completed: Cancelled at {time.ctime()}.", last_result
    counter += total_llava_steps
    if apply_supir:
        progress(counter / total_steps, desc="Processing images...")
        printt("Processing images (Stage 2)")
        last_result = supir_process(img_data, a_prompt, n_prompt, num_samples, upscale, edm_steps, s_stage1, s_stage2,
                                    s_cfg, seed, sampler_cls, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype,
                                    linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2, model_select,
                                    ckpt_select,
                                    num_images, random_seed, apply_llava, face_resolution, apply_bg, apply_face,
                                    face_prompt, unload=True, progress=progress)
        printt("Processing images (Stage 2) Completed")
    counter += total_supir_steps
    progress(counter / total_steps, desc="Processing completed.")

    if status_container.is_video and apply_supir:
        if not is_processing:
            printt("Processing cancelled, skipping video compilation.")
        printt("Processing outputs...")
        progress(counter / total_steps, desc="Compiling video...")
        extracted_folder = os.path.join(args.outputs_folder, "extracted_frames")
        output_quality = params.get('output_quality', 'medium')
        output_format = params.get('output_format', 'mp4')
        video_start = params.get('video_start', None)
        video_end = params.get('video_end', None)
        source_video_path = status_container.source_video_path
        media_output = compile_video(source_video_path, extracted_folder, args.outputs_folder,
                                     status_container.video_params,
                                     output_quality, output_format, video_start, video_end)
        if media_output:
            status_container.image_data = [media_output]
            printt("Video compiled successfully.")
        else:
            printt("Video compilation failed.")
    elif make_comparison_video:
        updates = []
        progress(counter / total_steps, desc="Creating comparison videos...")
        for image_data in status_container.image_data:
            output = save_compare_video(image_data, params)
            updates.append(output)
        status_container.image_data = updates
    progress(1, desc="Processing Completed in " + str(time.time() - start_time) + " seconds.")

    is_processing = False
    end_time = time.time()
    global unique_counter
    unique_counter = unique_counter + 1
    return f"Batch Processing Completed: processed {total_images * num_images} images at in {end_time - start_time:.2f} seconds #{unique_counter}", last_result


def save_image(image_data: MediaData, is_video_frame: bool):
    global status_container
    if len(image_data.metadata_list) >= 1:
        params = image_data.metadata_list[0]
    else:
        params = status_container.process_params
    save_caption = params.get('save_captions', False)
    output_dir = params.get('outputs_folder', args.outputs_folder)
    metadata_dir = os.path.join(output_dir, "images_meta_data")
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)
    results = image_data.outputs
    image_path = image_data.media_path
    result_paths = []
    for i, result in enumerate(results):
        event_dict = params.copy()

        if len(image_data.metadata_list) > i:
            event_dict = image_data.metadata_list[i]

        evt_id = event_dict.get('evt_id', str(time.time_ns()))

        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        if len(base_filename) > 250:
            base_filename = base_filename[:250]

        img = Image.fromarray(result)

        if args.log_history and not is_video_frame:
            history_path = f'./history/{evt_id[:5]}/{evt_id[5:]}'
            os.makedirs(history_path, exist_ok=True)
            with open(os.path.join(history_path, 'logs.txt'), 'w') as f:
                f.write(str(event_dict))
            img.save(os.path.join(history_path, f'HQ_{i}.png'))

        save_path = os.path.join(output_dir, f'{base_filename}.png')

        # If processing video, just save the image
        if is_video_frame:
            save_path = image_path
            img.save(save_path, "PNG")
        else:
            index = 1
            while os.path.exists(save_path):
                save_path = os.path.join(output_dir, f'{base_filename}_{str(index).zfill(4)}.png')
                index += 1
            remove_keys = ["face_gallery"]
            meta = PngImagePlugin.PngInfo()
            for key, value in event_dict.items():
                if key in remove_keys:
                    continue
                try:
                    renamed_key = rename_meta_key(key)
                    meta.add_text(renamed_key, str(value))
                except:
                    pass
            caption = image_data.caption.strip()
            if caption:
                meta.add_text('caption', caption)
                # This will save the caption to a file with the same name as the image
                if save_caption:
                    caption_path = f'{os.path.splitext(save_path)[0]}.txt'
                    with open(caption_path, 'w') as f:
                        f.write(caption)
            img.save(save_path, "PNG", pnginfo=meta)

            metadata_path = os.path.join(metadata_dir, f'{os.path.splitext(os.path.basename(save_path))[0]}.txt')
            with open(metadata_path, 'w') as f:
                for key, value in event_dict.items():
                    try:
                        f.write(f'{key}: {value}\n')
                    except:
                        pass
        result_paths.append(save_path)
    image_data.outputs = result_paths
    return image_data

def save_compare_video(image_data: MediaData, params):
    image_path = image_data.media_path
    output_dir = params.get('outputs_folder', args.outputs_folder)

    compare_videos_dir = os.path.join(output_dir, "compare_videos")
    if not os.path.exists(compare_videos_dir):
        os.makedirs(compare_videos_dir)

    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f'{base_filename}.mp4')
    index = 1
    while os.path.exists(save_path):
        save_path = os.path.join(output_dir, f'{base_filename}_{str(index).zfill(4)}.mp4')
        index += 1

    video_path = os.path.join(compare_videos_dir, f'{base_filename}.mp4')
    video_path = os.path.abspath(video_path)
    full_save_image_path = os.path.abspath(image_data.outputs[0])
    org_image_absolute_path = os.path.abspath(image_path)
    create_comparison_video(org_image_absolute_path, full_save_image_path, video_path, params)
    image_data.comparison_video = video_path
    return image_data

def stop_batch_upscale(progress=gr.Progress()):
    global is_processing
    progress(1, f"Stop command giving please wait to stop")
    print('\n***Stop command giving please wait to stop***\n')
    is_processing = False


def load_and_reset(param_setting):
    e_steps = 50  # steps
    sstage2 = 1  # Stage2 Guidance Strength
    sstage1 = -1.0  # Stage1 Guidance Strength
    schurn = 5  # S Churn default 5
    snoise = 1.003  # S Noise defualt 1.003
    ap = 'Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - ' \
         'realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore ' \
         'detailing, hyper sharpness, perfect without deformations.'
    np = 'painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, ' \
         '3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, ' \
         'signature, jpeg artifacts, deformed, lowres, over-smooth'
    cfix_type = 'Wavelet'
    l_s_stage2 = False  # Linear Stage2 Guidance checkbox
    l_s_s_stage2 = 0

    l_cfg = True  # Linear CFG checkbox
    if param_setting == "Quality":
        s_cfg = 7.5  # text cfg
        spt_linear_CFG = 4.0  # Linear CFG Start
    elif param_setting == "Fidelity":
        s_cfg = 4.0  # text cfg
        spt_linear_CFG = 1.0  # Linear CFG Start
    else:
        raise NotImplementedError
    return e_steps, s_cfg, sstage2, sstage1, schurn, snoise, ap, np, cfix_type, l_cfg, l_s_stage2, spt_linear_CFG, l_s_s_stage2


def submit_feedback(evt_id, f_score, f_text):
    if args.log_history:
        with open(f'./history/{evt_id[:5]}/{evt_id[5:]}/logs.txt', 'r') as f:
            event_dict = eval(f.read())
        f.close()
        event_dict['feedback'] = {'score': f_score, 'text': f_text}
        with open(f'./history/{evt_id[:5]}/{evt_id[5:]}/logs.txt', 'w') as f:
            f.write(str(event_dict))
        f.close()
        return 'Submit successfully, thank you for your comments!'
    else:
        return 'Submit failed, the server is not set to log history.'


preview_full = False


def toggle_full_preview():
    global preview_full
    gal_classes = ["preview_col"]
    btn_classes = ["slider_button"]

    if preview_full:
        preview_full = False
    else:
        preview_full = True
        gal_classes.append("full_preview")
        btn_classes.append("full")
    return gr.update(elem_classes=gal_classes), gr.update(elem_classes=btn_classes), gr.update(elem_classes=btn_classes)


def toggle_compare_elements(enable: bool) -> Tuple[gr.update, gr.update]:
    return gr.update(visible=enable), gr.update(visible=enable), gr.update(visible=enable)


title_md = """
# **SUPIR: Practicing Model Scaling for Photo-Realistic Image Restoration**

1 Click Installer (auto download models as well) : https://www.patreon.com/posts/99176057

FFmpeg Install Tutorial : https://youtu.be/-NjNy7afOQ0 &emsp; [[Paper](https://arxiv.org/abs/2401.13627)] &emsp; [[Project Page](http://supir.xpixel.group/)] &emsp; [[How to play](https://github.com/Fanghua-Yu/SUPIR/blob/master/assets/DemoGuide.png)]
"""

claim_md = """
## **Terms of use**

By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research. Please submit a feedback to us if you get any inappropriate answer! We will collect those to keep improving our models. For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.

## **License**
While the original readme for the project *says* it's non-commercial, it was *actually* released under the MIT license. That means that the project can be used for whatever you want.

And yes, it would certainly be nice if anything anybody stuck in a random readme were the ultimate gospel when it comes to licensing, unfortunately, that's just
not how the world works. MIT license means FREE FOR ANY PURPOSE, PERIOD.
The service is a research preview ~~intended for non-commercial use only~~, subject to the model [License](https://github.com/Fanghua-Yu/SUPIR#MIT-1-ov-file) of SUPIR.
"""

css_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'css', 'style.css'))
slider_css_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'css', 'nouislider.min.css'))

with open(css_file) as f:
    css = f.read()

with open(slider_css_file) as f:
    slider_css = f.read()

js_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'javascript', 'demo.js'))
no_slider_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'javascript', 'nouislider.min.js'))

with open(js_file) as f:
    js = f.read()

with open(no_slider_file) as f:
    no_slider = f.read()

head = f"""
<style media="screen">{css}</style>
<style media="screen">{slider_css}</style>
<script type="text/javascript">{js}</script>
<script type="text/javascript">{no_slider}</script>
"""

refresh_symbol = "\U000027F3"  # ⟳
dl_symbol = "\U00002B73"  # ⭳
fullscreen_symbol = "\U000026F6"  # ⛶


def update_meta(selected_file):
    # Returns [meta_image, meta_video]
    global meta_upload
    if meta_upload is not None and selected_file == "" and selected_file is not None:
        # Don't change if cleared from upload
        return gr.update(visible=True, value=None), gr.update()
    if is_video(selected_file):
        return gr.update(visible=False), gr.update(visible=True, value=selected_file)
    elif is_image(selected_file):
        return gr.update(visible=True, value=selected_file, sources=[]), gr.update(visible=False)
    else:
        return gr.update(visible=True, value=None), gr.update(visible=False)


def clear_meta():
    global meta_upload
    meta_upload = None
    # Returns [meta_image, meta_video, meta_file_browser, meta_file_upload, metadata_output]
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(value=None), gr.update(value=None)


default_llava_prompt = "Describe this image and its style in a very detailed manner. The image is a realistic photography, not an art painting."
prompt_styles = list_styles()
# Make a list of prompt_styles keys
prompt_styles_keys = list(prompt_styles.keys())

selected_pos, selected_neg, llava_style_prompt = select_style(
    prompt_styles_keys[0] if len(prompt_styles_keys) > 0 else "", default_llava_prompt, True)

block = gr.Blocks(title='SUPIR', theme=args.theme, css=css_file, head=head).queue()

with (block):
    with gr.Tab("Upscale"):
        # Execution buttons
        with gr.Column(scale=1):
            with gr.Row():
                start_single_button = gr.Button(value="Process Single")
                start_batch_button = gr.Button(value="Process Batch")
                stop_batch_button = gr.Button(value="Cancel")
                if args.outputs_folder_button:
                    btn_open_outputs = gr.Button("Open Outputs Folder")
                    btn_open_outputs.click(fn=open_folder)

        with gr.Column(scale=1):
            with gr.Row():
                output_label = gr.Label(label="Progress", elem_classes=["progress_label"])
            with gr.Row():
                target_res_textbox = gr.HTML(value="", visible=False, show_label=False)
        with gr.Row(equal_height=True):
            with gr.Column(elem_classes=['preview_col']) as input_col:
                src_input_file = gr.File(type="filepath", label="Input", elem_id="file-input",
                                         elem_classes=["preview_box"], height=400, visible=True, file_count="single",
                                         file_types=["image", "video"])
                src_image_display = gr.Image(type="filepath", elem_id="image-input", label="Input Image",
                                             elem_classes=["preview_box"], height=400, sources=[],
                                             visible=False)
                video_slider_display = gr.HTML(elem_id="video_slider_display", visible=False, value=slider_html)
                video_start_time_number = gr.Number(label="Start Time", value=0, visible=False, elem_id="start_time")
                video_end_time_number = gr.Number(label="End Time", value=0, visible=False, elem_id="end_time")
                video_current_time_number = gr.Number(label="Current Time", value=0, visible=False,
                                                      elem_id="current_time")
                video_fps_number = gr.Number(label="FPS", value=0, visible=False, elem_id="video_fps")
                video_total_frames_number = gr.Number(label="Total Frames", value=0, visible=False,
                                                      elem_id="total_frames")

            with gr.Column(visible=False, elem_classes=['preview_col']) as comparison_video_col:
                comparison_video = gr.Video(label="Comparison Video", elem_classes=["preview_box"], height=400,
                                            visible=False)
            with gr.Column(elem_classes=['preview_col'], elem_id="preview_column") as result_col:
                result_gallery = gr.Gallery(label='Output', elem_id="gallery2", elem_classes=["preview_box"],
                                            height=400, visible=False, rows=2, columns=4, allow_preview=True,
                                            show_download_button=False, show_share_button=False)
                result_slider = ImageSlider(label='Output', interactive=False, show_download_button=True,
                                            elem_id="gallery1",
                                            elem_classes=["preview_box", "preview_slider", "active"],
                                            height=400, container=True)
                result_video = gr.Video(label="Output Video", elem_classes=["preview_box"], height=400, visible=False)
                slider_dl_button = gr.Button(value=dl_symbol, elem_classes=["slider_button"], visible=True,
                                             elem_id="download_button")
                slider_full_button = gr.Button(value=fullscreen_symbol, elem_classes=["slider_button"], visible=True,
                                               elem_id="fullscreen_button")
        with gr.Row():
            with gr.Column():
                with gr.Accordion("General options", open=True):
                    if args.debug:
                        populate_slider_button = gr.Button(value="Populate Slider")
                        populate_gallery_button = gr.Button(value="Populate Gallery")
                        populate_slider_button.click(fn=populate_slider_single, outputs=[result_slider, result_gallery],
                                                     show_progress=True, queue=True)
                        populate_gallery_button.click(fn=populate_gallery, outputs=[result_gallery, result_slider],
                                                      show_progress=True, queue=True)
                    with gr.Row():
                        upscale_slider = gr.Slider(label="Upscale Size", minimum=1, maximum=8, value=1, step=0.1)
                    with gr.Row():
                        apply_llava_checkbox = gr.Checkbox(label="Apply LLaVa", value=False)
                        apply_supir_checkbox = gr.Checkbox(label="Apply SUPIR", value=True)
                    with gr.Row():
                        with gr.Column():
                            prompt_style_dropdown = gr.Dropdown(label="Prompt Style",
                                                                choices=prompt_styles_keys,
                                                                value=prompt_styles_keys[0] if len(
                                                                    prompt_styles_keys) > 0 else "")
                            refresh_styles_button = gr.Button(value=refresh_symbol, elem_classes=["refresh_button"],
                                                              size="sm")
                        show_select = args.ckpt_browser
                        with gr.Column():
                            with gr.Row(elem_id="model_select_row", visible=show_select):
                                ckpt_select_dropdown = gr.Dropdown(label="Model", choices=list_models(),
                                                                   value=selected_model(),
                                                                   interactive=True)
                                refresh_models_button = gr.Button(value=refresh_symbol, elem_classes=["refresh_button"],
                                                                  size="sm")
                    with gr.Row(elem_id="model_select_row", visible=show_select):
                        ckpt_type = gr.Dropdown(label="Checkpoint Type", choices=["Standard SDXL", "SDXL Lightning"],
                                                value="Standard SDXL")

                    prompt_textbox = gr.Textbox(label="Prompt", value="")
                    face_prompt_textbox = gr.Textbox(label="Face Prompt",
                                                     placeholder="Optional, uses main prompt if not provided",
                                                     value="")
                with gr.Accordion("LLaVA options", open=False):
                    with gr.Column():
                        auto_unload_llava_checkbox = gr.Checkbox(label="Auto Unload LLaVA (Low VRAM)",
                                                                 value=auto_unload)
                        setattr(auto_unload_llava_checkbox, "do_not_save_to_config", True)
                    qs_textbox = gr.Textbox(label="LLaVA prompt",value=llava_style_prompt)
                    temperature_slider = gr.Slider(label="Temperature", minimum=0., maximum=1.0, value=0.2, step=0.1)
                    top_p_slider = gr.Slider(label="Top P", minimum=0., maximum=1.0, value=0.7, step=0.1)

                with gr.Accordion("SUPIR options", open=False):
                    with gr.Row():
                        with gr.Column():
                            num_images_slider = gr.Slider(label="Number Of Images To Generate", minimum=1, maximum=200
                                                          , value=1, step=1)
                            num_samples_slider = gr.Slider(label="Batch Size", minimum=1,
                                                           maximum=4, value=1, step=1)
                        with gr.Column():
                            random_seed_checkbox = gr.Checkbox(label="Randomize Seed", value=True)
                    with gr.Row():
                        edm_steps_slider = gr.Slider(label="Steps", minimum=1, maximum=200, value=50, step=1)
                        s_cfg_slider = gr.Slider(label="Text Guidance Scale", minimum=1.0, maximum=15.0, value=7.5,
                                                 step=0.1)
                        s_stage2_slider = gr.Slider(label="Stage2 Guidance Strength", minimum=0., maximum=1., value=1.,
                                                    step=0.05)
                        s_stage1_slider = gr.Slider(label="Stage1 Guidance Strength", minimum=-1.0, maximum=6.0,
                                                    value=-1.0,
                                                    step=1.0)
                        seed_slider = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                        sampler_dropdown = gr.Dropdown(label="Sampler", choices=["EDM", "DPMPP2M"],
                                                       value="DPMPP2M")
                        s_churn_slider = gr.Slider(label="S-Churn", minimum=0, maximum=40, value=5, step=1)
                        s_noise_slider = gr.Slider(label="S-Noise", minimum=1.0, maximum=1.1, value=1.003, step=0.001)

                    with gr.Row():
                        a_prompt_textbox = gr.Textbox(label="Default Positive Prompt",
                                                      value=selected_pos)
                        n_prompt_textbox = gr.Textbox(label="Default Negative Prompt",
                                                      value=selected_neg)
                with gr.Accordion("Video options", open=False):
                    with gr.Column():
                        output_vq_slider = gr.Slider(label="Output Video Quality", minimum=0.1, maximum=1.0, value=0.6,
                                                     step=0.1)
                        output_vf_dropdown = gr.Dropdown(label="Video Format", choices=["mp4", "mkv"], value="mp4")

            with gr.Column():
                with gr.Accordion("Batch options", open=True):
                    with gr.Row():
                        with gr.Column():
                            batch_process_folder_textbox = gr.Textbox(
                                label="Batch Input Folder - Can use image captions from .txt",
                                placeholder="R:\SUPIR video\comparison_images")
                            outputs_folder_textbox = gr.Textbox(
                                label="Batch Output Path - Leave empty to save to default.",
                                placeholder="R:\SUPIR video\comparison_images\outputs")
                            save_captions_checkbox = gr.Checkbox(label="Save Captions",
                                                                 value=True)
                with gr.Accordion("Advanced options", open=False):
                    with gr.Row():
                        with gr.Column():
                            param_setting_select = gr.Dropdown(["Quality", "Fidelity"], interactive=True,
                                                               label="Param Setting",
                                                               value="Quality")
                        with gr.Column():
                            reset_button = gr.Button(value="Reset Param", scale=2)
                    with gr.Row():
                        with gr.Column():
                            linear_cfg_checkbox = gr.Checkbox(label="Linear CFG", value=True)
                            spt_linear_cfg_slider = gr.Slider(label="CFG Start", minimum=1.0,
                                                              maximum=9.0, value=4.0, step=0.5)
                        with gr.Column():
                            linear_s_stage2_checkbox = gr.Checkbox(label="Linear Stage2 Guidance", value=False)
                            spt_linear_s_stage2_slider = gr.Slider(label="Guidance Start", minimum=0,
                                                                   maximum=1, value=0, step=0.05)
                    with gr.Row():
                        with gr.Column():
                            diff_dtype_radio = gr.Radio(['fp32', 'fp16', 'bf16'], label="Diffusion Data Type",
                                                        value="bf16",
                                                        interactive=True)
                        with gr.Column():
                            ae_dtype_radio = gr.Radio(['fp32', 'bf16'], label="Auto-Encoder Data Type", value="bf16",
                                                      interactive=True)
                        with gr.Column():
                            color_fix_type_radio = gr.Radio(["None", "AdaIn", "Wavelet"], label="Color-Fix Type",
                                                            value="Wavelet",
                                                            interactive=True)
                        with gr.Column():
                            model_select_radio = gr.Radio(["v0-Q", "v0-F"], label="Model Selection", value="v0-Q",
                                                          interactive=True)
                with gr.Accordion("Face options", open=False):
                    face_resolution_slider = gr.Slider(label="Text Guidance Scale", minimum=256, maximum=2048,
                                                       value=1024,
                                                       step=32)
                    with gr.Row():
                        with gr.Column():
                            apply_bg_checkbox = gr.Checkbox(label="BG restoration", value=False)
                        with gr.Column():
                            apply_face_checkbox = gr.Checkbox(label="Face restoration", value=False)
                with gr.Accordion("Comparison Video options", open=False):
                    with gr.Row():
                        make_comparison_video_checkbox = gr.Checkbox(
                            label="Generate Comparison Video", value=False)
                    with gr.Row(visible=True) as compare_video_row:
                        video_duration_textbox = gr.Textbox(label="Duration", value="5")
                        video_fps_textbox = gr.Textbox(label="FPS", value="30")
                        video_width_textbox = gr.Textbox(label="Width", value="1920")
                        video_height_textbox = gr.Textbox(label="Height", value="1080")

    with gr.Tab("Restored Faces"):
        with gr.Row():
            face_gallery = gr.Gallery(label='Faces', show_label=False, elem_id="gallery2")

    with gr.Tab("Outputs", elem_id="output_tab") as outputsTab:
        with gr.Row(elem_id="output_view_row"):
            with gr.Column(elem_classes=["output_view_col"]):
                with gr.Row():
                    meta_file_browser = gr.FileExplorer(label="Output Folder", file_count="single",
                                                        elem_id="output_folder",
                                                        root_dir=args.outputs_folder, height="85vh")
                    output_files_refresh_btn = gr.Button(value=refresh_symbol, elem_classes=["refresh_button"],
                                                         size="sm")


                    def refresh_output_files():
                        return gr.update(value=args.outputs_folder)


                    output_files_refresh_btn.click(fn=refresh_output_files, outputs=[meta_file_browser],
                                                   show_progress=True, queue=True)
            with gr.Column(elem_classes=["output_view_col"]):
                meta_image = gr.Image(type="filepath", label="Output Image", elem_id="output_image", visible=True, height="42.5vh", sources=["upload"])
                meta_video = gr.Video(label="Output Video", elem_id="output_video", visible=False, height="42.5vh")
                metadata_output = gr.Textbox(label="Image Metadata", lines=25, max_lines=50, elem_id="output_metadata")

    with gr.Tab("About"):
        gr.HTML(f"<H2>SUPIR Version {SUPIR_REVISION}</H2>")
        gr.Markdown(title_md)
        with gr.Row():
            gr.Markdown(claim_md)
            event_id = gr.Textbox(label="Event ID", value="", visible=False)
        with gr.Accordion("Feedback", open=False):
            fb_score = gr.Slider(label="Feedback Score", minimum=1, maximum=5, value=3, step=1,
                                 interactive=True)
            fb_text = gr.Textbox(label="Feedback Text", value="",
                                 placeholder='Please enter your feedback here.')
            submit_button = gr.Button(value="Submit Feedback")

    # prompt_el, result_gallery_el, result_slider_el, result_video_el, comparison_video_el, event_id_el, seed_el, face_gallery_el
    output_elements = [
        prompt_textbox, result_gallery, result_slider, result_video, comparison_video, event_id, seed_slider,
        face_gallery
    ]

    refresh_models_button.click(fn=refresh_models_click, outputs=[ckpt_select_dropdown])
    refresh_styles_button.click(fn=refresh_styles_click, outputs=[prompt_style_dropdown])

    elements_dict = {
        "a_prompt": a_prompt_textbox,
        "ae_dtype": ae_dtype_radio,
        "apply_bg": apply_bg_checkbox,
        "apply_face": apply_face_checkbox,
        "apply_llava": apply_llava_checkbox,
        "apply_supir": apply_supir_checkbox,
        "auto_unload_llava": auto_unload_llava_checkbox,
        "batch_process_folder": batch_process_folder_textbox,
        "ckpt_select": ckpt_select_dropdown,
        "color_fix_type": color_fix_type_radio,
        "diff_dtype": diff_dtype_radio,
        "edm_steps": edm_steps_slider,
        "face_prompt": face_prompt_textbox,
        "face_resolution": face_resolution_slider,
        "linear_CFG": linear_cfg_checkbox,
        "linear_s_stage2": linear_s_stage2_checkbox,
        "main_prompt": prompt_textbox,
        "make_comparison_video": make_comparison_video_checkbox,
        "model_select": model_select_radio,
        "n_prompt": n_prompt_textbox,
        "num_images": num_images_slider,
        "num_samples": num_samples_slider,
        "output_video_format": output_vf_dropdown,
        "output_video_quality": output_vq_slider,
        "outputs_folder": outputs_folder_textbox,
        "qs": qs_textbox,
        "random_seed": random_seed_checkbox,
        "s_cfg": s_cfg_slider,
        "s_churn": s_churn_slider,
        "s_noise": s_noise_slider,
        "s_stage1": s_stage1_slider,
        "s_stage2": s_stage2_slider,
        "sampler": sampler_dropdown,
        "save_captions": save_captions_checkbox,
        "seed": seed_slider,
        "spt_linear_CFG": spt_linear_cfg_slider,
        "spt_linear_s_stage2": spt_linear_s_stage2_slider,
        "src_file": src_input_file,
        "temperature": temperature_slider,
        "top_p": top_p_slider,
        "upscale": upscale_slider,
        "video_duration": video_duration_textbox,
        "video_end": video_end_time_number,
        "video_fps": video_fps_textbox,
        "video_height": video_height_textbox,
        "video_start": video_start_time_number,
        "video_width": video_width_textbox,
    }

    extra_info_elements = {
        "prompt_style": prompt_style_dropdown,
        "checkpoint_type": ckpt_type,
    }

    elements_dict = set_info_attributes(elements_dict)

    # Add items here that are not passed to processing for labels
    extra_info_elements = set_info_attributes(extra_info_elements)

    elements = list(elements_dict.values())

    start_single_button.click(fn=start_single_process, inputs=elements, outputs=output_label,
                              show_progress=True, queue=True)
    start_batch_button.click(fn=start_batch_process, inputs=elements, outputs=output_label,
                             show_progress=True, queue=True)
    stop_batch_button.click(fn=stop_batch_upscale, show_progress=True, queue=True)
    reset_button.click(fn=load_and_reset, inputs=[param_setting_select],
                       outputs=[edm_steps_slider, s_cfg_slider, s_stage2_slider, s_stage1_slider, s_churn_slider,
                                s_noise_slider, a_prompt_textbox, n_prompt_textbox,
                                color_fix_type_radio, linear_cfg_checkbox, linear_s_stage2_checkbox,
                                spt_linear_cfg_slider, spt_linear_s_stage2_slider])

    # We just read the output_label and update all the elements when we find "Processing Complete"
    output_label.change(fn=update_elements, show_progress=False, queue=True, inputs=[output_label],
                        outputs=output_elements)

    meta_file_browser.change(fn=update_meta, inputs=[meta_file_browser], outputs=[meta_image, meta_video])
    meta_image.change(fn=read_image_metadata, inputs=[meta_image], outputs=[metadata_output])

    prompt_style_dropdown.change(fn=select_style, inputs=[prompt_style_dropdown, qs_textbox],
                                 outputs=[a_prompt_textbox, n_prompt_textbox, qs_textbox])

    make_comparison_video_checkbox.change(fn=toggle_compare_elements, inputs=[make_comparison_video_checkbox],
                                          outputs=[comparison_video_col, compare_video_row, comparison_video])
    submit_button.click(fn=submit_feedback, inputs=[event_id, fb_score, fb_text], outputs=[fb_text])
    upscale_slider.change(fn=update_target_resolution, inputs=[src_image_display, upscale_slider],
                          outputs=[target_res_textbox])

    # slider_dl_button.click(fn=download_slider_image, inputs=[result_slider], show_progress=False, queue=True)
    slider_full_button.click(fn=toggle_full_preview, outputs=[result_col, slider_full_button, slider_dl_button],
                             show_progress=False, queue=True, js="toggleFullscreen")

    input_elements = [src_input_file, src_image_display, video_slider_display, target_res_textbox,
                      video_start_time_number, video_end_time_number, video_current_time_number, video_fps_number,
                      video_total_frames_number]
    src_input_file.change(fn=update_inputs, inputs=[src_input_file, upscale_slider],
                          outputs=input_elements)
    src_image_display.clear(fn=update_inputs, inputs=[src_image_display, upscale_slider],
                            outputs=input_elements)

    model_settings_elements = [s_cfg_slider, spt_linear_cfg_slider, edm_steps_slider]

    ckpt_type.change(fn=update_model_settings, inputs=[ckpt_type, param_setting_select],
                     outputs=model_settings_elements)

    video_sliders = [video_start_time_number, video_current_time_number, video_end_time_number,
                     video_fps_number, video_total_frames_number, src_input_file, upscale_slider]
    video_current_time_number.change(fn=update_video_slider, inputs=video_sliders,
                                     outputs=[src_image_display, target_res_textbox], js="update_slider")
    video_start_time_number.change(fn=update_start_time,
                                   inputs=[src_input_file, upscale_slider, video_start_time_number],
                                   outputs=target_res_textbox)
    video_end_time_number.change(fn=update_end_time, inputs=[src_input_file, upscale_slider, video_end_time_number],
                                 outputs=target_res_textbox)


    def do_nothing():
        pass


    slider_dl_button.click(js="downloadImage", show_progress=False, queue=True, fn=do_nothing)

if args.port is not None:  # Check if the --port argument is provided
    block.launch(server_name=server_ip, server_port=args.port, share=args.share, inbrowser=args.open_browser)    
else:
    block.launch(server_name=server_ip, share=args.share, inbrowser=args.open_browser)

   