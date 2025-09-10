

import argparse
import datetime
import gc
import os
import shutil
import tempfile
import threading
import time
import traceback
import json
from datetime import datetime
from typing import Tuple, List, Any, Dict
import subprocess

import einops
import gradio as gr
import numpy as np
import requests
import torch
from PIL import Image, ImageFile
from PIL import PngImagePlugin
from gradio_imageslider import ImageSlider
import pillow_avif  # Import the AVIF plugin
from PIL import UnidentifiedImageError

# Enable loading of truncated images automatically
ImageFile.LOAD_TRUNCATED_IMAGES = True

import ui_helpers
from SUPIR.models.SUPIR_model import SUPIRModel
from SUPIR.util import HWC3, upscale_image, convert_dtype
from SUPIR.util import create_SUPIR_model
from SUPIR.utils import shared
from SUPIR.utils.compare import create_comparison_video
from SUPIR.utils.face_restoration_helper import FaceRestoreHelper
from SUPIR.utils.model_fetch import get_model
from SUPIR.utils.rename_meta import rename_meta_key, rename_meta_key_reverse
from SUPIR.utils.ckpt_downloader import download_checkpoint_handler, download_checkpoint

from SUPIR.utils.status_container import StatusContainer, MediaData
from llava.llava_agent import LLavaAgent
from ui_helpers import is_video, extract_video, compile_video, is_image, get_video_params, printt

SUPIR_REVISION = "v52"

def get_recent_images(num_images=20):
    """
    Get a list of recent image files from the outputs folder without scanning subfolders.
    """
    try:
        if not os.path.exists(args.outputs_folder):
            os.makedirs(args.outputs_folder, exist_ok=True)
            return []
        
        # Get all image files from only the main outputs folder, not subfolders
        image_files = []
        # Instead of os.walk, just list files in the main directory
        for file in os.listdir(args.outputs_folder):
            file_path = os.path.join(args.outputs_folder, file)
            # Only process files (not directories) with image extensions
            if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.avif')):
                # Get file modification time
                mtime = os.path.getmtime(file_path)
                image_files.append((file_path, mtime))
        
        # Sort by modification time, most recent first
        image_files.sort(key=lambda x: x[1], reverse=True)
        
        # Extract just the paths for the most recent images
        recent_paths = [item[0] for item in image_files[:num_images]]
        
        return recent_paths
    except Exception as e:
        print(f"Error getting recent images: {str(e)}")
        return []

def update_comparison_images(image1_path, image2_path):
    """
    Update the image comparison slider with the selected images.
    Resize smaller image to match the larger image's dimensions using nearest neighbor interpolation.
    """
    if image1_path and image2_path:
        try:
            img1 = safe_open_image(image1_path)
            img2 = safe_open_image(image2_path)
            
            # Get dimensions of both images
            width1, height1 = img1.size
            width2, height2 = img2.size
            
            # Determine which image is larger
            if (width1 * height1) > (width2 * height2):
                # img1 is larger, resize img2 to match img1's dimensions
                img2 = img2.resize((width1, height1), Image.NEAREST)
            elif (width2 * height2) > (width1 * height1):
                # img2 is larger, resize img1 to match img2's dimensions
                img1 = img1.resize((width2, height2), Image.NEAREST)
            
            return gr.update(visible=True, value=(img1, img2)), gr.update(value=f"Comparing: {os.path.basename(image1_path)} and {os.path.basename(image2_path)}")
        except Exception as e:
            return gr.update(visible=False), gr.update(value=f"Error loading images: {str(e)}")
    else:
        return gr.update(visible=False), gr.update(value="Please select two images to compare")

def refresh_image_list():
    """
    Refresh the list of recent images for the comparison tab.
    """
    recent_images = get_recent_images(20)
    return gr.update(value=recent_images)

# Global variables to keep track of selected images
selected_image1 = None
selected_image2 = None

def on_image_select_gallery1(evt: gr.SelectData, gallery_images):
    """Handle image selection from the first gallery."""
    global selected_image1
    
    if not gallery_images or evt.index >= len(gallery_images):
        return gr.update(value="Invalid selection"), gr.update(), gr.update()
    
    # Gallery images from get_recent_images should be paths, not tuples
    selected_image1 = gallery_images[evt.index]
    # Check if it's a tuple and extract the path if needed
    if isinstance(selected_image1, tuple):
        selected_image1 = selected_image1[0] if selected_image1 else None
    
    if not selected_image1:
        return gr.update(value="Invalid selection"), gr.update(), gr.update()
        
    message = f"Selected image 1: {os.path.basename(selected_image1)}"
    
    if selected_image2:
        # Check if selected_image2 is a tuple and extract the path if needed
        image2_path = selected_image2
        if isinstance(selected_image2, tuple):
            image2_path = selected_image2[0] if selected_image2 else None
            
        if not image2_path:
            return gr.update(value=message), gr.update(), gr.update(value=selected_image1)
            
        message += f" and image 2: {os.path.basename(image2_path)}"
        
        try:
            img1 = safe_open_image(selected_image1)
            img2 = safe_open_image(image2_path)
            
            # Get dimensions of both images
            width1, height1 = img1.size
            width2, height2 = img2.size
            
            # Determine which image is larger
            if (width1 * height1) > (width2 * height2):
                # img1 is larger, resize img2 to match img1's dimensions
                img2 = img2.resize((width1, height1), Image.NEAREST)
            elif (width2 * height2) > (width1 * height1):
                # img2 is larger, resize img1 to match img2's dimensions
                img1 = img1.resize((width2, height2), Image.NEAREST)
                
            return gr.update(value=message), gr.update(visible=True, value=(img1, img2)), gr.update(value=selected_image1)
        except Exception as e:
            return gr.update(value=f"Error loading images: {str(e)}"), gr.update(visible=False), gr.update(value=selected_image1)
    
    return gr.update(value=message), gr.update(), gr.update(value=selected_image1)

def on_image_select_gallery2(evt: gr.SelectData, gallery_images):
    """Handle image selection from the second gallery."""
    global selected_image2
    
    if not gallery_images or evt.index >= len(gallery_images):
        return gr.update(value="Invalid selection"), gr.update(), gr.update()
    
    # Gallery images from get_recent_images should be paths, not tuples
    selected_image2 = gallery_images[evt.index]
    # Check if it's a tuple and extract the path if needed
    if isinstance(selected_image2, tuple):
        selected_image2 = selected_image2[0] if selected_image2 else None
    
    if not selected_image2:
        return gr.update(value="Invalid selection"), gr.update(), gr.update()
        
    message = f"Selected image 2: {os.path.basename(selected_image2)}"
    
    if selected_image1:
        # Check if selected_image1 is a tuple and extract the path if needed
        image1_path = selected_image1
        if isinstance(selected_image1, tuple):
            image1_path = selected_image1[0] if selected_image1 else None
            
        if not image1_path:
            return gr.update(value=message), gr.update(), gr.update(value=selected_image2)
            
        message = f"Selected image 1: {os.path.basename(image1_path)} and image 2: {os.path.basename(selected_image2)}"
        
        try:
            img1 = safe_open_image(image1_path)
            img2 = safe_open_image(selected_image2)
            
            # Get dimensions of both images
            width1, height1 = img1.size
            width2, height2 = img2.size
            
            # Determine which image is larger
            if (width1 * height1) > (width2 * height2):
                # img1 is larger, resize img2 to match img1's dimensions
                img2 = img2.resize((width1, height1), Image.NEAREST)
            elif (width2 * height2) > (width1 * height1):
                # img2 is larger, resize img1 to match img2's dimensions
                img1 = img1.resize((width2, height2), Image.NEAREST)
                
            return gr.update(value=message), gr.update(visible=True, value=(img1, img2)), gr.update(value=selected_image2)
        except Exception as e:
            return gr.update(value=f"Error loading images: {str(e)}"), gr.update(visible=False), gr.update(value=selected_image2)
    
    return gr.update(value=message), gr.update(), gr.update(value=selected_image2)

def clear_selected_images():
    """Clear selected images and refresh the galleries."""
    global selected_image1, selected_image2
    selected_image1 = None
    selected_image2 = None
    recent_images = get_recent_images(20)
    return gr.update(value=recent_images), gr.update(value=recent_images), gr.update(value="Select one image from each gallery below"), gr.update(visible=False), gr.update(value=""), gr.update(value="")

def compare_selected_images(image1_path, image2_path, uploaded_img1=None, uploaded_img2=None):
    """
    Compare images from either the galleries or uploads.
    Resize smaller image to match the larger image's dimensions using nearest neighbor interpolation.
    """
    img1_path = image1_path if image1_path else uploaded_img1
    img2_path = image2_path if image2_path else uploaded_img2
    
    if not img1_path or not img2_path:
        return gr.update(visible=False), gr.update(value="Please select two images to compare")
    
    try:
        img1 = safe_open_image(img1_path)
        img2 = safe_open_image(img2_path)
        
        # Get dimensions of both images
        width1, height1 = img1.size
        width2, height2 = img2.size
        
        # Determine which image is larger
        if (width1 * height1) > (width2 * height2):
            # img1 is larger, resize img2 to match img1's dimensions
            img2 = img2.resize((width1, height1), Image.NEAREST)
        elif (width2 * height2) > (width1 * height1):
            # img2 is larger, resize img1 to match img2's dimensions
            img1 = img1.resize((width2, height2), Image.NEAREST)
        
        message = f"Comparing: {os.path.basename(img1_path)} and {os.path.basename(img2_path)}"
        return gr.update(visible=True, value=(img1, img2)), gr.update(value=message)
    except Exception as e:
        return gr.update(visible=False), gr.update(value=f"Error loading images: {str(e)}")

# Comment out toggle_compare_fullscreen function
"""
def toggle_compare_fullscreen():
    # Toggle fullscreen for the comparison slider.
    # Returns compare_result_col, fullscreen button, download button
    return (
        gr.update(elem_classes=["preview_col", "full_preview"]), 
        gr.update(elem_classes=["slider_button", "full"]), 
        gr.update(elem_classes=["slider_button", "full"])
    )
"""

parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default='127.0.0.1', help="IP address for the server to listen on.")
parser.add_argument("--share", type=str, default=False, help="Set to True to share the app publicly.")
parser.add_argument("--port", type=int, help="Port number for the server to listen on.")
parser.add_argument("--log_history", action='store_true', default=False, help="Enable logging of request history.")
parser.add_argument("--loading_half_params", action='store_true', default=False,
                    help="Enable loading model parameters in half precision to reduce memory usage.")
parser.add_argument("--fp8", action='store_true', default=False, 
                    help="Enable loading model parameters in FP8 precision to reduce memory usage.")
parser.add_argument("--autotune", action='store_true', default=False, help="Automatically set precision parameters based on the amount of VRAM available.")
parser.add_argument("--fast_load_sd", action='store_true', default=False, 
                    help="Enable fast loading of model state dict and to prevents unnecessary memory allocation.")
parser.add_argument("--use_tile_vae", action='store_true', default=False,
                    help="Enable tiling for the VAE to handle larger images with limited memory.")
parser.add_argument("--outputs_folder_button",action='store_true', default=False, help="Outputs Folder Button Will Be Enabled")
parser.add_argument("--use_fast_tile", action='store_true', default=False,
                    help="Use a faster tile encoding/decoding, may impact quality.")
parser.add_argument("--encoder_tile_size", type=int, default=512,
                    help="Tile size for the encoder. Larger sizes may improve quality but require more memory.")
parser.add_argument("--decoder_tile_size", type=int, default=64,
                    help="Tile size for the decoder. Larger sizes may improve quality but require more memory.")
parser.add_argument("--load_8bit_llava", action='store_true', default=False,
                    help="Load the LLAMA model in 8-bit precision to save memory.")
parser.add_argument("--load_4bit_llava", action='store_true', default=True,
                    help="Load the LLAMA model in 4-bit precision to significantly reduce memory usage.")
parser.add_argument("--ckpt", type=str, default='Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors',
                    help="Path to the checkpoint file for the model.")
parser.add_argument("--ckpt_browser", action='store_true', default=True, help="Enable a checkpoint selection dropdown.")
parser.add_argument("--ckpt_dir", type=str, default='models/checkpoints',
                    help="Directory where model checkpoints are stored.")
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
bf16_supported = torch.cuda.is_bf16_supported()
total_vram = 100000
auto_unload = False
if torch.cuda.is_available() and args.autotune:
    # Get total GPU memory
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print("Autotune enabled, Total VRAM: ", total_vram, "GB")
    if not args.fp8:
        args.fp8 = total_vram <= 8
    auto_unload = total_vram <= 12

    if total_vram <= 24:
        if not args.loading_half_params:
            args.loading_half_params = True
        if not args.use_tile_vae:
            args.use_tile_vae = True
    print("Auto Unload: ", auto_unload)
    print("Half Params: ", args.loading_half_params)
    print("FP8: ", args.fp8)
    print("Tile VAE: ", args.use_tile_vae)

shared.opts.half_mode = args.loading_half_params  
shared.opts.fast_load_sd = args.fast_load_sd

# Add this function after imports and before any other functions
def safe_open_image(image_path):
    """Safely open any image format including AVIF with fallback options."""
    try:
        # Try to open normally first
        return Image.open(image_path)
    except UnidentifiedImageError as e:
        print(f"Error opening image with PIL: {str(e)}. Attempting to convert...")
        try:
            # Create a temporary file for the converted image
            import tempfile
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
            
            # Try several methods to convert the image
            converted = False
            
            # 1. Try using pillow_avif directly
            try:
                with Image.open(image_path) as img:
                    img.save(temp_output, format='PNG')
                converted = True
            except Exception as conversion_error:
                print(f"Direct conversion failed: {str(conversion_error)}")
            
            # 2. Try using Wand (ImageMagick binding)
            if not converted:
                try:
                    from wand.image import Image as WandImage
                    with WandImage(filename=image_path) as wand_img:
                        wand_img.save(filename=temp_output)
                    converted = True
                    print(f"Converted image using Wand")
                except Exception as wand_error:
                    print(f"Wand conversion failed: {str(wand_error)}")
            
            # 3. Try using system commands
            if not converted:
                import subprocess
                try:
                    if os.name == 'nt':  # Windows
                        subprocess.run(['magick', 'convert', image_path, temp_output], check=True)
                    else:  # Linux/Mac
                        subprocess.run(['convert', image_path, temp_output], check=True)
                    converted = True
                    print(f"Converted image using system commands")
                except Exception as cmd_error:
                    print(f"System command conversion failed: {str(cmd_error)}")
            
            if converted:
                return Image.open(temp_output)
            else:
                raise Exception("All conversion methods failed")
        except Exception as e:
            print(f"Failed to convert image: {str(e)}")
            raise
    except Exception as e:
        print(f"Error opening image: {str(e)}")
        raise

def sanitize_filename_part(text):
    """Sanitize a string to be safe for use in filenames."""
    if not text:
        return ""
    
    # Characters that are invalid in Windows filenames
    invalid_chars = '<>:"/\\|?*'
    # Additional problematic characters
    invalid_chars += '\x00\r\n\t'
    
    # Replace invalid characters with underscore
    sanitized = text
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')
    
    # Remove control characters (0-31) and DEL (127)
    sanitized = ''.join(char if ord(char) > 31 and ord(char) != 127 else '_' for char in sanitized)
    
    # Remove leading/trailing spaces and dots (Windows doesn't like them)
    sanitized = sanitized.strip(' .')
    
    # Limit length to prevent issues (keep it reasonable for prefix/suffix)
    max_length = 50
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized

def apply_metadata(image_path):
    global elements_dict, extra_info_elements
    
    if image_path is None:
        return [gr.update(value="No image selected")] + [gr.update() for _ in range(len(elements_dict) + len(extra_info_elements))]

    # Open the image and extract metadata
    try:
        with safe_open_image(image_path) as img:
            metadata = img.info
           
            # First update is for output_label
            all_updates = [gr.update(value=f"Applied metadata from {os.path.basename(image_path)}")]
            
            # Add default update for each UI element
            for _ in elements_dict:
                all_updates.append(gr.update())
            for _ in extra_info_elements:
                all_updates.append(gr.update())
                
            # Map to track which elements have been updated
            updated_elements = set()
            
            # Special handling for the caption - look for it first
            caption_value = None
            for key in ["Used Final Prompt", "caption"]:
                if key in metadata:
                    caption_value = metadata[key]
                    break
                    
            if caption_value and "main_prompt" in elements_dict:
                # Get the index of main_prompt in the list (add 1 for output_label)
                main_prompt_index = list(elements_dict.keys()).index("main_prompt") + 1
                # Set the update for this element
                all_updates[main_prompt_index] = gr.update(value=caption_value)
                updated_elements.add("main_prompt")
                
            # Process the rest of the metadata
            for key, value in metadata.items():
                try:
                    # Try to use the key directly or find a renamed key
                    renamed_key = rename_meta_key_reverse(key)
                    
                    if renamed_key in elements_dict and renamed_key not in updated_elements:
                        # Get the index of the element in the list (add 1 for output_label)
                        index = list(elements_dict.keys()).index(renamed_key) + 1
                        
                        # Convert string boolean values to actual booleans for checkboxes
                        element = elements_dict[renamed_key]
                        if isinstance(element, gr.Checkbox):
                            if isinstance(value, str):
                                if value.lower() == "true":
                                    value = True
                                elif value.lower() == "false":
                                    value = False
                        # Convert numeric strings to numbers for sliders and number inputs
                        elif isinstance(element, (gr.Slider, gr.Number)):
                            try:
                                if isinstance(value, str):
                                    if '.' in value:
                                        value = float(value)
                                    else:
                                        value = int(value)
                            except (ValueError, TypeError):
                                pass  # Keep as string if conversion fails
                                
                        # Set the update for this element
                        all_updates[index] = gr.update(value=value)
                        updated_elements.add(renamed_key)
                    elif renamed_key in extra_info_elements and renamed_key not in updated_elements:
                        # Get the index in the combined list (after elements_dict and add 1 for output_label)
                        index = 1 + len(elements_dict) + list(extra_info_elements.keys()).index(renamed_key)
                        
                        # Convert string boolean values to actual booleans for checkboxes
                        element = extra_info_elements[renamed_key]
                        if isinstance(element, gr.Checkbox):
                            if isinstance(value, str):
                                if value.lower() == "true":
                                    value = True
                                elif value.lower() == "false":
                                    value = False
                        
                        all_updates[index] = gr.update(value=value)
                        updated_elements.add(renamed_key)
                except Exception as e:
                    print(f"Error processing metadata key '{key}': {str(e)}")
            
            return all_updates
    except Exception as e:
        print(f"Error applying metadata: {str(e)}")
        return [gr.update(value=f"Error applying metadata: {str(e)}")] + [gr.update() for _ in range(len(elements_dict) + len(extra_info_elements))]

if args.fp8:
    shared.opts.half_mode = args.fp8
    shared.opts.fp8_storage = args.fp8

server_ip = args.ip
if args.debug:
    args.open_browser = False

if args.ckpt_dir == "models/checkpoints":
    args.ckpt_dir = os.path.join(os.path.dirname(__file__), args.ckpt_dir)
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir, exist_ok=True)

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
llava_agent = None
models_loaded = False
unique_counter = 0
status_container = StatusContainer()

# Store this globally so we can update variables more easily
elements_dict = {}
extra_info_elements = {}

# Force initialization of processing state
single_process = False
is_processing = False
last_used_checkpoint = None

# Initialize batch processing variables to ensure clean state
batch_start_time = None
batch_total_count = 0
batch_processed_count = 0
batch_processing_times = []
batch_current_stage = ""
batch_current_image_name = ""


print(f"App startup: is_processing = {is_processing}")


def get_progress_display_html(title, content):
    """Generate HTML structure for progress display"""
    return f"""
    <div style="
        position: fixed !important;
        top: 20px !important;
        left: 20px !important;
        width: 45vw !important;
        max-width: 600px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; 
        color: white !important; 
        padding: 8px 12px !important; 
        border-radius: 12px !important; 
        font-family: 'Segoe UI', Arial, sans-serif !important; 
        box-shadow: 0 8px 32px rgba(0,0,0,0.8) !important;
        border: 3px solid #4299e1 !important;
        z-index: 9999999 !important;
        pointer-events: none !important;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <h2 style="margin: 0; font-size: 16px; font-weight: 700; color: #ffffff; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                {title}
            </h2>
        </div>
        {content}
    </div>
    """

def get_batch_progress_html():
    """Generate HTML for persistent batch progress display"""
    global batch_start_time, batch_processed_count, batch_total_count, batch_processing_times, batch_current_stage, batch_current_image_name
    
    if not is_processing or batch_total_count == 0:
        return ""
    
    # Calculate metrics
    remaining = batch_total_count - batch_processed_count
    
    # Calculate average processing time and ETA
    avg_time = 0
    eta_text = "Calculating..."
    if len(batch_processing_times) > 0:
        avg_time = sum(batch_processing_times) / len(batch_processing_times)
        if remaining > 0:
            eta_seconds = remaining * avg_time
            eta_minutes = int(eta_seconds // 60)
            eta_secs = int(eta_seconds % 60)
            eta_text = f"{eta_minutes}m {eta_secs}s"
        else:
            eta_text = "Complete"
    
    # Calculate elapsed time
    elapsed = "0m 0s"
    if batch_start_time:
        elapsed_seconds = time.time() - batch_start_time
        elapsed_minutes = int(elapsed_seconds // 60)
        elapsed_secs = int(elapsed_seconds % 60)
        elapsed = f"{elapsed_minutes}m {elapsed_secs}s"
    
    # Progress percentage
    progress_percent = (batch_processed_count / batch_total_count * 100) if batch_total_count > 0 else 0
    
    # Build progress content
    progress_counter = f"""
        <div style="background: rgba(255,255,255,0.25); color: white; padding: 8px 16px; border-radius: 20px; font-size: 14px; font-weight: bold; backdrop-filter: blur(10px);">
            {batch_processed_count}/{batch_total_count} ({progress_percent:.1f}%)
        </div>
    """
    
    progress_content = f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 8px;">
            {progress_counter}
        </div>
        
        <div style="background: rgba(255,255,255,0.2); border-radius: 10px; height: 10px; margin-bottom: 8px; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);">
            <div style="background: linear-gradient(90deg, #4CAF50, #8BC34A, #CDDC39); height: 100%; width: {progress_percent}%; border-radius: 10px; transition: width 0.5s ease; box-shadow: 0 2px 8px rgba(76,175,80,0.4);"></div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 6px;">
            <div style="text-align: center; background: rgba(255,255,255,0.1); padding: 6px; border-radius: 6px; backdrop-filter: blur(5px);">
                <div style="font-size: 14px; font-weight: bold; color: #4CAF50;">⏱️ {avg_time:.1f}s</div>
                <div style="font-size: 9px; opacity: 0.9;">Avg per Image</div>
            </div>
            <div style="text-align: center; background: rgba(255,255,255,0.1); padding: 6px; border-radius: 6px; backdrop-filter: blur(5px);">
                <div style="font-size: 14px; font-weight: bold; color: #FF9800;">🔄 {remaining}</div>
                <div style="font-size: 9px; opacity: 0.9;">Remaining</div>
            </div>
            <div style="text-align: center; background: rgba(255,255,255,0.1); padding: 6px; border-radius: 6px; backdrop-filter: blur(5px);">
                <div style="font-size: 14px; font-weight: bold; color: #E91E63;">⏰ {eta_text}</div>
                <div style="font-size: 9px; opacity: 0.9;">ETA</div>
            </div>
            <div style="text-align: center; background: rgba(255,255,255,0.1); padding: 6px; border-radius: 6px; backdrop-filter: blur(5px);">
                <div style="font-size: 14px; font-weight: bold; color: #2196F3;">📊 {elapsed}</div>
                <div style="font-size: 9px; opacity: 0.9;">Elapsed</div>
            </div>
        </div>
        
        <div style="text-align: center; font-size: 10px; color: rgba(255,255,255,0.8); font-style: italic; background: rgba(0,0,0,0.2); padding: 4px; border-radius: 4px;">
            {batch_current_stage}
        </div>
        {f'<div style="text-align: center; font-size: 11px; color: rgba(255,255,255,0.9); margin-top: 4px; background: rgba(76,175,80,0.2); padding: 4px; border-radius: 4px; font-weight: bold;">📄 {batch_current_image_name}</div>' if batch_current_image_name else ''}
    """
    
    html = get_progress_display_html("🚀 BATCH PROCESSING", progress_content)
    
    return html

def show_batch_progress():
    """Function to show/update batch progress display"""
    if is_processing and batch_total_count > 0:
        html = get_batch_progress_html()
        return gr.update(visible=True, value=html)
    else:
        return gr.update(visible=False, value="")

def hide_batch_progress():
    """Function to hide batch progress display"""
    return gr.update(visible=False, value="")

def clear_output_message():
    """Simple function to clear any previous output message"""
    return "", hide_batch_progress()


def show_single_progress():
    """Function to show single image processing status"""
    if is_processing:
        single_content = """
            <div style="text-align: center; font-size: 12px; color: rgba(255,255,255,0.9); font-style: italic; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 4px;">
                🖼️ Processing single image...
            </div>
        """
        html = get_progress_display_html("⚡ SINGLE PROCESSING", single_content)
        return gr.update(visible=True, value=html)
    else:
        return gr.update(visible=False, value="")

def start_batch_with_progress(*element_values):
    """Start batch processing and show progress display"""
    # Show initial batch progress
    initial_content = """
        <div style="display: flex; justify-content: flex-end; margin-bottom: 8px;">
            <div style="background: rgba(255,255,255,0.25); color: white; padding: 8px 16px; border-radius: 20px; font-size: 14px; font-weight: bold; backdrop-filter: blur(10px);">
                0%
            </div>
        </div>
        
        <div style="background: rgba(255,255,255,0.2); border-radius: 10px; height: 10px; margin-bottom: 8px; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);">
            <div style="background: linear-gradient(90deg, #4CAF50, #8BC34A, #CDDC39); height: 100%; width: 0%; border-radius: 10px; transition: width 0.5s ease; box-shadow: 0 2px 8px rgba(76,175,80,0.4);"></div>
        </div>
        
        <div style="text-align: center; font-size: 12px; color: rgba(255,255,255,0.9); font-style: italic; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 4px;">
            🔄 Initializing batch processing...
        </div>
    """
    initial_html = get_progress_display_html("🚀 BATCH STARTING...", initial_content)
    
    # Start the actual batch processing
    result = start_batch_process(*element_values)
    
    # Return result and show progress display
    return result, gr.update(visible=True, value=initial_html)

def stop_batch_with_progress(progress=gr.Progress()):
    """Stop batch processing and hide progress display"""
    result = stop_batch_upscale(progress)
    return result, gr.update(visible=False, value="")



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


def update_start_time(src_file, upscale_size, max_megapixels, max_resolution, start_time):
    global video_start
    video_start = start_time
    target_res_text = update_target_resolution(src_file, upscale_size, max_megapixels, max_resolution)
    return gr.update(value=target_res_text, visible=True)


def update_end_time(src_file, upscale_size, max_megapixels, max_resolution, end_time):
    global video_end
    video_end = end_time
    target_res_text = update_target_resolution(src_file, upscale_size, max_megapixels, max_resolution)
    return gr.update(value=target_res_text, visible=True)


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
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", open_folder_path])
    else:  # Linux and other Unix-like
        subprocess.run(["xdg-open", open_folder_path])


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
    if os.path.exists(model_dir):
        output = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if
                  f.endswith('.safetensors') or f.endswith('.ckpt')]
    else:
        local_model_dir = os.path.join(os.path.dirname(__file__), args.ckpt_dir)
        if os.path.exists(local_model_dir):
            output = [os.path.join(local_model_dir, f) for f in os.listdir(local_model_dir) if
                      f.endswith('.safetensors') or f.endswith('.ckpt')]
    if os.path.exists(args.ckpt) and args.ckpt not in output:
        output.append(args.ckpt)
    else:
        if os.path.exists(os.path.join(os.path.dirname(__file__), args.ckpt)):
            output.append(os.path.join(os.path.dirname(__file__), args.ckpt))
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
    target_model = args.ckpt
    if os.path.basename(target_model) in models:
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


def load_model(selected_model, selected_checkpoint, weight_dtype, sampler='DPMPP2M', device='cpu', progress=gr.Progress()):
    global model, last_used_checkpoint

    # Determine the need for model loading or updating
    need_to_load_model = last_used_checkpoint is None or last_used_checkpoint != selected_checkpoint
    need_to_update_model = selected_model != (model.current_model if model else None)
    if need_to_update_model:
        del model
        model = None

    # Resolve checkpoint path
    checkpoint_paths = [
        selected_checkpoint,
        os.path.join(args.ckpt_dir, selected_checkpoint),
        os.path.join(os.path.dirname(__file__), args.ckpt_dir, selected_checkpoint)
    ]
    checkpoint_use = next((path for path in checkpoint_paths if os.path.exists(path)), None)
    if checkpoint_use is None:
        raise FileNotFoundError(f"Checkpoint {selected_checkpoint} not found.")

    # Check if we need to load a new model
    if need_to_load_model or model is None:
        torch.cuda.empty_cache()
        last_used_checkpoint = checkpoint_use
        model_cfg = "options/SUPIR_v0_tiled.yaml" if args.use_tile_vae else "options/SUPIR_v0.yaml"
        weight_dtype = 'fp16' if not bf16_supported else weight_dtype
        model = create_SUPIR_model(model_cfg, weight_dtype, supir_sign=selected_model[-1], device=device, ckpt=checkpoint_use,
                                   sampler=sampler)
        model.current_model = selected_model
     
        if args.use_tile_vae:
            model.init_tile_vae(encoder_tile_size=512, decoder_tile_size=64, use_fast=args.use_fast_tile)
        if progress is not None:
            progress(1, desc="SUPIR loaded.")


def load_llava():
    global llava_agent
    if llava_agent is None:
        llava_path = get_model('liuhaotian/llava-v1.5-7b')
        llava_agent = LLavaAgent(llava_path, device=LLaVA_device, load_8bit=args.load_8bit_llava,
                                 load_4bit=args.load_4bit_llava)


def unload_llava():
    global llava_agent
    if args.load_4bit_llava or args.load_8bit_llava:
        printt("Clearing LLaVA.")
        clear_llava()
        printt("LLaVA cleared.")
    else:
        printt("Unloading LLaVA.")
        llava_agent = llava_agent.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
        printt("LLaVA unloaded.")


def clear_llava():
    global llava_agent
    del llava_agent
    llava_agent = None
    gc.collect()
    torch.cuda.empty_cache()


def all_to_cpu_background():
    if args.dont_move_cpu:
        return
    global face_helper, model, llava_agent, auto_unload
    printt("Moving all to CPU")
    if face_helper is not None:
        face_helper = face_helper.to('cpu')
        printt("Face helper moved to CPU")
    if model is not None:
        model = model.to('cpu')
        model.move_to('cpu')
        printt("Model moved to CPU")
    if llava_agent is not None:
        if auto_unload:
            unload_llava()
    gc.collect()
    torch.cuda.empty_cache()
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


def update_inputs(input_file, upscale_amount, max_megapixels, max_resolution):
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
        target_res = update_target_resolution(input_file, upscale_amount, max_megapixels, max_resolution)
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
        target_res = update_target_resolution(input_file, upscale_amount, max_megapixels, max_resolution)
        res_output = gr.update(value=target_res, visible=target_res != "")
    elif input_file is None:
        file_input = gr.update(visible=True, value=None)
    return file_input, image_input, video_slider, res_output, video_start_time, video_end_time, video_current_time, video_fps, video_total_frames


def update_target_resolution(img, do_upscale, max_megapixels=0, max_resolution=0):
    global last_input_path, last_video_params
    
    # Convert inputs to proper types
    try:
        do_upscale = float(do_upscale)
        max_megapixels = float(max_megapixels) if max_megapixels else 0
        max_resolution = int(float(max_resolution)) if max_resolution else 0
    except (ValueError, TypeError):
        do_upscale = 1.0
        max_megapixels = 0
        max_resolution = 0
    
    if img is None:
        last_video_params = None
        last_input_path = None
        return ""
        
    try:
        if is_image(img):
            last_input_path = img
            last_video_params = None
            try:
                # Use the safe_open_image helper instead of direct Image.open
                with safe_open_image(img) as img_obj:
                    width, height = img_obj.size
                    width_org, height_org = img_obj.size
            except Exception as e:
                print(f"Failed to open image: {str(e)}")
                raise
                    
        elif is_video(img):
            if img == last_input_path:
                params = last_video_params
            else:
                last_input_path = img
                params = get_video_params(img)
                last_video_params = params
            width, height = params['width'], params['height']
            width_org, height_org = params['width'], params['height']
            print(f"Video dimensions: {width}x{height}")
        else:
            last_input_path = None
            last_video_params = None
            print(f"Invalid media type: {type(img)}")
            return ""

        # Convert width and height to float for calculations
        width = float(width)
        height = float(height)
        width_org = float(width_org)
        height_org = float(height_org)

        # Calculate aspect ratio for maintaining proportion
        aspect_ratio = width / height
        
        # Apply standard upscale factor first
        width *= do_upscale
        height *= do_upscale
        #print(f"After upscale: {width}x{height}")

        # Store dimensions before applying minimum constraints (for detection purposes)
        width_before_min = width
        height_before_min = height

        # Default minimal resolution check
        if min(width, height) < 1024:
            do_upscale_factor = 1024 / min(width, height)
            width *= do_upscale_factor
            height *= do_upscale_factor
            #print(f"After min size adjustment: {width}x{height}")
        
        # Apply max megapixels limit if specified
        if max_megapixels > 0:
            current_megapixels = width * height / 1_000_000
            #print(f"Current MP: {current_megapixels}, Max allowed: {max_megapixels}")
            if current_megapixels > max_megapixels:
                scale_factor = (max_megapixels * 1_000_000 / (width * height)) ** 0.5
                width *= scale_factor
                height *= scale_factor
                #print(f"After max MP adjustment: {width}x{height}")
                
                # Re-apply minimum resolution check after max megapixels constraint
                if min(width, height) < 1024:
                    min_scale_factor = 1024 / min(width, height)
                    width *= min_scale_factor
                    height *= min_scale_factor
        
        # Apply max resolution limit if specified
        if max_resolution > 0:
            #print(f"Max resolution: {max_resolution}, Current max dimension: {max(width, height)}")
            if max(width, height) > max_resolution:
                if width > height:
                    scale_factor = max_resolution / width
                else:
                    scale_factor = max_resolution / height
                width *= scale_factor
                height *= scale_factor
                #print(f"After max resolution adjustment: {width}x{height}")
                
                # Re-apply minimum resolution check after max resolution constraint
                if min(width, height) < 1024:
                    min_scale_factor = 1024 / min(width, height)
                    width *= min_scale_factor
                    height *= min_scale_factor

        # Round dimensions to multiples of 32 to match upscale_image function
        unit_resolution = 32
        width = int(np.round(width / unit_resolution)) * unit_resolution
        height = int(np.round(height / unit_resolution)) * unit_resolution
        #print(f"After unit resolution adjustment: {width}x{height}")

        output_lines = [
            f"<td style='padding: 8px; border-bottom: 1px solid #ddd;'>Input: {int(width_org)}x{int(height_org)} px, {width_org * height_org / 1e6:.2f} Megapixels</td>",
            f"<td style='padding: 8px; border-bottom: 1px solid #ddd;'>Estimated Output Resolution: {int(width)}x{int(height)} px, {width * height / 1e6:.2f} Megapixels</td>",
        ]

        # Add a note if minimum size enforcement had to override max resolution or max megapixels constraints
        resized_due_to_max_constraints = False
        
        # Check if max megapixels constraint would've made it smaller than 1024px
        if max_megapixels > 0:
            potential_mp_megapixels = (width_before_min * height_before_min) / 1_000_000
            if potential_mp_megapixels > max_megapixels:
                potential_scale = (max_megapixels * 1_000_000 / (width_before_min * height_before_min)) ** 0.5
                potential_w = width_before_min * potential_scale
                potential_h = height_before_min * potential_scale
                if min(potential_w, potential_h) < 1024:
                    resized_due_to_max_constraints = True
        
        # Check if max resolution constraint would've made it smaller than 1024px
        if max_resolution > 0:
            if max(width_before_min, height_before_min) > max_resolution:
                scale = max_resolution / max(width_before_min, height_before_min)
                potential_w = width_before_min * scale
                potential_h = height_before_min * scale
                if min(potential_w, potential_h) < 1024:
                    resized_due_to_max_constraints = True
        
        # if resized_due_to_max_constraints:
        #     output_lines.append(f"<td style='padding: 8px; border-bottom: 1px solid #ddd; color: #ff6600;'><b>Note:</b> Resolution was adjusted to maintain minimum 1024px for smallest dimension</td>")
        
        if total_video_frames > 0 and is_video(img):
            selected_video_frames = video_end - video_start
            total_video_time = int(selected_video_frames / current_video_fps)
            output_lines += [
                f"<td style='padding: 8px; border-bottom: 1px solid #ddd;'>Selected video frames: {selected_video_frames}</td>",
                f"<td style='padding: 8px; border-bottom: 1px solid #ddd;'>Total video time: {total_video_time} seconds</td>"
            ]

        output = '<table style="width:100%;"><tr>'
        # Convert each pair of <td> elements into a single string before joining
        output += ''.join(''.join(output_lines[i:i + 2]) for i in range(0, len(output_lines), 2))
        output += '</tr></table>'
        #print(f"Generated output HTML")
        return output
    except Exception as e:
        #print(f"Error in update_target_resolution: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"<p>Error calculating resolution: {str(e)}</p>"


def read_image_metadata(image_path):
    if image_path is None:
        return
    # Check if the file exists
    if not os.path.exists(image_path):
        return "File does not exist."

    # Get the last modified date and format it
    last_modified_timestamp = os.path.getmtime(image_path)
    last_modified_date = datetime.fromtimestamp(last_modified_timestamp).strftime('%d %B %Y, %H:%M %p - UTC')

    try:
        # Open the image and extract metadata using the safe helper
        with safe_open_image(image_path) as img:
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
    except Exception as e:
        return f"Error reading metadata: {str(e)}"


def update_elements(status_label):
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
            prompt_el = gr.update(value=caption)
            if len(image_data.outputs) > 0:
                outputs = image_data.outputs
                params = image_data.metadata_list
                if len(params) != len(outputs):
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
    with safe_open_image(temp_path.name) as img:
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
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_path.write(requests.get(
        "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1151ce9f4b2043de0d2e3b7826127998.jpg").content)
    temp_path.close()
    lowres_path = temp_path.name.replace('.jpg', '_lowres.jpg')
    with safe_open_image(temp_path.name) as img:
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
    global status_container, is_processing, batch_total_count, batch_processed_count, batch_processing_times, batch_current_stage
    
    # Force reset processing state to handle any stuck flags
    is_processing = False
    batch_total_count = 0
    batch_processed_count = 0
    batch_processing_times = []
    batch_current_stage = ""
    status_container = StatusContainer()
    values_dict = zip(elements_dict.keys(), element_values)
    values_dict = dict(values_dict)
    
    # Store ckpt_type in process_params but don't pass to batch_process
    status_container.process_params = status_container.process_params or {}
    status_container.process_params['ckpt_type'] = ckpt_type.value
    
    img_data = []
    validate_upscale = float(values_dict.get('upscale', 1)) > 1 or values_dict.get('apply_face', False) or values_dict.get('apply_bg', False) or values_dict.get('apply_face_only', False)

    input_image = values_dict['src_file']
    if input_image is None:
        return "No input image provided.", hide_batch_progress()
    
    # Clear any previous completion message by immediately showing processing status
    # This is a workaround since Gradio doesn't allow immediate UI updates during processing

    image_files = [input_image]

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
            media_data.caption = values_dict['main_prompt']
            img_data.append(media_data)
    else:
        for file in image_files:
            try:
                media_data = MediaData(media_path=file)
                img = safe_open_image(file)
                media_data.media_data = np.array(img)
                media_data.caption = values_dict['main_prompt']
                img_data.append(media_data)
            except:
                pass
    result = "An exception occurred. Please try again."
    # auto_unload_llava, batch_process_folder, main_prompt, output_video_format, output_video_quality, outputs_folder,video_duration, video_fps, video_height, video_width
    keys_to_pop = ['batch_process_folder', 'main_prompt', 'output_video_format',
                   'output_video_quality', 'outputs_folder', 'video_duration', 'video_end', 'video_fps',
                   'video_height', 'video_start', 'video_width', 'src_file', 'ckpt_type']

    values_dict['outputs_folder'] = args.outputs_folder
    status_container.process_params = values_dict
    values_dict = {k: v for k, v in values_dict.items() if k not in keys_to_pop}

    try:
        is_processing = True
        # Show single processing status
        single_progress_display = show_single_progress()
        
        # IMPORTANT: Return immediately with a starting message to clear previous completion messages
        # This creates immediate UI feedback when the button is clicked
        starting_result = "Processing started..."
        # Note: This will be overridden by the actual result when batch_process completes
        
        # Override skip_existing_images to False for single image processing
        # Users expect single images to always be processed
        values_dict['skip_existing_images'] = False
        _, result = batch_process(img_data, **values_dict)
        
        is_processing = False
    except Exception as e:
        print(f"An exception occurred: {e} at {traceback.format_exc()}")
        is_processing = False
    
    # Return result and hidden batch progress display for single processing  
    batch_progress_display = hide_batch_progress()
    return result, batch_progress_display



def start_batch_process(*element_values):
    global status_container, is_processing, batch_start_time, batch_processed_count, batch_total_count, batch_processing_times, batch_current_stage
    
    # Force reset processing state to handle any stuck flags
    is_processing = False
    batch_total_count = 0
    batch_processed_count = 0
    batch_processing_times = []
    batch_current_stage = ""
    status_container = StatusContainer()
    
    # Initialize batch progress tracking
    batch_start_time = None
    batch_processed_count = 0
    batch_total_count = 0
    batch_processing_times = []
    batch_current_stage = "Initializing..."
    values_dict = zip(elements_dict.keys(), element_values)
    values_dict = dict(values_dict)
    
    # Store ckpt_type in process_params but don't pass to batch_process
    status_container.process_params = status_container.process_params or {}
    status_container.process_params['ckpt_type'] = ckpt_type.value
    
    batch_folder = values_dict.get('batch_process_folder')
    if not batch_folder:
        return "No input folder provided.", hide_batch_progress()
    if not os.path.exists(batch_folder):
        return "The input folder does not exist.", hide_batch_progress()

    if len(values_dict['outputs_folder']) < 2:
        values_dict['outputs_folder'] = args.outputs_folder

    image_files = [file for file in os.listdir(batch_folder) if
                   is_image(os.path.join(batch_folder, file))]

    # Make a dictionary to store the image data and path
    img_data = []
    for file in image_files:
        media_data = MediaData(media_path=os.path.join(batch_folder, file))
        img = safe_open_image(os.path.join(batch_folder, file))
        media_data.media_data = np.array(img)
        media_data.caption = values_dict['main_prompt']
        img_data.append(media_data)

    # Store it globally
    status_container.image_data = img_data
    
    # Set total count for batch progress tracking
    batch_total_count = len(img_data)
    
    result = "An exception occurred. Please try again."
    try:
        keys_to_pop = ['batch_process_folder', 'main_prompt', 'output_video_format',
                       'output_video_quality', 'outputs_folder', 'video_duration', 'video_end', 'video_fps',
                       'video_height', 'video_start', 'video_width', 'src_file', 'ckpt_type']

        status_container.outputs_folder = values_dict['outputs_folder']
        values_dict['outputs_folder'] = values_dict['outputs_folder']
        status_container.process_params = values_dict

        values_dict = {k: v for k, v in values_dict.items() if k not in keys_to_pop}
        result, _ = batch_process(img_data, **values_dict)
    except Exception as e:
        print(f"An exception occurred: {e} at {traceback.format_exc()}")
        is_processing = False
    
    # Return both result status and batch progress display
    batch_progress_display = show_batch_progress()
    return result, batch_progress_display



def llava_process(inputs: List[MediaData], temp, p, question=None, save_captions=False, progress=gr.Progress(), skip_llava_if_txt_exists: bool = True):
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
        
        # Check if filename.txt exists - if it does, skip LLaVA processing for this image
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if skip_llava_if_txt_exists and os.path.exists(txt_path):
            printt(f"Found {txt_path}, skipping LLaVA for this image (override active)")
            # Read caption from the text file instead
            with open(txt_path, 'r') as f:
                caption = f.read().strip()
            md.caption = caption
            outputs.append(md)
            step += 1
            batch_info = f" [BATCH: {batch_processed_count}/{batch_total_count}]" if batch_total_count > 0 else ""
            progress(step / total_steps, desc=f"Skipped LLaVA for image {step}/{len(inputs)}, using existing text file{batch_info}")
            continue
            
        batch_info = f" [BATCH: {batch_processed_count}/{batch_total_count}]" if batch_total_count > 0 else ""
        progress(step / total_steps, desc=f"Processing image {step}/{len(inputs)} with LLaVA...{batch_info}")
        if img is None:  ## this is for llava and video
            img = safe_open_image(img_path)
            img = np.array(img)
        lq = HWC3(img)
        lq = Image.fromarray(lq.astype('uint8'))
        caption = llava_agent.gen_image_caption([lq], temperature=temp, top_p=p, qs=question)
        caption = caption[0]
        md.caption = caption
        outputs.append(md)
        if save_captions:
            cap_path = os.path.splitext(img_path)[0] + ".txt"
            with open(cap_path, 'w', encoding='utf-8') as cf:
                cf.write(caption)
        if not is_processing:  # Check if batch processing has been stopped
            break
        step += 1

    progress(step / total_steps, desc="LLaVA processing completed.")
    status_container.image_data = outputs
    return f"LLaVA Processing Completed: {len(inputs)} images processed at {time.ctime()}."


# video_start_time_number, video_current_time_number, video_end_time_number,
#                      video_fps_number, video_total_frames_number, src_input_file, upscale_slider
def update_video_slider(start_time, current_time, end_time, fps, total_frames, src_file, upscale_size, max_megapixels, max_resolution):
    print(f"Updating video slider: {start_time}, {current_time}, {end_time}, {fps}, {src_file}")
    global video_start, video_end
    video_start = start_time
    video_end = end_time
    video_frame = ui_helpers.get_video_frame(src_file, current_time)
    target_res_text = update_target_resolution(src_file, upscale_size, max_megapixels, max_resolution)
    return gr.update(value=video_frame), gr.update(value=target_res_text, visible=target_res_text != "")


def supir_process(inputs: List[MediaData], a_prompt, n_prompt, num_samples,
                  upscale, edm_steps,
                  s_stage1, s_stage2, s_cfg, seed, sampler, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype,
                  linear_cfg, linear_s_stage2, spt_linear_cfg, spt_linear_s_stage2, model_select,
                  ckpt_select, num_images, random_seed, apply_llava, face_resolution, apply_bg, apply_face,
                  face_prompt, max_megapixels, max_resolution, first_downscale, apply_face_only=False, dont_update_progress=False, unload=True,
                  progress=gr.Progress()):
    global model, status_container, event_id, batch_processed_count, batch_processing_times, batch_current_stage, batch_total_count, batch_start_time, batch_current_image_name
    main_begin_time = time.time()
    
    # Ensure all parameters are of the correct type
    try:
        num_samples = int(num_samples) if num_samples is not None else 1
        edm_steps = int(edm_steps) if edm_steps is not None else 50
        s_stage1 = float(s_stage1) if s_stage1 is not None else -1.0
        s_stage2 = float(s_stage2) if s_stage2 is not None else 1.0
        s_cfg = float(s_cfg) if s_cfg is not None else 3.0
        s_churn = float(s_churn) if s_churn is not None else 5.0
        s_noise = float(s_noise) if s_noise is not None else 1.003
        num_images = int(num_images) if num_images is not None else 1
        face_resolution = int(float(face_resolution)) if face_resolution is not None else 1024
        
        # Convert string booleans to actual booleans if needed
        if isinstance(random_seed, str):
            random_seed = random_seed.lower() == 'true'
        if isinstance(apply_llava, str):
            apply_llava = apply_llava.lower() == 'true'
        if isinstance(apply_bg, str):
            apply_bg = apply_bg.lower() == 'true'
        if isinstance(apply_face, str):
            apply_face = apply_face.lower() == 'true'
        if isinstance(apply_face_only, str):
            apply_face_only = apply_face_only.lower() == 'true'
        if isinstance(linear_cfg, str):
            linear_cfg = linear_cfg.lower() == 'true'
        if isinstance(linear_s_stage2, str):
            linear_s_stage2 = linear_s_stage2.lower() == 'true'
        if isinstance(first_downscale, str):
            first_downscale = first_downscale.lower() == 'true'
    except (ValueError, TypeError) as e:
        print(f"Error converting parameters: {e}")
        # Provide default values in case of conversion errors
        if not isinstance(num_samples, int):
            num_samples = 1
    
    total_images = len(inputs) * num_images
    total_progress = total_images + 1
    if unload:
        total_progress += 1
    counter = 0
    batch_info = f" [BATCH: {batch_processed_count}/{batch_total_count} images queued]" if batch_total_count > 0 else ""
    progress(counter / total_progress, desc=f"Loading SUPIR Model...{batch_info}")
    load_model(model_select, ckpt_select, diff_dtype, sampler, progress=progress)
    to_gpu(model, SUPIR_device)

    counter += 1
    batch_info = f" [BATCH: {batch_processed_count}/{batch_total_count} images ready]" if batch_total_count > 0 else ""
    progress(counter / total_progress, desc=f"Model Loaded, Processing Images...{batch_info}")
    model.ae_dtype = convert_dtype('fp32' if bf16_supported == False else ae_dtype)
    model.model.dtype = convert_dtype('fp16' if bf16_supported == False else diff_dtype)

    # Ensure max_megapixels and max_resolution are numbers
    max_megapixels = float(max_megapixels) if max_megapixels is not None else 0
    max_resolution = float(max_resolution) if max_resolution is not None else 0

    idx = 0
    output_data = []
    processed_images = 0
    params = status_container.process_params

    for image_data in inputs:
        # Track processing time for this image
        image_start_time = time.time()
        
        gen_params_list = []
        img_params = params.copy()
        img = image_data.media_data
        image_path = image_data.media_path
        
        # Update batch progress
        batch_current_stage = f"Processing image {batch_processed_count + 1}/{batch_total_count}"
        # Extract and update current image name
        image_name = os.path.basename(image_path) if image_path else "Unknown"
        batch_current_image_name = image_name
        
        # Show batch metrics in the progress description where it's always visible
        batch_info = ""
        if batch_total_count > 0:
            if len(batch_processing_times) > 0:
                avg_time = sum(batch_processing_times) / len(batch_processing_times)
                remaining = batch_total_count - batch_processed_count
                elapsed_time = time.time() - batch_start_time if batch_start_time else 0
                elapsed_mins = int(elapsed_time // 60)
                if remaining > 0:
                    eta_seconds = remaining * avg_time
                    eta_minutes = int(eta_seconds // 60)
                    batch_info = f" 🚀 BATCH: {batch_processed_count + 1}/{batch_total_count} | {remaining} left | Avg: {avg_time:.1f}s | ETA: {eta_minutes}m | Elapsed: {elapsed_mins}m"
                else:
                    batch_info = f" 🚀 BATCH: {batch_processed_count + 1}/{batch_total_count} | Almost done! | Elapsed: {elapsed_mins}m"
            else:
                batch_info = f" 🚀 BATCH: {batch_processed_count + 1}/{batch_total_count} | Starting..."
        
        progress(counter / total_progress, desc=f"Processing image {counter}/{total_images}...{batch_info}")
        if img is None:
            printt(f"Image {counter}/{total_images} is None, loading from disk.")
            with safe_open_image(image_path) as img:
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
        
        # Calculate final upscale factor based on constraints
        h, w, _ = img.shape
        target_h = float(h) * float(upscale)
        target_w = float(w) * float(upscale)
        
        # Apply minimum resolution
        if min(target_h, target_w) < 1024:
            min_scale = 1024 / min(target_h, target_w)
            target_h *= min_scale
            target_w *= min_scale
        
        # Apply max megapixels constraint if specified
        if max_megapixels > 0:
            target_mp = (target_h * target_w) / 1_000_000
            if target_mp > max_megapixels:
                mp_scale = (max_megapixels * 1_000_000 / (target_h * target_w)) ** 0.5
                target_h *= mp_scale
                target_w *= mp_scale
        
        # Apply max resolution constraint if specified
        if max_resolution > 0:
            if max(target_h, target_w) > max_resolution:
                if target_w > target_h:
                    res_scale = max_resolution / target_w
                else:
                    res_scale = max_resolution / target_h
                target_h *= res_scale
                target_w *= res_scale
        
        # Calculate final upscale factor
        final_upscale = min(target_h / h, target_w / w)
        
        # Apply first downscale logic if enabled
        if first_downscale:
            # Calculate the target resolution after constraints
            final_target_h = int(target_h)
            final_target_w = int(target_w)
            
            # First, downscale the image to match the target resolution divided by upscale factor
            # This ensures that when we upscale by the specified factor, we get the exact target resolution
            downscale_target_h = int(final_target_h / upscale)
            downscale_target_w = int(final_target_w / upscale)
            
            printt(f"First Downscale enabled: Downscaling to {downscale_target_w}x{downscale_target_h}, then upscaling {upscale}x to {final_target_w}x{final_target_h}")
            
            # Use PIL Image for high-quality Lanczos downscaling
            img_pil = Image.fromarray(img.astype('uint8'))
            img_pil_downscaled = img_pil.resize((downscale_target_w, downscale_target_h), Image.LANCZOS)
            img = np.array(img_pil_downscaled)
            
            # Now set the final upscale to exactly the specified upscale factor
            final_upscale = float(upscale)
        
        printt(f"Final upscale factor: {final_upscale:.2f}")
        
        img = upscale_image(img, final_upscale, unit_resolution=32, min_size=1024)

        lq = np.array(img)
        lq = lq / 255 * 2 - 1
        lq = torch.tensor(lq, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(SUPIR_device)[:, :3, :, :]

        _faces = []
        if not dont_update_progress and progress is not None:
            progress_value = counter / total_images if total_images > 0 else 1.0
            progress(progress_value, desc=f"Upscaling Images {counter}/{total_images}")
        face_captions = [img_prompt]

        if apply_face or apply_face_only:
            lq = np.array(img)
            load_face_helper()
            if face_helper is None or not isinstance(face_helper, FaceRestoreHelper):
                raise ValueError('Face helper not loaded')
            # <<< FIX: Update face_helper's upscale factor >>>
            if apply_face_only:
                printt(f"DEBUG: Setting face_helper upscale_factor to 1 for face-only mode")
                face_helper.upscale_factor = 1
            else:
                printt(f"DEBUG: Setting face_helper upscale_factor to {final_upscale}")
                face_helper.upscale_factor = final_upscale 
            # <<< END FIX >>>
            face_helper.clean_all()
            face_helper.read_image(lq)
            # get face landmarks for each face
            printt(f"DEBUG: Getting face landmarks for image shape {lq.shape}...")
            face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
            printt(f"DEBUG: Found {len(face_helper.all_landmarks_5)} faces. Aligning and warping...")
            face_helper.align_warp_face()
            printt(f"DEBUG: Alignment complete. Number of cropped faces: {len(face_helper.cropped_faces)}")
            if len(face_helper.cropped_faces) > 0:
                printt(f"DEBUG: First cropped face dimensions: {face_helper.cropped_faces[0].shape}")
            else:
                printt("DEBUG: No faces found or cropped.")


            lq = lq / 255 * 2 - 1
            lq = torch.tensor(lq, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(SUPIR_device)[:, :3,
                           :, :]

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
                # Check if processing was canceled
                global is_processing
                if not is_processing:
                    printt("Process cancelled before starting sample processing")
                    return None
                
                # Create a cancellation check function that will be passed to the model
                def is_cancelled():
                    return not is_processing
                
                samples = model.batchify_sample(input_data, caption, num_steps=edm_steps, restoration_scale=s_stage1,
                                            s_churn=s_churn, s_noise=s_noise, cfg_scale=s_cfg,
                                            control_scale=s_stage2, seed=seed,
                                            num_samples=num_samples, p_p=a_prompt, n_p=n_prompt,
                                            color_fix_type=color_fix_type,
                                            use_linear_cfg=linear_cfg, use_linear_control_scale=linear_s_stage2,
                                            cfg_scale_start=spt_linear_cfg, control_scale_start=spt_linear_s_stage2, 
                                            sampler_cls=sampler, is_cancelled=is_cancelled)

                if samples is None:  # Check if processing was cancelled during model execution
                    printt("Process cancelled during sample processing or sample processing returned None")
                    return None

                # Ensure face_resolution is a number before comparison
                if is_face and face_resolution is not None and int(face_resolution) < 1024:
                    face_resolution = int(face_resolution)
                    samples = samples[:, :, 512 - face_resolution // 2:512 + face_resolution // 2,
                              512 - face_resolution // 2:512 + face_resolution // 2]
                return samples

            if apply_face or apply_face_only:
                faces = []
                restored_faces = []
                for face in face_helper.cropped_faces:
                    restored_faces.append(face)
                    face = np.array(face) / 255 * 2 - 1
                    face = torch.tensor(face, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(SUPIR_device)[:, :3,
                           :, :]
                    faces.append(face)
                
                printt(f"DEBUG: Prepared {len(faces)} faces for upscaling.")

                for index, face in enumerate(faces):
                    progress(index / len(faces), desc=f"Upscaling Face {index}/{len(faces)}")
                    caption = face_captions[0]  # currently we dont have multiple captions for faces
                    gen_params['face_prompt'] = caption
                    printt(f"DEBUG: Upscaling face {index+1}/{len(faces)} with resolution {face_resolution}...")
                    samples = process_sample(model, face, [caption], face_resolution, is_face=True)
                    if samples is None:
                        printt(f"DEBUG: Face processing was cancelled for face {index+1}")
                        continue
                    
                    printt(f"DEBUG: Upscaled face {index+1} dimensions before interpolation: {samples.shape}")
                    # Interpolate samples
                    samples = torch.nn.functional.interpolate(samples, size=face_helper.face_size, mode='bilinear',
                                                              align_corners=False)
                    printt(f"DEBUG: Upscaled face {index+1} dimensions after interpolation: {samples.shape}")
                    
                    x_samples = (einops.rearrange(samples,
                                                  'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().round().clip(0,
                                                                                                                    255).astype(
                        np.uint8)
                    
                    printt(f"DEBUG: Adding restored face {index+1} with dimensions {x_samples[0].shape} to helper.")
                    face_helper.add_restored_face(x_samples[0])
                    restored_faces.append(x_samples[0])
                gen_params["face_gallery"] = restored_faces
                printt(f"DEBUG: Finished upscaling {len(face_helper.restored_faces)} faces.")

            # Look ma, we can do either now!
            if apply_bg:
                printt("DEBUG: Applying background restoration.")
                caption = [img_prompt]
                samples = process_sample(model, lq, caption)
                if samples is None:
                    printt("DEBUG: Background processing was cancelled")
                    break
                _bg = (einops.rearrange(samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().round().clip(0,
                                                                                                                   255).astype(
                    np.uint8)
                printt(f"DEBUG: Background restored. Dimensions: {_bg[0].shape}")
                if apply_face:
                    printt(f"DEBUG: Getting inverse affine transform before pasting faces onto BG. Restored faces count: {len(face_helper.restored_faces)}")
                    face_helper.get_inverse_affine(None)
                    printt(f"DEBUG: Pasting {len(face_helper.restored_faces)} faces onto upscaled background image with shape {_bg[0].shape}")
                    result = face_helper.paste_faces_to_input_image(upsample_img=_bg[0])
                    printt(f"DEBUG: Pasting complete. Final image shape: {result.shape if result is not None else 'None'}")
                else:
                    result = _bg[0]
                    printt("DEBUG: Using only restored background. Final image shape: {result.shape if result is not None else 'None'}")

            if apply_face_only:
                printt(f"DEBUG: Applying face-only restoration (no upscaling). Restored faces count: {len(face_helper.restored_faces)}")
                printt("DEBUG: Using original image resolution (upscale_factor already set to 1).")
                face_helper.get_inverse_affine(None)
                printt(f"DEBUG: Pasting {len(face_helper.restored_faces)} faces onto original image. Shape: {face_helper.input_img.shape if hasattr(face_helper, 'input_img') else 'N/A'}")
                result = face_helper.paste_faces_to_input_image()
                printt(f"DEBUG: Face-only restoration complete. Final image shape: {result.shape if result is not None else 'None'}")
                
            elif not apply_bg and apply_face:                
                printt(f"DEBUG: Applying only face restoration. Restored faces count: {len(face_helper.restored_faces)}")
                printt("DEBUG: Getting inverse affine transform before pasting faces onto original sized image.")
                # <<< FIX: Update face_helper's upscale factor (redundant here if done earlier, but safe) >>>
                printt(f"DEBUG: Ensuring face_helper upscale_factor is {final_upscale}")
                face_helper.upscale_factor = final_upscale 
                # <<< END FIX >>>
                face_helper.get_inverse_affine(None)
                # the image the face helper is using is already scaled to the desired resolution using lanzcos
                # I believe the output from this function should be just the original image but with only the face
                # restoration. 
                # Let's log the image the helper is using
                printt(f"DEBUG: Pasting {len(face_helper.restored_faces)} faces onto helper's internal image. Shape: {face_helper.input_img.shape if hasattr(face_helper, 'input_img') else 'N/A'}")
                result = face_helper.paste_faces_to_input_image()
                printt(f"DEBUG: Pasting complete (no BG). Final image shape: {result.shape if result is not None else 'None'}")

            elif not apply_face and not apply_bg and not apply_face_only:
                printt("DEBUG: Applying standard SUPIR upscale (no face/BG restoration).")
                caption = [img_prompt]
                print("Batchifying sample...")
                samples = process_sample(model, lq, caption)
                if samples is None:
                    printt("Sample processing was cancelled")
                    break
                x_samples = (
                        einops.rearrange(samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().round().clip(
                    0, 255).astype(np.uint8)
                result = x_samples[0]

            gen_params['caption'] = img_prompt
            image_generation_time = time.time() - start_time
            if result is not None:
                results.append(result)
                gen_params_list.append(gen_params)
            counter += 1
            
            # Update batch progress tracking
            image_total_time = time.time() - image_start_time
            batch_processing_times.append(image_total_time)
            batch_processed_count += 1
            
            # Add batch info to completion message
            batch_completion_info = ""
            if batch_total_count > 0:
                avg_time = sum(batch_processing_times) / len(batch_processing_times)
                remaining = batch_total_count - batch_processed_count
                elapsed_time = time.time() - batch_start_time if batch_start_time else 0
                elapsed_mins = int(elapsed_time // 60)
                
                if remaining > 0:
                    eta_seconds = remaining * avg_time
                    eta_minutes = int(eta_seconds // 60)
                    batch_completion_info = f" 🚀 BATCH: {batch_processed_count}/{batch_total_count} DONE | {remaining} left | Avg: {avg_time:.1f}s | ETA: {eta_minutes}m | Elapsed: {elapsed_mins}m"
                else:
                    batch_completion_info = f" 🚀 BATCH COMPLETE! All {batch_processed_count} images processed | Total: {elapsed_mins}m"
            
            desc = f"Image {counter}/{total_images} completed{batch_completion_info}"
            progress_value = counter / total_images if total_images > 0 else 1.0
            progress(progress_value, desc=desc)
            
            # Note: batch_metrics_html will show persistent progress info automatically
            # The metrics display is updated via the global variables we just modified

        # Update outputs
        image_data.outputs = results
        image_data.metadata_list = gen_params_list
        # save_image will return the image data with the outputs as paths
        image_data = save_image(image_data, status_container.is_video)

        # Append the image data to the output data after saving
        output_data.append(image_data)

        progress_value = counter / total_images if total_images > 0 else 1.0
        progress(progress_value, desc=f"Image {counter}/{total_images} processed.")
        processed_images = counter
        # Check if cancellation was requested
        if not is_processing:
            printt("Processing canceled, stopping further image processing.")
            break
        
        # Check if cancellation was requested before moving to the next image
        if not is_processing:
            printt("Processing canceled, stopping further image processing.")
            break

    # Now we update the status container
    status_container.image_data = output_data
    if not is_processing or unload:
        progress_value = counter / total_images if total_images > 0 else 1.0
        progress(progress_value, desc="Unloading SUPIR...")
        all_to_cpu()
        progress(progress_value, desc="SUPIR Unloaded.")
    main_end_time = time.time()
    global unique_counter
    unique_counter = unique_counter + 1
    return f"Image Upscaling Completed: processed {total_images} images at in {main_end_time - main_begin_time:.2f} seconds #{unique_counter}"


def batch_process(img_data,
                  a_prompt, ae_dtype, apply_bg, apply_face, apply_face_only, apply_llava, apply_supir, ckpt_select, color_fix_type,
                  diff_dtype, edm_steps, face_prompt, face_resolution, filename_prefix, filename_suffix, first_downscale, linear_CFG, linear_s_stage2,
                  make_comparison_video, model_select, n_prompt, num_images, num_samples, qs, random_seed,
                  s_cfg, s_churn, s_noise, s_stage1, s_stage2, sampler, save_captions, seed, spt_linear_CFG,
                  spt_linear_s_stage2, temperature, top_p, upscale, max_megapixels, max_resolution, auto_unload_llava, skip_llava_if_txt_exists, skip_existing_images, progress=gr.Progress()
                  ):
    global is_processing, llava_agent, model, status_container, batch_total_count, batch_start_time, batch_processed_count, batch_processing_times, batch_current_stage, batch_current_image_name
    print(f"batch_process called: is_processing = {is_processing}")
    
    # Safety check: if is_processing is True but we're being called from a fresh start,
    # it might be stuck from a previous crash - reset it
    if is_processing and batch_total_count == 0:
        print("Detected stuck is_processing flag - resetting")
        is_processing = False
    
    # Ensure key parameters are of the correct type before processing
    try:
        # Convert string values to appropriate types
        num_images = int(num_images) if isinstance(num_images, str) else num_images
        num_samples = int(num_samples) if isinstance(num_samples, str) else num_samples
        edm_steps = int(edm_steps) if isinstance(edm_steps, str) else edm_steps
        upscale = float(upscale) if isinstance(upscale, str) else upscale
        max_megapixels = float(max_megapixels) if isinstance(max_megapixels, str) else max_megapixels
        max_resolution = float(max_resolution) if isinstance(max_resolution, str) else max_resolution
        face_resolution = int(float(face_resolution)) if isinstance(face_resolution, (str, float)) else face_resolution
        
        # Convert string booleans to actual booleans
        if isinstance(apply_llava, str):
            apply_llava = apply_llava.lower() == 'true'
        if isinstance(apply_supir, str):
            apply_supir = apply_supir.lower() == 'true'
        if isinstance(apply_bg, str):
            apply_bg = apply_bg.lower() == 'true'
        if isinstance(apply_face, str):
            apply_face = apply_face.lower() == 'true'
        if isinstance(apply_face_only, str):
            apply_face_only = apply_face_only.lower() == 'true'
        if isinstance(random_seed, str):
            random_seed = random_seed.lower() == 'true'
        if isinstance(linear_CFG, str):
            linear_CFG = linear_CFG.lower() == 'true'
        if isinstance(linear_s_stage2, str):
            linear_s_stage2 = linear_s_stage2.lower() == 'true'
        if isinstance(make_comparison_video, str):
            make_comparison_video = make_comparison_video.lower() == 'true'
        if isinstance(save_captions, str):
            save_captions = save_captions.lower() == 'true'
        if isinstance(auto_unload_llava, str):
            auto_unload_llava = auto_unload_llava.lower() == 'true'
        if isinstance(first_downscale, str):
            first_downscale = first_downscale.lower() == 'true'
        if isinstance(skip_existing_images, str):
            skip_existing_images = skip_existing_images.lower() == 'true'
    except (ValueError, TypeError) as e:
        print(f"Error converting parameters in batch_process: {e}")
        # Continue with original values if conversion fails
    
    ckpt_select = get_ckpt_path(ckpt_select)
    tiled = "TiledRestore" if args.use_tile_vae else "Restore"
    sampler_cls = f"sgm.modules.diffusionmodules.sampling.{tiled}{sampler}Sampler"
    if not ckpt_select:
        msg = "No checkpoint selected. Please select a checkpoint to continue."
        return msg, msg
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
    
    # Initialize batch progress tracking
    batch_start_time = time.time()
    batch_processed_count = 0
    batch_total_count = len(img_data)
    batch_current_stage = "Starting batch processing..."
    
    # Get the list of image files in the folder
    total_images = len(img_data)

    # Convert num_images to integer to prevent type errors
    num_images = int(num_images) if isinstance(num_images, str) else num_images

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
    batch_info = f" 🚀 BATCH: {batch_total_count} images queued for processing" if batch_total_count > 0 else ""
    # Clear any previous completion message by showing current processing status
    progress(0, desc=f"Starting - Processing {total_images} images...{batch_info}")
    printt(f"Processing {total_images} images...", reset=True)
    if apply_llava:
        batch_current_stage = "Processing LLaVA (image captioning)..."
        printt('Processing LLaVA')
        last_result = llava_process(img_data, temperature, top_p, qs, save_captions, progress=progress, skip_llava_if_txt_exists=skip_llava_if_txt_exists)
        printt('LLaVA processing completed')

        if auto_unload_llava:
            unload_llava()
            printt('LLaVA unloaded')

        # Update the img_data from the captioner
        img_data = status_container.image_data
    
    # Filter out images that already exist in target folder if skip option is enabled
    if skip_existing_images and apply_supir:
        output_dir = params.get('outputs_folder', args.outputs_folder)
        filtered_img_data = []
        skipped_count = 0
        
        # Get prefix and suffix from params and sanitize them
        filename_prefix = sanitize_filename_part(params.get('filename_prefix', ''))
        filename_suffix = sanitize_filename_part(params.get('filename_suffix', ''))
        
        for image_data in img_data:
            image_path = image_data.media_path
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            if len(base_filename) > 250:
                base_filename = base_filename[:250]
            
            # Apply prefix and suffix to the target filename
            final_filename = f'{filename_prefix}{base_filename}{filename_suffix}'
            target_path = os.path.join(output_dir, f'{final_filename}.png')
            
            # Check if the target file already exists
            if os.path.exists(target_path):
                printt(f"Skipping {final_filename}.png - already exists in target folder")
                skipped_count += 1
            else:
                filtered_img_data.append(image_data)
        
        img_data = filtered_img_data
        status_container.image_data = img_data
        
        # Recalculate totals after filtering
        total_images = len(img_data)
        # Update batch_total_count to reflect the actual number of images to be processed
        batch_total_count = total_images
        total_supir_steps = total_images * num_images + 2 if apply_supir else 0
        total_steps = total_supir_steps + total_llava_steps + 1
        
        if skipped_count > 0:
            printt(f"Skipped {skipped_count} already processed images. Processing {total_images} remaining images.")
    
    # Check for cancellation
    if not is_processing and model is not None:
        progress(total_steps / total_steps, desc="Cancelling SUPIR...")
        all_to_cpu()
        return f"Batch Processing Completed: Cancelled at {time.ctime()}.", last_result
    counter += total_llava_steps
    if apply_supir:
        batch_current_stage = "Processing SUPIR (image upscaling)..."
        progress(counter / total_steps, desc="Processing images...")
        printt("Processing images (Stage 2)")
        # Ensure upscale is a float before passing it to supir_process
        upscale = float(upscale)
        # Ensure max_megapixels and max_resolution are floats
        max_megapixels = float(max_megapixels) if max_megapixels is not None else 0
        max_resolution = float(max_resolution) if max_resolution is not None else 0
        # Ensure face_resolution is an integer
        face_resolution = int(float(face_resolution)) if face_resolution is not None else 0
        last_result = supir_process(img_data, a_prompt, n_prompt, num_samples, upscale, edm_steps, s_stage1, s_stage2,
                                    s_cfg, seed, sampler_cls, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype,
                                    linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2, model_select,
                                    ckpt_select,
                                    num_images, random_seed, apply_llava, face_resolution, apply_bg, apply_face,
                                    face_prompt, max_megapixels, max_resolution, first_downscale, apply_face_only, unload=True, progress=progress)
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

    # Finalize batch progress tracking
    batch_current_stage = "Batch processing completed!"
    
    is_processing = False
    # Clear batch tracking variables to ensure progress display hides
    batch_total_count = 0
    batch_processed_count = 0
    batch_processing_times = []
    batch_current_image_name = ""
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

        # Ensure max_megapixels and max_resolution are in the metadata
        if 'max_megapixels' not in event_dict and 'max_megapixels' in params:
            event_dict['max_megapixels'] = params['max_megapixels']
        if 'max_resolution' not in event_dict and 'max_resolution' in params:
            event_dict['max_resolution'] = params['max_resolution']

        evt_id = event_dict.get('evt_id', str(time.time_ns()))

        # Get the base filename and extension
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        if len(base_filename) > 250:
            base_filename = base_filename[:250]
        
        # Apply prefix and suffix from params and sanitize them
        filename_prefix = sanitize_filename_part(params.get('filename_prefix', ''))
        filename_suffix = sanitize_filename_part(params.get('filename_suffix', ''))
        
        # Construct the filename with prefix and suffix
        final_filename = f'{filename_prefix}{base_filename}{filename_suffix}'

        img = Image.fromarray(result)

        if args.log_history and not is_video_frame:
            history_path = f'./history/{evt_id[:5]}/{evt_id[5:]}'
            os.makedirs(history_path, exist_ok=True)
            with open(os.path.join(history_path, 'logs.txt'), 'w') as f:
                f.write(str(event_dict))
            img.save(os.path.join(history_path, f'HQ_{i}.png'))

        save_path = os.path.join(output_dir, f'{final_filename}.png')

        # If processing video, just save the image
        if is_video_frame:
            save_path = image_path
            img.save(save_path, "PNG")
        else:
            index = 1
            while os.path.exists(save_path):
                save_path = os.path.join(output_dir, f'{final_filename}_{str(index).zfill(4)}.png')
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
                # Use the renamed key for caption instead of raw "caption"
                meta.add_text(rename_meta_key("caption"), caption)
                # This will save the caption to a file with the same name as the image
                if save_caption:
                    caption_path = f'{os.path.splitext(save_path)[0]}.txt'
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(caption)
            img.save(save_path, "PNG", pnginfo=meta)

            metadata_path = os.path.join(metadata_dir, f'{os.path.splitext(os.path.basename(save_path))[0]}.txt')
            with open(metadata_path, 'w', encoding='utf-8') as f:
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
    
    # Apply prefix and suffix from params and sanitize them
    filename_prefix = sanitize_filename_part(params.get('filename_prefix', ''))
    filename_suffix = sanitize_filename_part(params.get('filename_suffix', ''))
    final_filename = f'{filename_prefix}{base_filename}{filename_suffix}'
    
    save_path = os.path.join(output_dir, f'{final_filename}.mp4')
    index = 1
    while os.path.exists(save_path):
        save_path = os.path.join(output_dir, f'{final_filename}_{str(index).zfill(4)}.mp4')
        index += 1

    video_path = os.path.join(compare_videos_dir, f'{final_filename}.mp4')
    video_path = os.path.abspath(video_path)
    full_save_image_path = os.path.abspath(image_data.outputs[0])
    org_image_absolute_path = os.path.abspath(image_path)
    create_comparison_video(org_image_absolute_path, full_save_image_path, video_path, params)
    image_data.comparison_video = video_path
    return image_data


def stop_batch_upscale(progress=gr.Progress()):
    global is_processing
    is_processing = False  # Immediately set flag to cancel processing
    progress(1, desc="Cancelling processing... This will take effect immediately.")
    print('\n*** Cancel command received - processing will stop at the next checkpoint ***\n')
    # Hide batch progress when stopping
    batch_progress_display = hide_batch_progress()
    return "Processing cancelled. Please wait for current operations to stop...", batch_progress_display

def reset_processing_state():
    """Reset processing state in case it gets stuck"""
    global is_processing, batch_total_count, batch_processed_count, batch_processing_times, batch_current_stage, batch_start_time
    is_processing = False
    batch_total_count = 0
    batch_processed_count = 0
    batch_processing_times = []
    batch_current_stage = ""
    batch_start_time = None
    print("Processing state has been reset.")
    batch_progress_display = hide_batch_progress()
    return "Processing state reset successfully.", batch_progress_display


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


# Comment out toggle_full_preview function
"""
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
"""

def toggle_compare_elements(enable: bool) -> Tuple[gr.update, gr.update]:
    return gr.update(visible=enable), gr.update(visible=enable), gr.update(visible=enable)


def auto_enable_bg_restore(face_restore_enabled):
    """
    Automatically enable background restore when face restore is enabled.
    This merges the behavior so users don't need to manage both checkboxes.
    """
    return gr.update(value=face_restore_enabled)

def handle_face_only_toggle(face_only_enabled):
    """
    Handle mutual exclusivity between face-only and combined face+bg restoration.
    When face-only is enabled, disable face and bg restoration.
    """
    if face_only_enabled:
        return gr.update(value=False), gr.update(value=False)
    return gr.update(), gr.update()

def handle_face_restoration_toggle(face_enabled, face_only_enabled):
    """
    Handle face restoration toggle - disable face-only when face+bg is enabled.
    """
    if face_enabled and face_only_enabled:
        return gr.update(value=False)
    return gr.update()


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
compare_fullscreen_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'javascript', 'compare_fullscreen.js'))

with open(js_file) as f:
    js = f.read()

with open(no_slider_file) as f:
    no_slider = f.read()
    
with open(compare_fullscreen_file) as f:
    compare_fullscreen_js = f.read()

head = f"""
<style media="screen">{css}</style>
<style media="screen">{slider_css}</style>
<style media="screen">
/* Force batch progress display to stay on top of ALL Gradio elements */
#batch_progress_display {{
    position: fixed !important;
    z-index: 2147483647 !important;  /* Maximum possible z-index value */
    pointer-events: none !important;
    opacity: 1 !important;
    visibility: visible !important;
    filter: none !important;
    backdrop-filter: none !important;
}}

/* Override any Gradio stacking contexts that might interfere */
.gradio-container, .gradio-container * {{
    z-index: auto !important;
}}

/* Prevent Gradio overlays from affecting our display */
.gradio-overlay, .gr-overlay, .loading-overlay, .progress-overlay {{
    z-index: 2147483646 !important;
}}

/* Prevent any dimming effects on our status display */
#batch_progress_display * {{
    opacity: 1 !important;
    visibility: visible !important;
    filter: none !important;
}}

/* Override any global overlay or dimming effects */
body.gradio-loading #batch_progress_display,
.gradio-container.loading #batch_progress_display {{
    opacity: 1 !important;
    visibility: visible !important;
}}
</style>
<script type="text/javascript">{js}</script>
<script type="text/javascript">{no_slider}</script>
<script type="text/javascript">{compare_fullscreen_js}</script>
<script>
// Modified downloadImage function to accept sliderId
function downloadImage(sliderId) {{
    console.log("Attempting download for slider:", sliderId);
    const slider = document.getElementById(sliderId);
    if (!slider) {{
        console.error("Download failed: Slider element not found:", sliderId);
        return;
    }}
    // Try common structures for the *second* image (usually the processed one)
    let img = slider.querySelector('.noUi-origin[style*="z-index: 5"] img'); // ImageSlider often uses this for the top image
    if (!img) img = slider.querySelector('.noUi-base > div:nth-of-type(2) img'); // Alternative structure
    if (!img) img = slider.querySelector('.image-zoom-wrapper > img:nth-of-type(2)'); // If using custom wrapper
    if (!img) img = slider.querySelector('img:nth-of-type(2)'); // Generic fallback

    if (img && img.src) {{
        console.log("Found image source:", img.src);
        const link = document.createElement('a');
        let filename = 'downloaded_image.png';
        try {{
             // Basic filename extraction from URL
             const urlParts = img.src.split('/');
             const potentialFilename = urlParts[urlParts.length - 1];
             // Basic check to avoid data URIs or weird names
             if (potentialFilename && potentialFilename.includes('.')) {{
                 filename = potentialFilename.split('?')[0]; // Remove query params if any
             }}
        }} catch (e) {{ console.error("Error parsing img src for filename", e); }}

        link.href = img.src;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        console.log("Download initiated for:", filename);
    }} else {{
        console.error('Could not find image source to download for slider:', sliderId, "Tried selectors, found:", img);
        alert("Could not find the image source to download.");
    }}
}}

// Batch progress polling system
function startBatchProgressPolling() {{
    setInterval(function() {{
        const updateBtn = document.getElementById('update_batch_progress_btn');
        if (updateBtn) {{
            updateBtn.click();
        }}
    }}, 2000); // Poll every 2 seconds
}}


// Start polling when page loads
document.addEventListener('DOMContentLoaded', function() {{
    startBatchProgressPolling();
}});

// Also start if DOM is already loaded
if (document.readyState !== 'loading') {{
    startBatchProgressPolling();
}}
</script>
"""

refresh_symbol = "\U000027F3"  # ⟳
dl_symbol = "\U00002B73"  # ⭳
fullscreen_symbol = "⛶"  # Simple fullscreen arrow that works across fonts


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
    # START CHANGE: Move the batch progress HTML component to the top level.
    # This escapes any nested CSS stacking contexts from tabs or columns,
    # ensuring its high z-index is respected and it appears over Gradio's overlay.
    batch_progress_html = gr.HTML(
        value="",
        visible=False,  # Start hidden. Processing functions will make it visible.
        show_label=False,
        elem_id="batch_progress_display"
    )
    
    # END CHANGE

    gr.Markdown("SUPIR V97 - https://www.patreon.com/posts/99176057")
    
    def do_nothing():
        pass
        
    with gr.Tab("Upscale"):
        # Execution buttons
        with gr.Column(scale=1):
            with gr.Row():
                start_single_button = gr.Button(value="Process Single")
                start_batch_button = gr.Button(value="Process Batch")
                stop_batch_button = gr.Button(value="Cancel")
                reset_state_button = gr.Button(value="Reset State", variant="secondary")
                btn_open_outputs = gr.Button("Open Outputs Folder")
                btn_open_outputs.click(fn=open_folder)

        with gr.Column(scale=1):
            with gr.Row():
                output_label = gr.Label(label="Progress", elem_classes=["progress_label"])
            with gr.Row():
                # START CHANGE: The gr.HTML component is now defined at the top level.
                # We only need the hidden button for polling here.
                update_batch_progress_button = gr.Button(visible=False, elem_id="update_batch_progress_btn")
                # END CHANGE
            with gr.Row():
                target_res_textbox = gr.HTML(value="", visible=True, show_label=False)
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
                        upscale_slider = gr.Slider(label="Upscale Size", minimum=1, maximum=20, value=1, step=0.1,
                                                  info="Base upscale factor. Image will be scaled by this amount, then constrained by Max Megapixels and Max Resolution if set. Set like 20x upscale and limit resolution with Max Megapixels and Max Resolution if you need.")
                        first_downscale_checkbox = gr.Checkbox(label="First Downscale to Match Target Upscale Size", value=False,
                                                             info="When enabled, the image is first downscaled using high-quality Lanczos algorithm to match the target resolution (after max megapixel/resolution constraints), then upscaled by the specified factor. This can help achieve more consistent results when working with size-constrained outputs.")
                    with gr.Row():
                        max_mp_slider = gr.Slider(label="Max Megapixels (0 = no limit)", minimum=0, maximum=100, value=0, step=1,
                                                 info="Limit output megapixels. Output will be constrained by Max Megapixels and Max Resolution if set. Note: Minimum dimension will always be at least 1024px.",
                                                 interactive=True)
                        max_res_slider = gr.Slider(label="Max Resolution (0 = no limit)", minimum=0, maximum=8192, value=0, step=64,
                                                  info="Limit maximum resolution (width or height). Note: Minimum dimension will always be at least 1024px.",
                                                  interactive=True)
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

                    prompt_textbox = gr.Textbox(label="Prompt", value="", lines=4)
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
                    skip_llava_if_txt_exists_checkbox = gr.Checkbox(label="Skip LLaVA if .txt caption exists", value=True) # Default to True to maintain original behavior
                with gr.Accordion("Comparison Video options", open=False):
                    with gr.Row():
                        output_vq_slider = gr.Slider(label="Output Video Quality", minimum=0.1, maximum=1.0, value=0.6,
                                                     step=0.1)
                        output_vf_dropdown = gr.Dropdown(label="Video Format", choices=["mp4", "mkv"], value="mp4")
                    with gr.Row():
                        make_comparison_video_checkbox = gr.Checkbox(
                            label="Generate Comparison Video", value=False)
                    with gr.Row(visible=True) as compare_video_row:
                        video_duration_textbox = gr.Textbox(label="Duration", value="5")
                        video_fps_textbox = gr.Textbox(label="FPS", value="30")
                        video_width_textbox = gr.Textbox(label="Width", value="1920")
                        video_height_textbox = gr.Textbox(label="Height", value="1080")


            with gr.Column():
                with gr.Accordion("Batch options", open=True):
                    with gr.Row():
                        # Add Open Outputs Folder button
                        btn_open_outputs = gr.Button("Open Outputs Folder")
                        btn_open_outputs.click(fn=open_folder)
                    with gr.Row():
                        filename_prefix_textbox = gr.Textbox(
                            label="Filename Prefix",
                            placeholder="e.g., upscaled_",
                            value="")
                        filename_suffix_textbox = gr.Textbox(
                            label="Filename Suffix (before extension)",
                            placeholder="e.g., _hq",
                            value="")
                    with gr.Row():
                        with gr.Column():
                            batch_process_folder_textbox = gr.Textbox(
                                label="Batch Input Folder - Can use image captions from .txt",
                                placeholder="R:\SUPIR video\comparison_images")
                            outputs_folder_textbox = gr.Textbox(
                                label="Batch Output Path - Leave empty to save to default.",
                                placeholder="R:\SUPIR video\comparison_images\outputs")
                            with gr.Row():
                                save_captions_checkbox = gr.Checkbox(label="Save Captions",
                                                                     value=True)
                                skip_existing_images_checkbox = gr.Checkbox(label="Skip already upscaled images in target folder", value=True) # Default to True to avoid re-processing

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
                        s_cfg_slider = gr.Slider(label="Text Guidance Scale", minimum=1.0, maximum=15.0, value=3.0,
                                                 step=0.1)
                        s_stage2_slider = gr.Slider(label="Stage2 Guidance Strength", minimum=0., maximum=2., value=1.,
                                                    step=0.05)
                        s_stage1_slider = gr.Slider(label="Stage1 Guidance Strength", minimum=-1.0, maximum=6.0,
                                                    value=-1.0,
                                                    step=1.0)
                        seed_slider = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                        sampler_dropdown = gr.Dropdown(label="Sampler", choices=["EDM", "DPMPP2M"],
                                                       value="EDM")
                        s_churn_slider = gr.Slider(label="S-Churn", minimum=0, maximum=40, value=5, step=1)
                        s_noise_slider = gr.Slider(label="S-Noise", minimum=1.0, maximum=1.1, value=1.003, step=0.001)

                    with gr.Row():
                        a_prompt_textbox = gr.Textbox(label="Default Positive Prompt",
                                                      value=selected_pos)
                        n_prompt_textbox = gr.Textbox(label="Default Negative Prompt",
                                                      value=selected_neg)
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
                            apply_face_checkbox = gr.Checkbox(label="Face restoration", value=False)
                        with gr.Column():
                            apply_face_only_checkbox = gr.Checkbox(label="Only Face Restoration", value=False)
                    apply_bg_checkbox = gr.Checkbox(label="BG restoration", value=False, visible=False)


                with gr.Accordion("Presets", open=True):
                    presets_dir = os.path.join(os.path.dirname(__file__), 'presets')
                    if not os.path.exists(presets_dir):
                        os.makedirs(presets_dir)
                    
                    # Last used preset file path
                    last_used_preset_file = os.path.join(presets_dir, 'last_used_preset.json')

                    def save_preset(preset_name, config):
                    #Save the current configuration to a file.
                        file_path = os.path.join(presets_dir, f"{preset_name}.json")
                        with open(file_path, 'w') as f:
                            json.dump(config, f)
                        
                        # Save as last used preset
                        with open(last_used_preset_file, 'w') as f:
                            json.dump(preset_name, f)
                            
                        return "Preset saved successfully!"

                    def load_preset(preset_name):
                        global elements_dict, extra_info_elements
                        file_path = os.path.join(presets_dir, f"{preset_name}.json")
                        if not os.path.exists(file_path):
                            print("Don't forget to select a valid preset file")
                            return ["Error"] + [gr.update() for _ in range(len(elements_dict) + len(extra_info_elements))]
                        try:
                            with open(file_path, 'r') as f:
                                # Load the JSON string from the file
                                json_string = json.load(f)
                                # Decode the JSON string into a dictionary
                                config = json.loads(json_string)
                        except Exception as e:
                            print(f"Error loading preset: {str(e)}")
                            return [f"Error loading preset: {str(e)}"] + [gr.update() for _ in range(len(elements_dict) + len(extra_info_elements))]

                        # Create default updates (no change) for all elements
                        all_updates = []
                        # Clear the message after a short delay instead of keeping it permanent
                        all_updates.append(gr.update(value=""))  # First update is the output message
                        
                        # Add updates for elements_dict
                        for _ in elements_dict:
                            all_updates.append(gr.update())
                        
                        # Add updates for extra_info_elements
                        for _ in extra_info_elements:
                            all_updates.append(gr.update())
                        
                        # Apply config values to elements in elements_dict
                        for key, value in config.items():
                            try:
                                if key in elements_dict:
                                    if key == "src_file":
                                        # Skip updating the source file
                                        continue
                                        
                                    # Update the value in the elements_dict
                                    elements_dict[key].value = value
                                    # Get the index of the element
                                    index = list(elements_dict.keys()).index(key)
                                    # Update the corresponding update object
                                    all_updates[index+1] = gr.update(value=value)
                                elif key in extra_info_elements:
                                    # Update extra info elements
                                    if value is not None:
                                        extra_info_elements[key].value = value
                                        # Calculate index (after elements_dict entries)
                                        index = len(elements_dict) + list(extra_info_elements.keys()).index(key)
                                        all_updates[index+1] = gr.update(value=value)
                            except Exception as e:
                                print(f"Error updating element {key}: {str(e)}")
                        
                        # Save as last used preset
                        with open(last_used_preset_file, 'w') as f:
                            json.dump(preset_name, f)

                        # Make sure we're returning exactly the expected number of outputs
                        expected_outputs = 1 + len(elements_dict) + len(extra_info_elements)
                        if len(all_updates) < expected_outputs:
                            # Add any missing updates if needed
                            all_updates.extend([gr.update() for _ in range(expected_outputs - len(all_updates))])
                        elif len(all_updates) > expected_outputs:
                            # Trim if we somehow have too many
                            all_updates = all_updates[:expected_outputs]
                            
                        return all_updates

                    def get_preset_list():
                        """List all saved presets."""
                        return [f.replace('.json', '') for f in os.listdir(presets_dir) if f.endswith('.json')]

                    def serialize_settings(ui_elements):
                        global elements_dict, extra_info_elements
                        serialized_dict = {}

                        # Process elements in elements_dict
                        last_index = 0
                        for e_key, element_index in zip(elements_dict.keys(), range(len(ui_elements))):
                            element = ui_elements[element_index]
                            last_index = element_index 
                            # Check if element should be excluded from config saving
                            if getattr(element, 'do_not_save_to_config', False):
                                continue
                            # Check if the element has a 'value' attribute, otherwise use it directly
                            if hasattr(element, 'value'):
                                serialized_dict[e_key] = element.value
                            else:
                                serialized_dict[e_key] = element

                        # Process extra elements in extra_info_elements
                        last_index=last_index+1
                        for extra_key, extra_element in extra_info_elements.items():
                            # Check if the extra element has a 'value' attribute, otherwise use directly
                            element = ui_elements[last_index]
                            last_index=last_index+1
                            # Check if element should be excluded from config saving
                            if getattr(element, 'do_not_save_to_config', False):
                                continue
                            if hasattr(element, 'value'):
                                serialized_dict[extra_key] = element.value
                            else:
                                serialized_dict[extra_key] = element

                        # Convert dictionary to JSON string
                        json_settings = json.dumps(serialized_dict)
                        return json_settings

                    def save_current_preset(preset_name,*elements):
                        if preset_name:
                            preset_path = os.path.join(presets_dir, f"{preset_name}.json")
                            settings = serialize_settings(elements)
                            with open(preset_path, 'w') as f:
                                json.dump(settings, f)
                                
                            # Save as last used preset
                            with open(last_used_preset_file, 'w') as f:
                                json.dump(preset_name, f)
                                
                            # Return message and updated dropdown
                            presets = list_presets()
                            return f"Preset {preset_name} saved successfully!", gr.update(choices=presets, value=preset_name)
                        return "Please provide a valid preset name.", gr.update()

                    def list_presets():
                        presets = [file.split('.')[0] for file in os.listdir(presets_dir) if file.endswith('.json') and file != 'last_used_preset.json']
                        return presets

                    # Function to auto-load the last used preset
                    def auto_load_last_preset():
                        try:
                            if os.path.exists(last_used_preset_file):
                                with open(last_used_preset_file, 'r') as f:
                                    last_preset = json.load(f)
                                if last_preset and os.path.exists(os.path.join(presets_dir, f"{last_preset}.json")):
                                    print(f"Auto-loading last used preset: {last_preset}")
                                    # First update the dropdown
                                    dropdown_update = gr.update(value=last_preset)
                                    
                                    # Then load the preset content
                                    preset_updates = load_preset(last_preset)
                                    
                                    # Return only what this function needs to return
                                    return dropdown_update, preset_updates[0]
                        except Exception as e:
                            print(f"Error auto-loading last used preset: {str(e)}")
                        return gr.update(), ""

                    with gr.Row():
                        preset_name_textbox = gr.Textbox(label="Preset Name")
                        save_preset_button = gr.Button("Save Current Preset")
                        load_preset_dropdown = gr.Dropdown(label="Load Preset", choices=list_presets())
                        load_preset_button = gr.Button("Load Preset")
                        refresh_presets_button = gr.Button("Refresh Presets")

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
                apply_metadata_button = gr.Button("Apply Metadata")  # Add this line
                meta_image = gr.Image(type="filepath", label="Output Image", elem_id="output_image", visible=True, height="42.5vh", sources=["upload"])
                meta_video = gr.Video(label="Output Video", elem_id="output_video", visible=False, height="42.5vh")
                metadata_output = gr.Textbox(label="Image Metadata", lines=25, max_lines=50, elem_id="output_metadata")

    with gr.Tab("Compare Images", elem_id="compare_tab"):
        with gr.Column():
            compare_status = gr.HTML(value="Select one image from each gallery below", elem_id="compare_status")
            with gr.Row():
                refresh_compare_btn = gr.Button(value=f"{refresh_symbol} Refresh Image List", elem_id="refresh_compare_btn")
                toggle_fullscreen_btn = gr.Button(value=f"{fullscreen_symbol} Toggle Fullscreen", elem_id="toggle_fullscreen_compare_btn")
            
            with gr.Row():
                # Two separate galleries side by side
                with gr.Column(scale=1):
                    gr.Markdown("### Select First Image")
                    compare_gallery1 = gr.Gallery(
                        value=get_recent_images(20),
                        label="Image 1",
                        elem_id="compare_gallery1",
                        columns=4,
                        rows=2,
                        height=300,
                        object_fit="contain"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### Select Second Image")
                    compare_gallery2 = gr.Gallery(
                        value=get_recent_images(20),
                        label="Image 2", 
                        elem_id="compare_gallery2",
                        columns=4,
                        rows=2,
                        height=300,
                        object_fit="contain"
                    )
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Upload option for first image
                    compare_img1_upload = gr.Image(
                        label="Or upload image 1", 
                        type="filepath",
                        elem_id="compare_img1_upload",
                        height=300
                    )
                    image1_path = gr.Textbox(value="", visible=False) # Hidden storage for selected image 1 path
                
                with gr.Column(scale=1):
                    # Upload option for second image
                    compare_img2_upload = gr.Image(
                        label="Or upload image 2", 
                        type="filepath",
                        elem_id="compare_img2_upload",
                        height=300
                    )
                    image2_path = gr.Textbox(value="", visible=False) # Hidden storage for selected image 2 path
            
            with gr.Row():
                compare_btn = gr.Button(value="Compare Selected Images", elem_id="compare_btn")
                compare_fullscreen_btn = gr.Button(value="Toggle Fullscreen", elem_id="test_fullscreen_btn")
            
            with gr.Column(elem_classes=['preview_col'], elem_id="compare_preview_column") as compare_result_col:
                compare_result_gallery = gr.Gallery(label='Comparison Results', elem_id="compare_gallery2", elem_classes=["preview_box"],
                                            height=400, visible=False, rows=2, columns=4, allow_preview=True,
                                            show_download_button=False, show_share_button=False)
                compare_slider = ImageSlider(label='Image Comparison', interactive=False, show_download_button=True,
                                            elem_id="compare_slider",
                                            elem_classes=["preview_box", "preview_slider", "active"],
                                            height=500, container=True, visible=False)
                compare_result_video = gr.Video(label="Comparison Video", elem_classes=["preview_box"], height=400, visible=False)
                compare_slider_dl_button = gr.Button(value=dl_symbol, elem_classes=["slider_button"], visible=True,
                                                 elem_id="compare_download_button")
                compare_slider_full_button = gr.Button(value=fullscreen_symbol, elem_classes=["slider_button"], visible=True,
                                                    elem_id="compare_fullscreen_button")

    with gr.Tab("Download Checkpoints"):
        gr.Markdown("## Download Checkpoints")
        with gr.Row():
            with gr.Column():
                model_choice = gr.Dropdown(
                    choices=["SDXL 1.0 Base", "RealVisXL_V4", "Animagine XL V3.1", "Juggernaut XL V10"],
                    label="Select Model"
                )
            with gr.Column():      
                download_button = gr.Button("Download")
                download_output = gr.Textbox(label="Download Status")

        model_download_dir = gr.Textbox(value=args.ckpt_dir, visible=False)  # Invisible Textbox
        download_button.click(
            fn=download_checkpoint_handler,
            inputs=[model_choice, model_download_dir],
            outputs=download_output
        )


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
        "apply_face_only": apply_face_only_checkbox,
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
        "filename_prefix": filename_prefix_textbox,
        "filename_suffix": filename_suffix_textbox,
        "first_downscale": first_downscale_checkbox,
        "linear_CFG": linear_cfg_checkbox,
        "linear_s_stage2": linear_s_stage2_checkbox,
        "main_prompt": prompt_textbox,
        "make_comparison_video": make_comparison_video_checkbox,
        "max_megapixels": max_mp_slider,
        "max_resolution": max_res_slider,
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
        "skip_llava_if_txt_exists": skip_llava_if_txt_exists_checkbox, # Added new checkbox
        "skip_existing_images": skip_existing_images_checkbox, # Added skip existing images checkbox
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

    elements_extra = list(extra_info_elements.values())

    # Batch progress update handler (triggered by JavaScript)
    update_batch_progress_button.click(fn=show_batch_progress, outputs=batch_progress_html, 
                                      show_progress=False, queue=False)
    

    # Clear previous messages first, then start processing
    start_single_button.click(fn=clear_output_message, outputs=[output_label, batch_progress_html],
                              show_progress=False, queue=True).then(
        fn=start_single_process, inputs=elements, outputs=[output_label, batch_progress_html],
        show_progress=True, queue=True)
    
    start_batch_button.click(fn=clear_output_message, outputs=[output_label, batch_progress_html],
                             show_progress=False, queue=True).then(
        fn=start_batch_process, inputs=elements, outputs=[output_label, batch_progress_html],
        show_progress=True, queue=True)
    stop_batch_button.click(fn=stop_batch_upscale, outputs=[output_label, batch_progress_html], show_progress=True, queue=True)
    reset_state_button.click(fn=reset_processing_state, outputs=[output_label, batch_progress_html], show_progress=False, queue=False)
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
    
    # Auto-enable background restore when face restore is enabled
    apply_face_checkbox.change(fn=auto_enable_bg_restore, inputs=[apply_face_checkbox], outputs=[apply_bg_checkbox])
    
    # Handle mutual exclusivity between face restoration modes
    apply_face_only_checkbox.change(fn=handle_face_only_toggle, inputs=[apply_face_only_checkbox], outputs=[apply_face_checkbox, apply_bg_checkbox])
    apply_face_checkbox.change(fn=handle_face_restoration_toggle, inputs=[apply_face_checkbox, apply_face_only_checkbox], outputs=[apply_face_only_checkbox])
    
    submit_button.click(fn=submit_feedback, inputs=[event_id, fb_score, fb_text], outputs=[fb_text])
    upscale_slider.change(fn=update_target_resolution, inputs=[src_image_display, upscale_slider, max_mp_slider, max_res_slider],
                          outputs=[target_res_textbox])
    first_downscale_checkbox.change(fn=update_target_resolution, inputs=[src_image_display, upscale_slider, max_mp_slider, max_res_slider],
                                   outputs=[target_res_textbox])
    
    # Remove previous handlers that aren't working
    # Add new handlers with explicit update logic
    max_mp_slider.change(fn=update_target_resolution, 
                         inputs=[src_image_display, upscale_slider, max_mp_slider, max_res_slider],
                         outputs=[target_res_textbox])
    max_res_slider.change(fn=update_target_resolution, 
                         inputs=[src_image_display, upscale_slider, max_mp_slider, max_res_slider],
                         outputs=[target_res_textbox])

    # slider_dl_button.click(fn=download_slider_image, inputs=[result_slider], show_progress=False, queue=True)
    slider_full_button.click(
        fn=None, # No Python function needed, just JS
        inputs=None,
        outputs=None,
        show_progress=False,
        queue=False, # Can be false as it's pure JS
        # Call the JS function with IDs for upscale tab elements
        js="() => toggleSliderFullscreen('gallery1', 'preview_column', 'fullscreen_button', 'download_button')"
    )
    slider_dl_button.click(
        js="() => downloadImage('gallery1')",
        show_progress=False, 
        queue=True, 
        fn=do_nothing
    )

    input_elements = [src_input_file, src_image_display, video_slider_display, target_res_textbox,
                      video_start_time_number, video_end_time_number, video_current_time_number, video_fps_number,
                      video_total_frames_number]
    src_input_file.change(fn=update_inputs, inputs=[src_input_file, upscale_slider, max_mp_slider, max_res_slider],
                          outputs=input_elements)
    src_image_display.clear(fn=update_inputs, inputs=[src_image_display, upscale_slider, max_mp_slider, max_res_slider],
                            outputs=input_elements)

    model_settings_elements = [s_cfg_slider, spt_linear_cfg_slider, edm_steps_slider]

    ckpt_type.change(fn=update_model_settings, inputs=[ckpt_type, param_setting_select],
                     outputs=model_settings_elements)

    video_sliders = [video_start_time_number, video_current_time_number, video_end_time_number,
                     video_fps_number, video_total_frames_number, src_input_file, upscale_slider, max_mp_slider, max_res_slider]
    video_current_time_number.change(fn=update_video_slider, inputs=video_sliders,
                                     outputs=[src_image_display, target_res_textbox], js="update_slider")
    video_start_time_number.change(fn=update_start_time,
                                   inputs=[src_input_file, upscale_slider, max_mp_slider, max_res_slider, video_start_time_number],
                                   outputs=target_res_textbox)
    video_end_time_number.change(fn=update_end_time, inputs=[src_input_file, upscale_slider, max_mp_slider, max_res_slider, video_end_time_number],
                                 outputs=target_res_textbox)

    save_preset_button.click(fn=save_current_preset, inputs=[preset_name_textbox]+elements+elements_extra, outputs=[output_label, load_preset_dropdown])
    refresh_presets_button.click(fn=lambda: gr.update(choices=list_presets()), inputs=[], outputs=[load_preset_dropdown])
    
    # Add an event handler for auto-loading the last preset when dropdown value changes
    load_preset_dropdown.change(fn=load_preset, inputs=[load_preset_dropdown], outputs=[output_label] + elements+elements_extra,
                show_progress=True, queue=True)
    
    # Regular load button click
    load_preset_button.click(fn=load_preset, inputs=[load_preset_dropdown], outputs=[output_label] + elements+elements_extra,
                show_progress=True, queue=True)
    
    # Auto-load the last used preset on interface load
    block.load(fn=auto_load_last_preset, outputs=[load_preset_dropdown, output_label])

    apply_metadata_button.click(fn=apply_metadata, inputs=[meta_image], outputs=[output_label] + elements + elements_extra,
                show_progress=True, queue=True)
    
    # Event handlers for the Compare Images tab
    # Note: This handler is superseded by the more comprehensive one below that also clears selected images
    # refresh_compare_btn.click(
    #    fn=refresh_image_list,
    #    outputs=[compare_gallery],
    #    show_progress=True,
    #    queue=True
    # )
    
    # Function to handle comparison with uploaded images
    def compare_uploaded_images(img1_path, img2_path):
        if img1_path and img2_path:
            return update_comparison_images(img1_path, img2_path)
        else:
            return gr.update(visible=False), gr.update(value="Please select two images to compare")
    
    # Compare button event handler
    compare_btn.click(
        fn=compare_selected_images,
        inputs=[image1_path, image2_path, compare_img1_upload, compare_img2_upload],
        outputs=[compare_slider, compare_status],
        show_progress=True,
        queue=True
    )
    
    # Gallery selection handler
    compare_gallery1.select(
        fn=on_image_select_gallery1,
        inputs=[compare_gallery1],
        outputs=[compare_status, compare_slider, image1_path],
        show_progress=True,
        queue=True
    )
    
    compare_gallery2.select(
        fn=on_image_select_gallery2,
        inputs=[compare_gallery2],
        outputs=[compare_status, compare_slider, image2_path],
        show_progress=True,
        queue=True
    )
    
    # Refresh and clear selections
    refresh_compare_btn.click(
        fn=clear_selected_images,
        outputs=[compare_gallery1, compare_gallery2, compare_status, compare_slider, image1_path, image2_path],
        show_progress=True,
        queue=True
    )
    
    # Handle when both uploads have images
    def check_uploads(img1_path, img2_path):
        if img1_path and img2_path:
            return update_comparison_images(img1_path, img2_path)
        return gr.update(), gr.update()
    
    # Watch both uploads
    compare_img1_upload.change(
        fn=check_uploads,
        inputs=[compare_img1_upload, compare_img2_upload],
        outputs=[compare_slider, compare_status],
        show_progress=True,
        queue=True
    )
    
    compare_img2_upload.change(
        fn=check_uploads,
        inputs=[compare_img1_upload, compare_img2_upload],
        outputs=[compare_slider, compare_status],
        show_progress=True,
        queue=True
    )
    
    # Fullscreen toggle for comparison slider
    compare_slider_full_button.click(
        fn=None, # No Python function needed
        inputs=None,
        outputs=None,
        show_progress=False,
        queue=False,
        # Call the JS function with IDs for compare tab elements
        js="() => toggleSliderFullscreen('compare_slider', 'compare_preview_column', 'compare_fullscreen_button', 'compare_download_button')"
    )
    
    # Add download functionality
    compare_slider_dl_button.click(
        js="() => downloadImage('compare_slider')", # Pass the slider ID
        show_progress=False,
        queue=True,
        fn=do_nothing
    )
    
    # Add direct fullscreen toggle button handler
    compare_fullscreen_btn.click(
        fn=None, # No Python function needed
        inputs=None,
        outputs=None,
        show_progress=False,
        queue=False,
        # This targets the compare slider by default
        js="() => toggleSliderFullscreen('compare_slider', 'compare_preview_column', 'compare_fullscreen_button', 'compare_download_button')"
    )
    
    # Update the top row fullscreen toggle button handler
    toggle_fullscreen_btn.click(
        fn=None, # No Python function needed
        inputs=None,
        outputs=None,
        show_progress=False,
        queue=False,
        # This targets the compare slider by default
        js="() => toggleSliderFullscreen('compare_slider', 'compare_preview_column', 'compare_fullscreen_button', 'compare_download_button')"
    )
    
    # Batch progress now integrated directly into Gradio progress descriptions

if args.port is not None:  # Check if the --port argument is provided
    # Remove direct loading here as it won't work
    block.launch(server_name=server_ip, server_port=args.port, share=args.share, inbrowser=args.open_browser)
else:
    # Remove direct loading here as it won't work
    block.launch(server_name=server_ip, share=args.share, inbrowser=args.open_browser)