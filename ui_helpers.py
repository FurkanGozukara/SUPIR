import json
import os
import re
import subprocess
from typing import List, Dict, Union

import cv2
import filetype
import gradio as gr
from ffmpeg_progress_yield import FfmpegProgress
from tqdm import tqdm

from SUPIR.perf_timer import PerfTimer
from SUPIR.utils.status_container import MediaData


def is_video(video_path: str) -> bool:
    return is_file(video_path) and filetype.helpers.is_video(video_path)


def is_image(image_path: str) -> bool:
    return is_file(image_path) and filetype.helpers.is_image(image_path)


def is_file(file_path: str) -> bool:
    return bool(file_path and os.path.isfile(file_path))


def detect_hardware_acceleration() -> (str, str, str):
    hw_accel_methods = [
        {'name': 'cuda', 'encoder': 'h264_nvenc', 'decoder': 'h264_cuvid', 'regex': re.compile(r'\bh264_nvenc\b')},
        {'name': 'qsv', 'encoder': 'h264_qsv', 'decoder': 'h264_qsv', 'regex': re.compile(r'\bh264_qsv\b')},
        {'name': 'vaapi', 'encoder': 'h264_vaapi', 'decoder': 'h264_vaapi', 'regex': re.compile(r'\bh264_vaapi\b')},
        # Add more methods here as needed, following the same structure
    ]

    ffmpeg_output = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True).stdout

    for method in hw_accel_methods:
        if method['regex'].search(ffmpeg_output):
            # Hardware acceleration method found
            return method['name'], method['decoder'], method['encoder']

    # No supported hardware acceleration found
    return '', '', ''


def extract_video(video_path: str, output_path: str, quality: int = 100, format: str = 'png', video_start=None,
                  video_end=None) -> (
        bool, Dict[str, str]):
    video_params = get_video_params(video_path)
    temp_frame_compression = 31 - (quality * 0.31)
    trim_frame_start = video_start
    trim_frame_end = video_end
    target_path = output_path
    printt(f"Extracting frames to: {target_path}, {format}")
    temp_frames_pattern = os.path.join(target_path, '%04d.' + format)
    commands = ['-hwaccel', 'auto', '-i', video_path, '-q:v', str(temp_frame_compression), '-pix_fmt', 'rgb24']
    resolution = f"{video_params['width']}x{video_params['height']}"
    video_fps = video_params['framerate']
    if trim_frame_start is not None and trim_frame_end is not None:
        commands.extend(['-vf', 'trim=start_frame=' + str(trim_frame_start) + ':end_frame=' + str(
            trim_frame_end) + ',scale=' + resolution + ',fps=' + str(video_fps)])
    elif trim_frame_start is not None:
        commands.extend(
            ['-vf', 'trim=start_frame=' + str(trim_frame_start) + ',scale=' + resolution + ',fps=' + str(video_fps)])
    elif trim_frame_end is not None:
        commands.extend(['-vf',
                         'trim=end_frame=' + str(trim_frame_end) + ',scale=' + resolution + ',fps=' + str(
                             video_fps)])
    else:
        commands.extend(['-vf', 'scale=' + resolution + ',fps=' + str(video_fps)])
    commands.extend(['-vsync', '0', temp_frames_pattern])
    printt(f"Extracting frames from video: '{' '.join(commands)}'")
    video_params['start_frame'] = trim_frame_start
    video_params['end_frame'] = trim_frame_end
    return run_ffmpeg_progress(commands), video_params


def get_video_params(video_path: str) -> Dict[str, str]:
    # Command to get video dimensions, codec, frame rate, duration, and frame count
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height,r_frame_rate,avg_frame_rate,codec_name,nb_read_frames',
           '-show_entries', 'format=duration', '-count_frames', '-of', 'json', video_path]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        # Parse ffprobe output to json
        info = json.loads(result.stdout)
        # Extract video stream information
        stream = info['streams'][0]  # Assuming the first stream is the video
        # Extract format (container) information, including duration
        format_info = info['format']

        # Calculate framerate as float
        read_frames = int(stream['nb_read_frames'])
        duration = float(format_info['duration'])
        framerate = read_frames / duration if duration > 0 else float(stream['r_frame_rate'])

        return {
            'width': stream['width'],
            'height': stream['height'],
            'framerate': framerate,
            'fps': framerate,
            'codec': stream['codec_name'],
            'duration_seconds': duration,
            'frames': read_frames  # Number of frames (might be N/A if not available)
        }
    except Exception as e:
        print(f"Error extracting video parameters: {e}")
        return {}


def compile_video(src_video: str, extracted_path: str, output_path: str, video_params: Dict[str, str], quality: int = 1,
                  file_type: str = 'mp4', video_start=None, video_end=None) -> bool:
    # if quality is a string, just make it 1
    if isinstance(quality, str):
        quality = 1.0
    output_path_with_type = f"{output_path}.{file_type}"
    if os.path.exists(output_path_with_type):
        existing_idx = 1
        while os.path.exists(f"{output_path}_{existing_idx}.{file_type}"):
            existing_idx += 1
        output_path_with_type = f"{output_path}_{existing_idx}.{file_type}"

    temp_frames_pattern = os.path.join(extracted_path, '%04d.png')
    video_fps = video_params['framerate']
    output_video_encoder = 'libx264'
    commands = ['-hwaccel', 'auto', '-r', str(video_fps), '-i', temp_frames_pattern, '-c:v',
                output_video_encoder]
    if output_video_encoder in ['libx264', 'libx265', 'h264_nvenc', 'hevc_nvenc']:
        output_video_compression = round(51 - (quality * 0.51))
        if not "nvenc" in output_video_encoder:
            commands.extend(['-crf', str(output_video_compression), '-preset', 'veryfast'])
    if output_video_encoder in ['libvpx-vp9']:
        output_video_compression = round(63 - (quality * 0.63))
        commands.extend(['-crf', str(output_video_compression)])
    commands.extend(['-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-y', output_path_with_type])
    printt(f"Merging frames to video: '{' '.join(commands)}'")
    if run_ffmpeg_progress(commands):
        image_data = MediaData(output_path_with_type, 'video')
        if restore_audio(src_video, output_path_with_type, video_fps, video_start, video_end):
            printt(f"Audio restored to video successfully: {output_path_with_type}")
        else:
            printt(f"Audio restoration failed: {output_path_with_type}")
        image_data.outputs = [output_path_with_type]
        return image_data
    return False


def run_ffmpeg_progress(args: List[str], progress=gr.Progress()):
    commands = ['ffmpeg', '-hide_banner', '-loglevel', 'error']
    commands.extend(args)
    print(f"Executing ffmpeg: '{' '.join(commands)}'")
    try:
        ff = FfmpegProgress(commands)
        last_progress = 0  # Keep track of the last progress value
        with tqdm(total=100, position=1, desc="Processing") as pbar:
            for p in ff.run_command_with_progress():
                increment = p - last_progress  # Calculate the increment since the last update
                pbar.update(increment)  # Update tqdm bar with the increment
                pbar.set_postfix(progress=p)
                progress(p / 100, "Extracting frames")  # Update gr.Progress with the normalized progress value
                last_progress = p  # Update the last progress value
        return True
    except Exception as e:
        print(f"Exception in run_ffmpeg_progress: {e}")
        return False


def get_video_frame(video_path: str, frame_number: int = 0):
    if is_video(video_path):
        video_capture = cv2.VideoCapture(video_path)
        if video_capture.isOpened():
            frame_total = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, min(frame_total, frame_number - 1))
            has_vision_frame, vision_frame = video_capture.read()
            video_capture.release()
            if has_vision_frame:
                vision_frame = cv2.cvtColor(vision_frame, cv2.COLOR_BGR2RGB)
                return vision_frame
    return None


last_time = None
ui_args = None
timer = None


def printt(msg, progress=gr.Progress(), reset: bool = False):
    global ui_args, last_time, timer
    graph = None
    if ui_args is not None and ui_args.debug:
        if timer is None:
            timer = PerfTimer(print_log=True)
        if reset:
            graph = timer.make_graph()
            timer.reset()
        if not timer.print_log:
            timer.print_log = True
        timer.record(msg)
    else:
        print(msg)
    if graph:
        return graph


import os
import subprocess


def restore_audio(src_video, target_video, video_fps, frame_start, frame_end) -> bool:
    output_video_path = os.path.splitext(target_video)[0] + "_audio_restored.mp4"
    commands = ['ffmpeg', '-hwaccel', 'auto']
    commands.extend(['-i', target_video])
    commands.extend(['-i', src_video])

    # Applying the frame cut if specified
    if frame_start is not None:
        start_time = frame_start / video_fps
        commands.extend(['-ss', str(start_time)])
    if frame_end is not None:
        end_time = frame_end / video_fps
        commands.extend(['-to', str(end_time)])

    # Copy video from target_video and audio from src_video, map them accordingly
    commands.extend(['-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-shortest', output_video_path])

    # Print command for debugging
    print(f"Executing FFmpeg command: {' '.join(commands)}")

    # Execute FFmpeg command
    try:
        subprocess.run(commands, check=True)
        print(f"Audio restored to video successfully: {output_video_path}")
        # Delete the original target_video
        os.remove(target_video)
        # Rename the restored video to the original target_video name
        os.rename(output_video_path, target_video)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error restoring audio: {e}")
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        return False

