import subprocess


def create_comparison_video(image_a, image_b, output_video, params=None):
    duration=5
    frame_rate=30
    video_width=1920
    video_height=1080
    if params:       
        duration = params.get('video_duration', duration)
        frame_rate = params.get('video_fps', frame_rate)
        video_width = params.get('video_width', video_width)
        video_height = params.get('video_height', video_height)
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-loop', '1',  # Loop input images
        '-i', image_a,
        '-loop', '1',
        '-i', image_b,
        '-filter_complex',
        f"color=c=white:s=4x{video_height}[slider];"
        f"[0]scale={video_width}:{video_height}[img1];"
        f"[1]scale={video_width}:{video_height}[img2];"
        f"[img1][img2]blend=all_expr='if(gte(X,W*T/{duration}),A,B)':shortest=1[comp];"
        f"[comp][slider]overlay=x=W*t/{duration}:y=0,"
        f"format=yuv420p,scale={video_width}:{video_height}",
        '-t', str(duration),  # Duration of the video
        '-r', str(frame_rate),  # Frame rate
        '-c:v', 'libx264',  # Video codec
        '-preset', 'slow',  # Encoding preset
        '-crf', '12',  # Constant rate factor (quality)
        output_video
    ]

    subprocess.run(ffmpeg_cmd, check=True)
