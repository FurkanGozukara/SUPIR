import subprocess

def create_comparison_video(image_a, image_b, output_video, duration=5, frame_rate=30, video_width=1920, video_height=1080):
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-loop', '1',  # Loop input images
        '-i', image_a,
        '-loop', '1',
        '-i', image_b,
        '-filter_complex',
        f"[0]scale={video_width}:{video_height}[img1];"  # Scale image A
        f"[1]scale={video_width}:{video_height}[img2];"  # Scale image B
        f"[img1][img2]blend=all_expr='if(gte(X,W*T/{duration}),A,B)':shortest=1,"  # Slide comparison
        f"format=yuv420p,scale={video_width}:{video_height}",  # Format and scale output
        '-t', str(duration),  # Duration of the video
        '-r', str(frame_rate),  # Frame rate
        '-c:v', 'libx264',  # Video codec
        '-preset', 'slow',  # Encoding preset
        '-crf', '12',  # Constant rate factor (quality)
        output_video
    ]

    subprocess.run(ffmpeg_cmd, check=True)

create_comparison_video("1.png", "2.png", "comparison_video.mp4")