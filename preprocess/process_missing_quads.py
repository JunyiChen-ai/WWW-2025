import os
from PIL import Image
from tqdm import tqdm

def process_single_video(vid_folder, input_folder, output_folder):
    """Process a single video folder to generate quad frames"""
    frames_folder = os.path.join(input_folder, vid_folder)
    
    frames = []
    for i in range(15):  # Only 15 frames (0-14) in these videos
        frame_path = os.path.join(frames_folder, f"frame_{i:03d}.jpg")
        if os.path.exists(frame_path):
            frame = Image.open(frame_path)
            frames.append(frame)
        else:
            print(f"Cannot find frame: {frame_path}")
            return False
    
    if len(frames) < 15:
        print(f"Folder {vid_folder} contains fewer than 15 frames")
        return False
    
    # Duplicate last frame to make 16
    frames.append(frames[-1])

    # Create 4 images with 2x2 grid layout
    for quad_index in range(4):
        start_index = quad_index * 4
        quad_frames = frames[start_index:start_index+4]
        
        min_width = min(frame.width for frame in quad_frames)
        min_height = min(frame.height for frame in quad_frames)
        quad_frames = [frame.resize((min_width, min_height)) for frame in quad_frames]

        grid = Image.new('RGB', (min_width * 2, min_height * 2))
        grid.paste(quad_frames[0], (0, 0))
        grid.paste(quad_frames[1], (min_width, 0))
        grid.paste(quad_frames[2], (0, min_height))
        grid.paste(quad_frames[3], (min_width, min_height))

        output_path = os.path.join(output_folder, f"{vid_folder}_quad_{quad_index}.jpg")
        grid.save(output_path, 'JPEG')
        print(f"Saved: {output_path}")
    
    return True

# Process missing videos
missing_videos = [
    "3x432tjvxinwezy",
    "6797622943372365067", 
    "6800199936664292608",
    "6821060814020021516",
    "6852979877516283151"
]

input_folder = 'data/FakeSV/frames_16'
output_folder = 'data/FakeSV/quads_4'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

success_count = 0
for vid in tqdm(missing_videos, desc="Processing missing videos"):
    if process_single_video(vid, input_folder, output_folder):
        success_count += 1
        print(f"Successfully processed: {vid}")
    else:
        print(f"Failed to process: {vid}")

print(f"\nProcessed {success_count}/{len(missing_videos)} videos successfully")