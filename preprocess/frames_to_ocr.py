import cv2
import numpy as np
import easyocr
import os
import json
import glob
import pandas as pd
from tqdm import tqdm
from PIL import Image
import re

# Initialize EasyOCR reader
reader = easyocr.Reader(['en', 'ch_sim'], gpu=True)  # Support both English and Chinese

def clean_text(text):
    """Clean and normalize OCR text"""
    if not text:
        return ""
    
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\u4e00-\u9fff]', '', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_ocr_from_frame(frame_path):
    """Extract OCR text from a single frame"""
    try:
        # Read frame
        frame = cv2.imread(frame_path)
        if frame is None:
            return ""
        
        # Run OCR
        results = reader.readtext(frame, detail=False)
        
        # Combine all detected text
        if results:
            text = ' '.join(results)
            return clean_text(text)
        else:
            return ""
            
    except Exception as e:
        print(f"OCR error for {frame_path}: {e}")
        return ""

def process_video_frames(video_frames_dir, video_id):
    """Process all frames for a single video"""
    # Get all frame files
    frame_files = sorted(glob.glob(os.path.join(video_frames_dir, "frame_*.jpg")))
    
    if len(frame_files) != 16:
        print(f"Warning: Expected 16 frames for {video_id}, found {len(frame_files)}")
    
    # Extract OCR from each frame
    frame_ocr_list = []
    global_ocr_text = []
    
    for frame_file in frame_files:
        ocr_text = extract_ocr_from_frame(frame_file)
        frame_ocr_list.append(ocr_text)
        if ocr_text:  # Only add non-empty text to global
            global_ocr_text.append(ocr_text)
    
    # Create global OCR by combining all frame OCR
    global_ocr = ' '.join(global_ocr_text)
    global_ocr = clean_text(global_ocr)
    
    return frame_ocr_list, global_ocr

def process_dataset():
    """Process FakeSV dataset for OCR extraction"""
    
    dataset_name = "FakeSV"
    frames_dir = f"data/{dataset_name}/frames_16"
    
    if not os.path.exists(frames_dir):
        print(f"Error: Frames directory not found: {frames_dir}")
        return
    
    # Get list of all video IDs
    video_dirs = [d for d in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, d))]
    print(f"Found {len(video_dirs)} videos to process")
    
    # Storage for results
    frame_ocr_results = []
    global_ocr_results = []
    
    # Process each video
    for video_id in tqdm(video_dirs, desc="Processing OCR"):
        video_frames_dir = os.path.join(frames_dir, video_id)
        
        # Process frames
        frame_ocr_list, global_ocr = process_video_frames(video_frames_dir, video_id)
        
        # Store results
        frame_ocr_results.append({
            'vid': video_id,
            'frame_ocr': frame_ocr_list
        })
        
        global_ocr_results.append({
            'vid': video_id,
            'ocr': global_ocr
        })
    
    # Save frame-level OCR
    with open(f"data/{dataset_name}/ocr_frames.jsonl", 'w', encoding='utf-8') as f:
        for item in frame_ocr_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Save global OCR
    with open(f"data/{dataset_name}/ocr_global.jsonl", 'w', encoding='utf-8') as f:
        for item in global_ocr_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"OCR extraction complete!")
    print(f"  Frame-level OCR: {len(frame_ocr_results)} videos -> data/{dataset_name}/ocr_frames.jsonl")
    print(f"  Global OCR: {len(global_ocr_results)} videos -> data/{dataset_name}/ocr_global.jsonl")
    
    # Print some statistics
    total_frame_texts = sum(len([t for t in item['frame_ocr'] if t]) for item in frame_ocr_results)
    total_global_texts = sum(1 for item in global_ocr_results if item['ocr'])
    
    print(f"Statistics:")
    print(f"  Total frame texts extracted: {total_frame_texts}")
    print(f"  Videos with global OCR text: {total_global_texts}")

if __name__ == "__main__":
    process_dataset()