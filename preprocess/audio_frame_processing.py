import os
import json
import librosa
import torch
import torchaudio
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model, WhisperProcessor, WhisperForConditionalGeneration
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize models
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Using device: {device}")

# Initialize Wav2Vec2 model
print("Loading Wav2Vec2 model...")
wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model.to(device)
wav2vec_model.eval()

# Initialize Whisper model  
print("Loading Whisper model...")
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", torch_dtype=torch_dtype)
whisper_model.to(device)
whisper_model.eval()

def load_frame_timestamps(timestamp_file):
    """Load frame timestamps from file"""
    timestamps = []
    with open(timestamp_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                frame_name = parts[0]
                start_time = float(parts[1])
                end_time = float(parts[2])
                timestamps.append((frame_name, start_time, end_time))
    return timestamps

def process_frame_aligned_audio(video_id, audio_path, frame_timestamps):
    """Process audio segments aligned with video frames in-memory"""
    try:
        # Load full audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Audio validation warnings
        audio_duration = len(audio) / sr
        if audio_duration < 1.0:
            print(f"WARNING: {video_id} - Very short audio duration: {audio_duration:.2f}s")
        elif audio_duration > 300.0:  # 5 minutes
            print(f"WARNING: {video_id} - Very long audio duration: {audio_duration:.2f}s")
        
        if len(frame_timestamps) != 16:
            print(f"WARNING: {video_id} - Expected 16 frames, got {len(frame_timestamps)} timestamps")
        
        frame_features = []
        frame_transcripts = []
        invalid_segments = 0
        short_segments = 0
        zero_segments = 0
        
        for i, (frame_name, start_time, end_time) in enumerate(frame_timestamps):
            # Timestamp validation
            segment_duration = end_time - start_time
            if segment_duration <= 0:
                print(f"WARNING: {video_id}, {frame_name} - Invalid timestamp: start={start_time:.3f}s, end={end_time:.3f}s, duration={segment_duration:.3f}s")
                invalid_segments += 1
            elif segment_duration > audio_duration:
                print(f"WARNING: {video_id}, {frame_name} - Segment longer than audio: segment={segment_duration:.3f}s, audio={audio_duration:.3f}s")
            
            # Extract segment in memory (numpy array)
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Bounds validation
            if start_sample < 0:
                print(f"WARNING: {video_id}, {frame_name} - Negative start sample: {start_sample} (start_time={start_time:.3f}s)")
            if end_sample > len(audio):
                print(f"WARNING: {video_id}, {frame_name} - End sample exceeds audio: {end_sample} > {len(audio)} (end_time={end_time:.3f}s, audio_duration={audio_duration:.3f}s)")
            
            # Ensure valid segment bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            if end_sample <= start_sample:
                # Invalid segment, use zeros
                print(f"WARNING: {video_id}, {frame_name} - Invalid segment bounds: start_sample={start_sample}, end_sample={end_sample}")
                segment_length = int(0.5 * sr)  # 0.5 second default
                segment = torch.zeros(segment_length)
                transcript = ""
                zero_segments += 1
            else:
                segment = audio[start_sample:end_sample]
                segment_length_sec = len(segment) / sr
                
                # Segment length warnings
                if segment_length_sec < 0.1:
                    short_segments += 1
                    if short_segments <= 3:  # Limit spam
                        print(f"WARNING: {video_id}, {frame_name} - Very short segment: {segment_length_sec:.3f}s")
                elif segment_length_sec > 5.0:  # Very long segment
                    print(f"WARNING: {video_id}, {frame_name} - Very long segment: {segment_length_sec:.3f}s")
                
                # Generate transcript with whisper (only if segment is long enough)
                if len(segment) > 0.1 * sr:  # At least 0.1 seconds
                    try:
                        # Process with Whisper
                        input_features = whisper_processor(segment, sampling_rate=16000, return_tensors="pt").input_features
                        input_features = input_features.to(device, dtype=torch_dtype)  # Match model dtype
                        
                        with torch.no_grad():
                            predicted_ids = whisper_model.generate(input_features)
                            transcript = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                            
                        # Transcript quality warnings
                        if len(transcript.strip()) == 0:
                            pass  # Silent segment is normal
                        elif len(transcript) > 200:  # Very long transcript
                            print(f"WARNING: {video_id}, {frame_name} - Very long transcript ({len(transcript)} chars): {transcript[:50]}...")
                            
                    except Exception as e:
                        print(f"ERROR: Whisper failed for {video_id}, {frame_name}: {e}")
                        transcript = ""
                else:
                    transcript = ""
                
                # Convert to torch tensor for wav2vec
                segment = torch.tensor(segment, dtype=torch.float32)
            
            # Process through wav2vec (ensure minimum length)
            if len(segment) < 400:  # Wav2Vec2 minimum length
                segment = torch.cat([segment, torch.zeros(400 - len(segment))])
            
            try:
                # Process with Wav2Vec2
                segment = segment.unsqueeze(0).to(device)
                with torch.no_grad():
                    wav2vec_outputs = wav2vec_model(segment)
                    # Use mean pooling over sequence length to get fixed-size features
                    features = wav2vec_outputs.last_hidden_state.mean(dim=1)  # [1, 768]
                    frame_features.append(features.cpu())
            except Exception as e:
                print(f"Wav2Vec2 error for {video_id}, {frame_name}: {e}")
                # Fallback: zero features
                frame_features.append(torch.zeros(1, 768))
            
            frame_transcripts.append(transcript.strip())
        
        # Video-level summary warnings
        if invalid_segments > 0:
            print(f"WARNING: {video_id} - {invalid_segments} invalid segments found")
        if short_segments > 3:
            print(f"WARNING: {video_id} - {short_segments} segments shorter than 0.1s")
        if zero_segments > 0:
            print(f"WARNING: {video_id} - {zero_segments} segments replaced with zeros")
        
        # Stack all frame features
        if frame_features:
            stacked_features = torch.cat(frame_features, dim=0)  # [16, 768]
        else:
            print(f"ERROR: {video_id} - No valid features extracted, using zero tensor")
            stacked_features = torch.zeros(16, 768)
            
        return stacked_features, frame_transcripts
        
    except Exception as e:
        print(f"Error processing {video_id}: {e}")
        # Return zero features and empty transcripts
        return torch.zeros(16, 768), [""] * 16

def process_global_audio(audio_path):
    """Process full audio file for global features"""
    try:
        # Load full audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Process through wav2vec
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            wav2vec_outputs = wav2vec_model(audio_tensor)
            # Use mean pooling over sequence length to get fixed-size features
            global_features = wav2vec_outputs.last_hidden_state.mean(dim=1)  # [1, 768]
            
        return global_features.cpu().squeeze(0)  # [768]
        
    except Exception as e:
        print(f"Error processing global audio {audio_path}: {e}")
        return torch.zeros(768)

def load_existing_results(dataset_name):
    """Load existing processing results to enable resume functionality"""
    existing_frame_features = []
    existing_global_features = []
    existing_frame_transcripts = []
    existing_video_ids = set()
    
    # Try to load existing features
    frame_features_path = f"data/{dataset_name}/fea/audio_features_frames.pt"
    global_features_path = f"data/{dataset_name}/fea/audio_features_global.pt"
    video_ids_path = f"data/{dataset_name}/fea/audio_video_ids.txt"
    transcripts_path = f"data/{dataset_name}/transcript_frames.jsonl"
    
    if os.path.exists(frame_features_path) and os.path.exists(global_features_path) and os.path.exists(video_ids_path):
        try:
            # Load existing features
            frame_features_data = torch.load(frame_features_path)
            global_features_data = torch.load(global_features_path)
            
            # Check if both are dictionaries (new format)
            if isinstance(frame_features_data, dict) and isinstance(global_features_data, dict):
                # New format: both are dictionaries with video IDs as keys
                existing_video_list = list(global_features_data.keys())
                
                # Load existing transcripts
                existing_transcripts_dict = {}
                if os.path.exists(transcripts_path):
                    with open(transcripts_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line.strip())
                            existing_transcripts_dict[data['vid']] = data['frame_transcripts']
                
                # Convert to lists for merging
                for video_id in existing_video_list:
                    existing_frame_features.append(frame_features_data[video_id])
                    existing_global_features.append(global_features_data[video_id])
                    existing_frame_transcripts.append(existing_transcripts_dict.get(video_id, [""] * 16))
                    existing_video_ids.add(video_id)
            elif isinstance(global_features_data, dict):
                # Mixed format: frame features is tensor, global is dict
                existing_video_list = list(global_features_data.keys())
                
                # Load existing transcripts
                existing_transcripts_dict = {}
                if os.path.exists(transcripts_path):
                    with open(transcripts_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line.strip())
                            existing_transcripts_dict[data['vid']] = data['frame_transcripts']
                
                # Convert to lists for merging
                for i, video_id in enumerate(existing_video_list):
                    existing_frame_features.append(frame_features_data[i])
                    existing_global_features.append(global_features_data[video_id])
                    existing_frame_transcripts.append(existing_transcripts_dict.get(video_id, [""] * 16))
                    existing_video_ids.add(video_id)
            else:
                # Old format: tensor with separate video ID file
                # Load existing video IDs
                with open(video_ids_path, 'r') as f:
                    existing_video_list = [line.strip() for line in f.readlines()]
                
                # Load existing transcripts
                existing_transcripts_dict = {}
                if os.path.exists(transcripts_path):
                    with open(transcripts_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line.strip())
                            existing_transcripts_dict[data['vid']] = data['frame_transcripts']
                
                # Convert to lists for merging
                for i, video_id in enumerate(existing_video_list):
                    existing_frame_features.append(frame_features_tensor[i])
                    existing_global_features.append(global_features_data[i])
                    existing_frame_transcripts.append(existing_transcripts_dict.get(video_id, [""] * 16))
                    existing_video_ids.add(video_id)
            
            print(f"Loaded {len(existing_video_list)} existing processed videos")
            
        except Exception as e:
            print(f"Warning: Failed to load existing results: {e}")
            print("Starting from scratch...")
            existing_frame_features = []
            existing_global_features = []
            existing_frame_transcripts = []
            existing_video_ids = set()
    else:
        print("No existing results found, starting from scratch...")
    
    return existing_frame_features, existing_global_features, existing_frame_transcripts, existing_video_ids

def process_dataset(dataset_name):
    """Process dataset for frame-level and global audio features with resume capability"""
    
    frames_dir = f"data/{dataset_name}/frames_16"
    audios_dir = f"data/{dataset_name}/audios"
    
    # Create output directories
    os.makedirs(f"data/{dataset_name}/fea", exist_ok=True)
    
    # Get list of all video IDs from frames directory
    if not os.path.exists(frames_dir):
        print(f"Error: Frames directory not found: {frames_dir}")
        return
    
    video_dirs = [d for d in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, d))]
    print(f"Found {len(video_dirs)} total videos in frames directory")
    
    # Load existing results
    all_frame_features, all_global_features, all_frame_transcripts, existing_video_ids = load_existing_results(dataset_name)
    all_video_ids = list(existing_video_ids)
    
    # Filter out already processed videos
    remaining_videos = [vid for vid in video_dirs if vid not in existing_video_ids]
    print(f"Already processed: {len(existing_video_ids)} videos")
    print(f"Remaining to process: {len(remaining_videos)} videos")
    
    # Process remaining videos only
    if remaining_videos:
        print(f"Processing {len(remaining_videos)} remaining videos...")
        for video_id in tqdm(remaining_videos, desc="Processing remaining videos"):
            video_frames_dir = os.path.join(frames_dir, video_id)
            timestamp_file = os.path.join(video_frames_dir, "frame_timestamps.txt")
            audio_path = os.path.join(audios_dir, f"{video_id}_full.wav")
            
            # Check if timestamp file exists
            if not os.path.exists(timestamp_file):
                print(f"ERROR: Timestamp file not found for {video_id}: {timestamp_file}")
                continue
                
            # Check if audio file exists
            if not os.path.exists(audio_path):
                print(f"ERROR: Audio file not found for {video_id}: {audio_path}")
                continue
            
            # Load frame timestamps
            frame_timestamps = load_frame_timestamps(timestamp_file)
            if len(frame_timestamps) != 16:
                print(f"ERROR: Expected 16 frame timestamps for {video_id}, got {len(frame_timestamps)} - skipping video")
                continue
            elif len(frame_timestamps) == 0:
                print(f"ERROR: No timestamps found in file for {video_id}: {timestamp_file}")
                continue
            
            # Process frame-level audio
            frame_features, frame_transcripts = process_frame_aligned_audio(video_id, audio_path, frame_timestamps)
            
            # Process global audio
            global_features = process_global_audio(audio_path)
            
            # Store results
            all_frame_features.append(frame_features)
            all_global_features.append(global_features)
            all_frame_transcripts.append(frame_transcripts)
            all_video_ids.append(video_id)
    else:
        print("All videos have already been processed!")
    
    # Final processing statistics
    total_available = len(video_dirs)
    successfully_processed = len(all_video_ids)
    newly_processed = successfully_processed - len(existing_video_ids)
    skipped = total_available - successfully_processed
    
    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Total videos available: {total_available}")
    print(f"Previously processed: {len(existing_video_ids)}")
    print(f"Newly processed: {newly_processed}")
    print(f"Total successfully processed: {successfully_processed}")
    print(f"Skipped due to errors: {skipped}")
    
    if skipped > 0:
        print(f"WARNING: {skipped} videos were skipped due to missing files or invalid timestamps")
    
    if all_video_ids:
        # Create dictionaries for both frame and global features with video IDs as keys
        frame_features_dict = {}
        global_features_dict = {}
        for video_id, frame_feat, global_feat in zip(all_video_ids, all_frame_features, all_global_features):
            frame_features_dict[video_id] = frame_feat
            global_features_dict[video_id] = global_feat
        
        # Save features as dictionaries
        torch.save(frame_features_dict, f"data/{dataset_name}/fea/audio_features_frames.pt")
        torch.save(global_features_dict, f"data/{dataset_name}/fea/audio_features_global.pt")
        
        # Save frame transcripts
        frame_transcripts_data = []
        for video_id, transcripts in zip(all_video_ids, all_frame_transcripts):
            frame_transcripts_data.append({
                'vid': video_id,
                'frame_transcripts': transcripts
            })
        
        with open(f"data/{dataset_name}/transcript_frames.jsonl", 'w', encoding='utf-8') as f:
            for item in frame_transcripts_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Save video ID mapping for reference
        with open(f"data/{dataset_name}/fea/audio_video_ids.txt", 'w') as f:
            for video_id in all_video_ids:
                f.write(f"{video_id}\n")
        
        # Calculate transcript quality statistics
        total_transcripts = 0
        non_empty_transcripts = 0
        total_transcript_chars = 0
        
        for transcripts in all_frame_transcripts:
            for transcript in transcripts:
                total_transcripts += 1
                if transcript.strip():
                    non_empty_transcripts += 1
                    total_transcript_chars += len(transcript)
        
        transcript_success_rate = (non_empty_transcripts / total_transcripts * 100) if total_transcripts > 0 else 0
        avg_transcript_length = (total_transcript_chars / non_empty_transcripts) if non_empty_transcripts > 0 else 0
        
        print(f"\n=== OUTPUT FILES ===")
        print(f"Frame features: dict with {len(frame_features_dict)} video IDs -> data/{dataset_name}/fea/audio_features_frames.pt")
        print(f"Global features: dict with {len(global_features_dict)} video IDs -> data/{dataset_name}/fea/audio_features_global.pt")
        print(f"Frame transcripts: {len(frame_transcripts_data)} videos -> data/{dataset_name}/transcript_frames.jsonl")
        
        print(f"\n=== TRANSCRIPT QUALITY ===")
        print(f"Total frame segments: {total_transcripts}")
        print(f"Successful transcripts: {non_empty_transcripts} ({transcript_success_rate:.1f}%)")
        print(f"Average transcript length: {avg_transcript_length:.1f} characters")
        
        if transcript_success_rate < 10:
            print(f"WARNING: Very low transcript success rate ({transcript_success_rate:.1f}%) - check audio quality")
        elif transcript_success_rate < 30:
            print(f"WARNING: Low transcript success rate ({transcript_success_rate:.1f}%) - many silent segments")
        
    else:
        print("No videos were successfully processed!")

def main():
    parser = argparse.ArgumentParser(description='Process audio features for video dataset')
    parser.add_argument('--dataset', type=str, default='FakeSV', 
                        choices=['FakeSV', 'FakeTT', 'FVC'],
                        help='Dataset name (default: FakeSV)')
    
    args = parser.parse_args()
    
    print(f"Processing dataset: {args.dataset}")
    print(f"Expected directory structure:")
    print(f"  - data/{args.dataset}/frames_16/ (video frame directories with timestamps)")
    print(f"  - data/{args.dataset}/audios/ (audio files)")
    print(f"Will create:")
    print(f"  - data/{args.dataset}/fea/audio_features_frames.pt")
    print(f"  - data/{args.dataset}/fea/audio_features_global.pt")
    print(f"  - data/{args.dataset}/transcript_frames.jsonl")
    print()
    
    process_dataset(args.dataset)

if __name__ == "__main__":
    main()