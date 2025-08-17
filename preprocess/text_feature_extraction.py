import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import ChineseCLIPProcessor, ChineseCLIPTextModel
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize models
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize Chinese-CLIP text model
print("Loading Chinese-CLIP text model...")
model_name = "OFA-Sys/chinese-clip-vit-large-patch14"
text_processor = ChineseCLIPProcessor.from_pretrained(model_name)
text_model = ChineseCLIPTextModel.from_pretrained(model_name)
text_model.to(device)
text_model.eval()

def load_metadata(dataset_name):
    """Load video metadata with titles and labels"""
    metadata_path = f"data/{dataset_name}/data_complete.jsonl"
    if not os.path.exists(metadata_path):
        print(f"Warning: Metadata file not found: {metadata_path}")
        return {}
    
    metadata = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            video_id = data.get('video_id', '')
            title = data.get('title', '')
            metadata[video_id] = {
                'title': title,
                'annotation': data.get('annotation', ''),
                'ocr': data.get('ocr', '')  # Global OCR from metadata
            }
    
    print(f"Loaded metadata for {len(metadata)} videos")
    return metadata

def load_ocr_data(dataset_name):
    """Load frame-level and global OCR data"""
    ocr_frames = {}
    ocr_global = {}
    
    # Load frame-level OCR
    frames_path = f"data/{dataset_name}/ocr_frames.jsonl"
    if os.path.exists(frames_path):
        with open(frames_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                video_id = data['vid']
                ocr_frames[video_id] = data['frame_ocr']
        print(f"Loaded frame-level OCR for {len(ocr_frames)} videos")
    
    # Load global OCR  
    global_path = f"data/{dataset_name}/ocr_global.jsonl"
    if os.path.exists(global_path):
        with open(global_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                video_id = data['vid']
                ocr_global[video_id] = data['ocr']
        print(f"Loaded global OCR for {len(ocr_global)} videos")
    
    return ocr_frames, ocr_global

def load_transcript_data(dataset_name):
    """Load frame-level transcript data"""
    transcripts = {}
    
    # Load frame-level transcripts
    transcripts_path = f"data/{dataset_name}/transcript_frames.jsonl"
    if os.path.exists(transcripts_path):
        with open(transcripts_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                video_id = data['vid']
                transcripts[video_id] = data['frame_transcripts']
        print(f"Loaded frame-level transcripts for {len(transcripts)} videos")
    
    return transcripts

def load_cot_data(dataset_name):
    """Load Chain-of-Thought refined text data (optional enhancement)"""
    cot_data = {}
    cot_dir = f"data/{dataset_name}/CoT/gpt-4o"
    
    if os.path.exists(cot_dir):
        # Load text refinement
        text_refine_path = os.path.join(cot_dir, "lm_text_refine.jsonl")
        if os.path.exists(text_refine_path):
            text_refine = {}
            with open(text_refine_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    video_id = data.get('vid', '')
                    text_refine[video_id] = data.get('text', '')
            cot_data['text_refine'] = text_refine
            print(f"Loaded CoT text refinement for {len(text_refine)} videos")
        
        # Load visual refinement
        visual_refine_path = os.path.join(cot_dir, "lm_visual_refine.jsonl")
        if os.path.exists(visual_refine_path):
            visual_refine = {}
            with open(visual_refine_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    video_id = data.get('vid', '')
                    visual_refine[video_id] = data.get('text', '')
            cot_data['visual_refine'] = visual_refine
            print(f"Loaded CoT visual refinement for {len(visual_refine)} videos")
    
    return cot_data

def encode_text(text, max_length=77):
    """Encode text using Chinese-CLIP text encoder"""
    try:
        if not text or not text.strip():
            # Return zero vector for empty text
            return torch.zeros(512, device='cpu')
        
        # Process text
        inputs = text_processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = text_model(**inputs)
            # Get pooled output (sentence-level representation)
            text_features = outputs.pooler_output.squeeze(0).cpu()
            
        return text_features
        
    except Exception as e:
        print(f"Error encoding text: {e}")
        return torch.zeros(512, device='cpu')

def combine_text_sources(video_id, metadata, ocr_frames, ocr_global, transcripts, cot_data, use_cot=True):
    """Combine different text sources for a video"""
    
    # Get title
    title = metadata.get(video_id, {}).get('title', '')
    
    # Get OCR data
    frame_ocr = ocr_frames.get(video_id, [''] * 16)
    global_ocr = ocr_global.get(video_id, '')
    
    # Get transcript data
    frame_transcripts = transcripts.get(video_id, [''] * 16)
    
    # Ensure we have exactly 16 frames
    if len(frame_ocr) != 16:
        print(f"Warning: {video_id} has {len(frame_ocr)} OCR frames, padding/truncating to 16")
        frame_ocr = (frame_ocr + [''] * 16)[:16]
    
    if len(frame_transcripts) != 16:
        print(f"Warning: {video_id} has {len(frame_transcripts)} transcript frames, padding/truncating to 16")
        frame_transcripts = (frame_transcripts + [''] * 16)[:16]
    
    # Frame-level text: combine title + frame_ocr + frame_transcript for each frame
    frame_texts = []
    for i in range(16):
        combined_text = []
        if title.strip():
            combined_text.append(title.strip())
        if frame_ocr[i].strip():
            combined_text.append(frame_ocr[i].strip())
        if frame_transcripts[i].strip():
            combined_text.append(frame_transcripts[i].strip())
        
        frame_text = ' '.join(combined_text) if combined_text else ''
        frame_texts.append(frame_text)
    
    # Global text: combine title + global_ocr + global_transcript
    global_text_parts = []
    if title.strip():
        global_text_parts.append(title.strip())
    if global_ocr.strip():
        global_text_parts.append(global_ocr.strip())
    
    # Create global transcript from frame transcripts
    global_transcript = ' '.join([t.strip() for t in frame_transcripts if t.strip()])
    if global_transcript.strip():
        global_text_parts.append(global_transcript)
    
    # Add CoT refinement if available and requested
    if use_cot and cot_data:
        if 'text_refine' in cot_data and video_id in cot_data['text_refine']:
            refined_text = cot_data['text_refine'][video_id].strip()
            if refined_text:
                global_text_parts.append(refined_text)
        
        if 'visual_refine' in cot_data and video_id in cot_data['visual_refine']:
            visual_refined = cot_data['visual_refine'][video_id].strip()
            if visual_refined:
                global_text_parts.append(visual_refined)
    
    global_text = ' '.join(global_text_parts) if global_text_parts else ''
    
    return frame_texts, global_text

def process_dataset(dataset_name="FakeSV", use_cot=True, batch_size=16):
    """Process dataset for text feature extraction"""
    
    print(f"Processing {dataset_name} dataset for text features...")
    
    # Load all data sources
    metadata = load_metadata(dataset_name)
    ocr_frames, ocr_global = load_ocr_data(dataset_name)
    transcripts = load_transcript_data(dataset_name)
    cot_data = load_cot_data(dataset_name) if use_cot else {}
    
    # Get list of all video IDs from visual features (our reference)
    visual_features_path = f"data/{dataset_name}/fea/vit_tensor.pt"
    if not os.path.exists(visual_features_path):
        print(f"Error: Visual features not found: {visual_features_path}")
        return
    
    visual_data = torch.load(visual_features_path)
    video_ids = list(visual_data.keys())
    print(f"Processing text features for {len(video_ids)} videos")
    
    # Create output directory
    os.makedirs(f"data/{dataset_name}/fea", exist_ok=True)
    
    # Process videos in batches
    all_frame_features = []
    all_global_features = []
    processed_video_ids = []
    
    print("Extracting text features...")
    for i in tqdm(range(0, len(video_ids), batch_size), desc="Processing batches"):
        batch_video_ids = video_ids[i:i+batch_size]
        batch_frame_features = []
        batch_global_features = []
        
        for video_id in batch_video_ids:
            try:
                # Combine text sources
                frame_texts, global_text = combine_text_sources(
                    video_id, metadata, ocr_frames, ocr_global, transcripts, cot_data, use_cot
                )
                
                # Encode frame-level texts
                frame_features = []
                for frame_text in frame_texts:
                    feature = encode_text(frame_text)
                    frame_features.append(feature)
                
                frame_features_tensor = torch.stack(frame_features)  # [16, 512]
                batch_frame_features.append(frame_features_tensor)
                
                # Encode global text
                global_feature = encode_text(global_text)
                batch_global_features.append(global_feature)
                
                processed_video_ids.append(video_id)
                
            except Exception as e:
                print(f"Error processing {video_id}: {e}")
                # Add zero tensors for failed videos
                batch_frame_features.append(torch.zeros(16, 512))
                batch_global_features.append(torch.zeros(512))
                processed_video_ids.append(video_id)
        
        # Add batch to overall lists
        all_frame_features.extend(batch_frame_features)
        all_global_features.extend(batch_global_features)
    
    # Stack all features
    if all_frame_features:
        frame_features_tensor = torch.stack(all_frame_features)  # [N, 16, 512]
        global_features_tensor = torch.stack(all_global_features)  # [N, 512]
        
        # Save features
        frame_output_path = f"data/{dataset_name}/fea/text_features_frames.pt"
        global_output_path = f"data/{dataset_name}/fea/text_features_global.pt"
        
        torch.save(frame_features_tensor, frame_output_path)
        torch.save(global_features_tensor, global_output_path)
        
        # Save video ID mapping for reference
        with open(f"data/{dataset_name}/fea/text_video_ids.txt", 'w') as f:
            for video_id in processed_video_ids:
                f.write(f"{video_id}\n")
        
        print(f"\n=== TEXT FEATURE EXTRACTION SUMMARY ===")
        print(f"Processed videos: {len(processed_video_ids)}")
        print(f"Frame features shape: {frame_features_tensor.shape}")
        print(f"Global features shape: {global_features_tensor.shape}")
        print(f"Frame features saved to: {frame_output_path}")
        print(f"Global features saved to: {global_output_path}")
        
        # Calculate text coverage statistics
        non_empty_frame_texts = 0
        non_empty_global_texts = 0
        total_frame_texts = 0
        
        for video_id in processed_video_ids[:100]:  # Sample first 100 videos
            frame_texts, global_text = combine_text_sources(
                video_id, metadata, ocr_frames, ocr_global, transcripts, cot_data, use_cot
            )
            
            for frame_text in frame_texts:
                total_frame_texts += 1
                if frame_text.strip():
                    non_empty_frame_texts += 1
            
            if global_text.strip():
                non_empty_global_texts += 1
        
        frame_coverage = (non_empty_frame_texts / total_frame_texts * 100) if total_frame_texts > 0 else 0
        global_coverage = (non_empty_global_texts / 100 * 100) if len(processed_video_ids) > 0 else 0
        
        print(f"\n=== TEXT COVERAGE STATISTICS (sample) ===")
        print(f"Frame-level text coverage: {frame_coverage:.1f}%")
        print(f"Global text coverage: {global_coverage:.1f}%")
        print(f"CoT enhancement: {'Enabled' if use_cot else 'Disabled'}")
        
    else:
        print("No videos were successfully processed!")

if __name__ == "__main__":
    # Process FakeSV dataset with CoT enhancement
    process_dataset("FakeSV", use_cot=True, batch_size=16)