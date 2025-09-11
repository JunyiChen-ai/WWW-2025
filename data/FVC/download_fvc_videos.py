#!/usr/bin/env python3

import csv
import json
import os
import subprocess
import time
import hashlib
import sys
import re
import glob

def load_checkpoint(checkpoint_file):
    """Load checkpoint data to track processed videos"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_checkpoint(checkpoint_file, data):
    """Save current state to checkpoint file"""
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def extract_real_video_id(url):
    """Extract real video ID from URL"""
    # YouTube
    youtube_patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in youtube_patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # Facebook - extract from URL
    facebook_pattern = r'facebook\.com/.+/videos/(\d+)'
    match = re.search(facebook_pattern, url)
    if match:
        return f"fb_{match.group(1)}"
    
    # For other platforms, use a hash of the URL as fallback
    return f"hash_{hashlib.md5(url.encode()).hexdigest()[:12]}"

def extract_metadata_with_ytdlp(url):
    """Extract metadata using yt-dlp without downloading"""
    try:
        cmd = [
            'yt-dlp',
            '--dump-json',
            '--no-download',
            '--no-warnings',
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            metadata = json.loads(result.stdout)
            
            # Get real video ID from metadata, fallback to URL extraction
            real_id = metadata.get('id', extract_real_video_id(url))
            
            return {
                'success': True,
                'real_id': real_id,
                'title': metadata.get('title', ''),
                'description': metadata.get('description', ''),
                'uploader': metadata.get('uploader', ''),
                'upload_date': metadata.get('upload_date', ''),
                'timestamp': metadata.get('timestamp', None),
                'duration': metadata.get('duration', None),
                'view_count': metadata.get('view_count', None)
            }
        else:
            return {
                'success': False,
                'error': result.stderr.strip(),
                'real_id': extract_real_video_id(url)
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'real_id': extract_real_video_id(url)
        }

def download_video_mp4(url, output_path, real_id):
    """Download video using yt-dlp, force MP4 format"""
    try:
        cmd = [
            'yt-dlp',
            '-o', f'{output_path}/{real_id}.%(ext)s',
            '--recode-video', 'mp4',
            '--no-write-info-json',
            '--no-write-description',
            '--no-write-annotations',
            '--no-write-thumbnail',
            '--merge-output-format', 'mp4',
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            expected_file = f'{output_path}/{real_id}.mp4'
            if os.path.exists(expected_file):
                return {'success': True, 'message': 'Downloaded successfully'}
            else:
                return {'success': False, 'error': 'File not found after download'}
        else:
            return {'success': False, 'error': result.stderr.strip()}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def format_timestamp(timestamp):
    """Convert timestamp to milliseconds format"""
    if timestamp:
        try:
            return int(timestamp * 1000)
        except:
            pass
    return int(time.time() * 1000)  # Use current time as fallback

def is_critical_error(error_msg):
    """Check if error is critical (unrecoverable)"""
    critical_patterns = [
        'Video unavailable',
        'removed for violating',
        'Community Guidelines',
        'Terms of Service',
        'account associated with this video has been terminated',
        'Private video',
        'This video has been removed',
        'This video has been removed by the uploader',
        'only available for registered users'  # Facebook login required
    ]
    
    for pattern in critical_patterns:
        if pattern in error_msg:
            return True
    return False

def should_retry_video(item):
    """Check if a video should be retried based on its previous processing status"""
    if not item:
        return True  # New video, should process
    
    status = item.get('processing_status', '')
    error = item.get('error', '')
    
    # Always skip if successfully downloaded
    if status == 'downloaded':
        return False
    
    # Don't retry critical errors
    if status == 'failed' and error and is_critical_error(error):
        return False
    
    # Retry temporary errors (503, network issues, etc.)
    return True

def cleanup_old_files(video_dir):
    """Clean up old format files and rename to new format"""
    print("Cleaning up old format files...")
    
    # Find all .info.json files to extract real IDs
    info_files = glob.glob(f"{video_dir}/*.info.json")
    
    for info_file in info_files:
        try:
            # Read the info.json to get real ID
            with open(info_file, 'r', encoding='utf-8') as f:
                info_data = json.load(f)
                real_id = info_data.get('id', '')
            
            if real_id:
                # Find corresponding video files
                base_name = info_file.replace('.info.json', '')
                video_files = glob.glob(f"{base_name}.*")
                
                for video_file in video_files:
                    if not video_file.endswith('.info.json'):
                        # This is a video file, rename it
                        new_name = f"{video_dir}/{real_id}.mp4"
                        if not os.path.exists(new_name):
                            # Convert to mp4 if not already
                            if not video_file.endswith('.mp4'):
                                print(f"  Converting {video_file} to {new_name}")
                                cmd = ['ffmpeg', '-i', video_file, '-c', 'copy', new_name, '-y']
                                subprocess.run(cmd, capture_output=True)
                                if os.path.exists(new_name):
                                    os.remove(video_file)
                            else:
                                print(f"  Renaming {video_file} to {new_name}")
                                os.rename(video_file, new_name)
                        else:
                            print(f"  Target {new_name} already exists, removing {video_file}")
                            os.remove(video_file)
            
            # Remove the info.json file
            os.remove(info_file)
            
        except Exception as e:
            print(f"  Error processing {info_file}: {e}")
    
    print("Cleanup completed.")

def process_videos():
    """Main processing function with real ID support"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, 'FVC_dup.csv')
    checkpoint_file = os.path.join(base_dir, 'data.json')
    video_dir = os.path.join(base_dir, 'video')
    
    # Clean up old format files first
    cleanup_old_files(video_dir)
    
    # Load existing checkpoint data
    processed_data = load_checkpoint(checkpoint_file)
    processed_urls_map = {item.get('url'): item for item in processed_data}
    
    print(f"Starting processing. Found {len(processed_data)} already processed videos.")
    
    # Read CSV file
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Total videos in CSV: {len(rows)}")
    
    processed_count = 0
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for i, row in enumerate(rows):
        cascade_id = row['cascade_id']
        video_url = row['video_url']
        label = row['label']
        
        print(f"\n[{i+1}/{len(rows)}] Processing cascade:{cascade_id}")
        print(f"  URL: {video_url}")
        
        # Check if we should retry this video
        existing_item = processed_urls_map.get(video_url)
        if existing_item and not should_retry_video(existing_item):
            if existing_item.get('processing_status') == 'downloaded':
                print(f"  → Skipping (already downloaded)")
            else:
                print(f"  → Skipping (critical error: {existing_item.get('error', 'unknown')[:50]}...)")
            skipped_count += 1
            continue
        elif existing_item:
            print(f"  → Retrying (previous attempt failed with temporary error)")
        
        # Extract metadata first to get real video ID
        print(f"  Extracting metadata...")
        metadata = extract_metadata_with_ytdlp(video_url)
        
        real_id = metadata['real_id']
        print(f"  Real ID: {real_id}")
        
        # Initialize video data with real ID
        video_data = {
            'video_id': real_id,  # Use real video ID
            'description': 'placeholder',
            'annotation': label,
            'user_certify': 0,
            'user_description': 'placeholder',
            'publish_time': int(time.time() * 1000),
            'event': cascade_id,  # Use cascade_id as event
            'url': video_url,
            'processing_status': 'failed',
            'error': None
        }
        
        if metadata['success']:
            # Update video data with extracted metadata
            video_data.update({
                'description': metadata.get('title', 'placeholder'),
                'user_description': metadata.get('uploader', 'placeholder'),
                'publish_time': format_timestamp(metadata.get('timestamp')),
                'processing_status': 'metadata_extracted'
            })
            
            # Check if video file already exists
            expected_file = f'{video_dir}/{real_id}.mp4'
            if os.path.exists(expected_file):
                print(f"  ✓ Video file already exists: {real_id}.mp4")
                video_data['processing_status'] = 'downloaded'
                success_count += 1
            else:
                # Attempt to download video
                print(f"  Downloading video...")
                download_result = download_video_mp4(video_url, video_dir, real_id)
                
                if download_result['success']:
                    video_data['processing_status'] = 'downloaded'
                    success_count += 1
                    print(f"  ✓ Successfully downloaded: {real_id}.mp4")
                else:
                    video_data['error'] = download_result['error']
                    failed_count += 1
                    print(f"  ✗ Download failed: {download_result['error'][:100]}...")
        else:
            video_data['error'] = metadata['error']
            failed_count += 1
            print(f"  ✗ Metadata extraction failed: {metadata['error'][:100]}...")
        
        # Add to processed data or update existing entry
        if existing_item:
            # Update existing entry
            for i, item in enumerate(processed_data):
                if item.get('url') == video_url:
                    processed_data[i] = video_data
                    break
        else:
            # Add new entry
            processed_data.append(video_data)
        
        # Save checkpoint after each video
        save_checkpoint(checkpoint_file, processed_data)
        processed_count += 1
        
        # Progress summary
        progress_pct = (i + 1) / len(rows) * 100
        success_rate = success_count / processed_count * 100 if processed_count > 0 else 0
        print(f"  Progress: {i+1}/{len(rows)} ({progress_pct:.1f}%) | New: {processed_count} | Success: {success_count} | Failed: {failed_count} | Rate: {success_rate:.1f}%")
        
        # Small delay to be nice to servers
        time.sleep(1)
    
    print(f"\nProcessing complete!")
    print(f"Total processed (new): {processed_count}")
    print(f"Total skipped (existing): {skipped_count}")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Total in database: {len(processed_data)}")
    if processed_count > 0:
        print(f"Success rate: {success_count/processed_count*100:.1f}%")

if __name__ == "__main__":
    try:
        process_videos()
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user. Progress has been saved.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)