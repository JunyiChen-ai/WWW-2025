#!/usr/bin/env python3
"""
Extract video descriptions and temporal evolution from videos using LLM
Supports multiple datasets with flexible field mapping
"""

import os
import json
import base64
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time
from tqdm import tqdm
from PIL import Image
import io
import signal
import sys
import atexit
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_description_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class VideoDescriptionExtractor:
    def __init__(self, 
                 data_dir: str,
                 api_key: str = None,
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.3,
                 max_tokens: int = 800,
                 rate_limit_delay: float = 0.5,
                 resume: bool = True,
                 max_concurrent: int = 10):
        """
        Initialize the video description extractor
        
        Args:
            data_dir: Path to the data directory containing video data and images
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o-mini)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            rate_limit_delay: Delay between API calls in seconds
            resume: Whether to resume from checkpoint
            max_concurrent: Maximum number of concurrent API calls
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rate_limit_delay = rate_limit_delay
        self.resume = resume
        self.max_concurrent = max_concurrent
        
        # Data paths
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        # Output file paths
        self.output_file = self.data_dir / "llm_video_descriptions.jsonl"
        self.checkpoint_file = self.data_dir / "llm_checkpoint.json"
        self.temp_file = self.data_dir / "llm_temp_results.jsonl"
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "start_time": datetime.now().isoformat()
        }
        
        # Processing state
        self.processed_videos = set()
        self.all_results = []
        
        # Shutdown flag
        self.shutdown_requested = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Load checkpoint if exists
        if self.resume:
            self.load_checkpoint()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("\n=== Shutdown requested, saving progress... ===")
        self.shutdown_requested = True
    
    def load_checkpoint(self):
        """Load checkpoint if exists"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                self.processed_videos = set(checkpoint.get('processed_videos', []))
                self.stats = checkpoint.get('stats', self.stats)
                logger.info(f"Resumed from checkpoint: {len(self.processed_videos)} videos already processed")
        
        # Load temp results
        if self.temp_file.exists():
            with open(self.temp_file, 'r', encoding='utf-8') as f:
                for line in f:
                    result = json.loads(line)
                    self.all_results.append(result)
                logger.info(f"Loaded {len(self.all_results)} existing results")
    
    def save_checkpoint(self):
        """Save current progress to checkpoint"""
        checkpoint = {
            'processed_videos': list(self.processed_videos),
            'stats': self.stats,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        logger.info(f"Checkpoint saved: {len(self.processed_videos)} videos processed")
    
    def save_temp_result(self, result: Dict):
        """Save result to temporary file immediately"""
        with open(self.temp_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        self.all_results.append(result)
        self.processed_videos.add(result['video_id'])
    
    def load_video_data(self, data_filename: str = "data.json") -> List[Dict]:
        """Load video data from data file (supports both .json and .jsonl)"""
        
        # Try different possible data files
        possible_files = [
            self.data_dir / data_filename,
            self.data_dir / "data_complete.jsonl", 
            self.data_dir / "data.json",
            self.data_dir / "data.jsonl"
        ]
        
        data_file = None
        for file_path in possible_files:
            if file_path.exists():
                data_file = file_path
                break
        
        if not data_file:
            raise FileNotFoundError(f"No data file found. Tried: {[str(f) for f in possible_files]}")
        
        video_data = []
        logger.info(f"Loading video data from: {data_file}")
        
        # Handle both JSON and JSONL formats
        if data_file.suffix == '.jsonl':
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        video_data.append(item)
        else:
            with open(data_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content.startswith('['):
                    # JSON array format
                    video_data = json.loads(content)
                else:
                    # JSONL format in .json file (each line is a JSON object)
                    for line in content.split('\n'):
                        if line.strip():
                            video_data.append(json.loads(line))
        
        logger.info(f"Loaded {len(video_data)} videos")
        return video_data
    
    def load_transcript(self, video_id: str) -> str:
        """Load transcript for a video"""
        transcript_file = self.data_dir / "transcript.jsonl"
        
        if not transcript_file.exists():
            return ""
        
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Handle both 'vid' and 'video_id' field names
                    vid_key = 'vid' if 'vid' in item else 'video_id'
                    if item.get(vid_key) == video_id:
                        return item.get('transcript', '')
        
        return ""
    
    def prepare_composite_frames(self, video_id: str) -> List[str]:
        """Load and encode composite frames as base64"""
        # Try different possible frame directories
        frame_dirs = [
            self.data_dir / "quads_4",
            self.data_dir / "frames_4", 
            self.data_dir / "composite_frames"
        ]
        
        quads_dir = None
        for dir_path in frame_dirs:
            if dir_path.exists():
                quads_dir = dir_path
                break
        
        if not quads_dir:
            logger.warning(f"No composite frames directory found for {video_id}")
            return []
        
        encoded_images = []
        
        # Try different naming conventions
        for i in range(4):
            possible_names = [
                f"{video_id}_quad_{i}.jpg",
                f"{video_id}_frame_{i}.jpg", 
                f"{video_id}_{i}.jpg"
            ]
            
            image_path = None
            for name in possible_names:
                path = quads_dir / name
                if path.exists():
                    image_path = path
                    break
            
            if image_path:
                try:
                    with open(image_path, 'rb') as img_file:
                        img_data = img_file.read()
                        # Optionally resize if too large
                        img = Image.open(io.BytesIO(img_data))
                        if img.width > 1024 or img.height > 1024:
                            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                            buffer = io.BytesIO()
                            img.save(buffer, format='JPEG', quality=85)
                            img_data = buffer.getvalue()
                        
                        encoded = base64.b64encode(img_data).decode('utf-8')
                        encoded_images.append(encoded)
                except Exception as e:
                    logger.warning(f"Failed to load image {image_path}: {e}")
        
        return encoded_images
    
    def format_timestamp(self, timestamp_ms: Optional[int]) -> str:
        """Convert millisecond timestamp to YYYY-MM-DD format"""
        if timestamp_ms:
            try:
                return datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d')
            except:
                pass
        return ""
    
    def create_prompt(self, video_data: Dict, transcript: str) -> str:
        """Create the prompt for video content analysis"""
        title = video_data.get('title', '') or video_data.get('description', '')
        
        # Handle both keywords and event fields flexibly
        keywords_or_event = ""
        if 'keywords' in video_data:
            keywords_or_event = video_data.get('keywords', '')
        elif 'event' in video_data:
            keywords_or_event = video_data.get('event', '')
        
        publish_time = self.format_timestamp(video_data.get('publish_time', 0) or video_data.get('publish_time_norm', 0))
        
        prompt = f"""Analyze the content of this short video and extract video description information.

Video Information:
- Title: {title}
- Keywords: {keywords_or_event if keywords_or_event else "No keywords/event provided"}
- Audio Transcript: {transcript if transcript else "No audio transcript"}
- Publish Time: {publish_time}

Video Frame Temporal Information:
This video contains 4 composite frame images, each containing 4 consecutive frames (2x2 grid layout):
- 1st composite frame (frames 0-3, beginning part of video)
- 2nd composite frame (frames 4-7, early-middle part of video)
- 3rd composite frame (frames 8-11, late-middle part of video)
- 4th composite frame (frames 12-15, ending part of video)

Please extract the following information (respond in English):

1. Video Description:
   Briefly describe what this video is about (50-100 words)

2. Temporal Evolution:
   Describe how the content evolves across the 4 composite frames

Return in JSON format:
{{
  "description": "Detailed description of the video",
  "temporal_evolution": "Description of temporal evolution"
}}"""
        
        return prompt
    
    async def extract_from_video(self, video_data: Dict, images: List[str]) -> Optional[Dict]:
        """Extract description and temporal evolution from a single video"""
        video_id = video_data.get('video_id')
        
        try:
            # Load transcript
            transcript = self.load_transcript(video_id)
            
            # Create prompt
            prompt = self.create_prompt(video_data, transcript)
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Add images if available
            for i, img_base64 in enumerate(images):
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                        "detail": "high"
                    }
                })
            
            # Call API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            # Create extraction result
            extraction = {
                "video_id": video_id,
                "annotation": video_data.get('annotation'),
                "title": video_data.get('title', '') or video_data.get('description', ''),
                "description": result.get("description", ""),
                "temporal_evolution": result.get("temporal_evolution", ""),
                "metadata": {
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "has_images": len(images) > 0,
                    "has_transcript": bool(transcript)
                }
            }
            
            # Add keywords/event field if present
            if 'keywords' in video_data:
                extraction["keywords"] = video_data.get('keywords', '')
            elif 'event' in video_data:
                extraction["event"] = video_data.get('event', '')
            
            # Update statistics
            self.stats["successful"] += 1
            
            return extraction
            
        except Exception as e:
            logger.error(f"Failed to process video {video_id}: {e}")
            self.stats["failed"] += 1
            return None
    
    async def process_videos_concurrent(self, videos: List[Dict]) -> List[Dict]:
        """Process videos with controlled concurrency"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        checkpoint_interval = 20  # Save checkpoint every N videos
        
        async def process_with_semaphore(video_data):
            async with semaphore:
                video_id = video_data.get('video_id')
                
                # Skip if already processed
                if video_id in self.processed_videos:
                    return None
                
                # Check for shutdown
                if self.shutdown_requested:
                    return None
                
                images = self.prepare_composite_frames(video_id)
                result = await self.extract_from_video(video_data, images)
                
                # Save immediately if successful
                if result:
                    self.save_temp_result(result)
                    
                    # Save checkpoint periodically
                    if len(self.processed_videos) % checkpoint_interval == 0:
                        self.save_checkpoint()
                
                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)
                
                # Update progress
                self.stats["total_processed"] += 1
                
                return result
        
        # Filter videos that haven't been processed
        videos_to_process = [v for v in videos if v.get('video_id') not in self.processed_videos]
        logger.info(f"Processing {len(videos_to_process)} videos (skipping {len(videos) - len(videos_to_process)} already processed)")
        
        if not videos_to_process:
            logger.info("All videos already processed!")
            return self.all_results
        
        # Process in batches for better checkpoint management
        batch_size = 50
        all_results = []
        
        for batch_start in range(0, len(videos_to_process), batch_size):
            if self.shutdown_requested:
                logger.info("Processing interrupted. Saving checkpoint...")
                self.save_checkpoint()
                break
                
            batch_end = min(batch_start + batch_size, len(videos_to_process))
            batch_videos = videos_to_process[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}: videos {batch_start+1}-{batch_end} of {len(videos_to_process)}")
            
            # Create tasks for this batch
            with tqdm(total=len(batch_videos), desc=f"Batch {batch_start//batch_size + 1}") as pbar:
                tasks = []
                for video in batch_videos:
                    tasks.append(process_with_semaphore(video))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results and update progress bar
                for video, result in zip(batch_videos, results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to process video {video['video_id']}: {result}")
                        self.stats["failed"] += 1
                    elif result:
                        all_results.append(result)
                    
                    pbar.update(1)
            
            # Save checkpoint after each batch
            self.save_checkpoint()
            logger.info(f"Batch complete. Processed {len(self.processed_videos)}/{len(videos)} total videos")
        
        return self.all_results
    
    def save_final_results(self):
        """Save all results to final output file"""
        logger.info(f"Saving {len(self.all_results)} results to {self.output_file}")
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for result in self.all_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info(f"Results saved to: {self.output_file}")
    
    def cleanup_temp_files(self):
        """Clean up temporary files after successful completion"""
        if self.temp_file.exists():
            self.temp_file.unlink()
            logger.info("Cleaned up temporary result file")
        
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Cleaned up checkpoint file")
    
    async def run(self, sample_size: Optional[int] = None, data_filename: str = "data.json"):
        """Run the video description extraction pipeline"""
        logger.info("Starting video description extraction pipeline")
        logger.info(f"Using model: {self.model}")
        logger.info(f"Max concurrent API calls: {self.max_concurrent}")
        logger.info(f"Rate limit delay: {self.rate_limit_delay}s")
        
        try:
            # Load video data
            videos = self.load_video_data(data_filename)
            
            # Sample if requested
            if sample_size:
                videos = videos[:sample_size]
                logger.info(f"Processing sample of {sample_size} videos")
            else:
                logger.info(f"Processing all {len(videos)} videos")
            
            # Update stats
            self.stats["total_videos"] = len(videos)
            
            # Process videos with concurrency
            results = await self.process_videos_concurrent(videos)
            
            # Check if user requested shutdown
            if self.shutdown_requested:
                logger.info("Processing interrupted by user. Partial results saved to temporary file.")
                logger.info("Run again to continue processing from where it stopped.")
                return
            
            # Save final results
            logger.info("All videos processed successfully!")
            self.save_final_results()
            
            # Clean up temporary files
            self.cleanup_temp_files()
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            # Save checkpoint on error
            self.save_checkpoint()
            logger.info("Checkpoint saved. Run again to continue from where it stopped.")
            raise
        
        # Print final statistics
        self.print_statistics()
    
    def print_statistics(self):
        """Print processing statistics"""
        logger.info("\n=== Processing Statistics ===")
        logger.info(f"Total videos: {self.stats.get('total_videos', 0)}")
        logger.info(f"Total processed: {self.stats['total_processed']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        
        if self.stats['successful'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_processed']) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        
        # Calculate processing time
        start_time = datetime.fromisoformat(self.stats['start_time'])
        duration = datetime.now() - start_time
        logger.info(f"Processing time: {duration}")
        
        logger.info(f"Output file: {self.output_file}")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract video descriptions and temporal evolution using LLM")
    parser.add_argument("--dataset", type=str, default="FakeTT", help="Dataset name (default: FakeTT)")
    parser.add_argument("--sample", type=int, help="Process only N samples for testing")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use (default: gpt-4o-mini)")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature (default: 0)")
    parser.add_argument("--max-tokens", type=int, default=800, help="Maximum tokens in response (default: 800)")
    parser.add_argument("--rate-limit", type=float, default=0.5, help="Delay between API calls in seconds (default: 0.5)")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Maximum concurrent API calls (default: 10)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore checkpoint")
    parser.add_argument("--data-file", type=str, default="data.json", help="Data filename to load (default: data.json)")
    
    args = parser.parse_args()
    
    # Auto-construct data directory path
    script_dir = Path(__file__).parent.parent  # Go up from preprocess/ to WWW-2025/
    data_dir = script_dir / "data" / args.dataset
    
    logger.info(f"Using dataset: {args.dataset}")
    logger.info(f"Data directory: {data_dir}")
    
    # Create extractor
    try:
        extractor = VideoDescriptionExtractor(
            data_dir=str(data_dir),
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            rate_limit_delay=args.rate_limit,
            resume=not args.no_resume,
            max_concurrent=args.max_concurrent
        )
    except Exception as e:
        logger.error(f"Failed to initialize extractor: {e}")
        return
    
    # Run extraction
    try:
        await extractor.run(sample_size=args.sample, data_filename=args.data_file)
        logger.info("Extraction completed successfully!")
    except KeyboardInterrupt:
        logger.info("Extraction interrupted by user")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())