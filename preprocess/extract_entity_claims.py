#!/usr/bin/env python3
"""
Extract entities and factual claims from FakeSV videos using GPT-4o-mini
Phase 1: Process videos with annotation='真' (true/real videos)
Phase 2: Process videos with annotation='假' (fake videos) with similarity matching
"""

import os
import json
import base64
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
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
import torch
import numpy as np
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('entity_claim_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class EntityClaimExtractor:
    def __init__(self, 
                 api_key: str = None,
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.3,
                 max_tokens: int = 800,
                 rate_limit_delay: float = 0.5,
                 resume: bool = True):
        """
        Initialize the entity-claim extractor
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o-mini)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            rate_limit_delay: Delay between API calls in seconds
            resume: Whether to resume from checkpoint
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
        
        # Data paths
        self.data_dir = Path("data/FakeSV")
        self.output_dir = self.data_dir / "entity_claims"
        self.output_dir.mkdir(exist_ok=True)
        
        # Checkpoint file
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.temp_extractions_file = self.output_dir / "temp_extractions.jsonl"
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_entities": 0,
            "total_claims": 0
        }
        
        # Extraction results
        self.all_extractions = []
        self.processed_videos = set()
        
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
        
        # Load temp extractions
        if self.temp_extractions_file.exists():
            with open(self.temp_extractions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    extraction = json.loads(line)
                    self.all_extractions.append(extraction)
                logger.info(f"Loaded {len(self.all_extractions)} existing extractions")
    
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
    
    def save_temp_extraction(self, extraction: Dict):
        """Save extraction to temporary file immediately"""
        with open(self.temp_extractions_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(extraction, ensure_ascii=False) + '\n')
        self.all_extractions.append(extraction)
        self.processed_videos.add(extraction['video_id'])
    
    def load_video_data(self, filter_label: str = '真') -> List[Dict]:
        """Load and filter video data"""
        data_file = self.data_dir / "data_complete_orig.jsonl"
        filtered_data = []
        
        logger.info(f"Loading videos with annotation='{filter_label}'")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if item.get('annotation') == filter_label:
                    filtered_data.append(item)
        
        logger.info(f"Loaded {len(filtered_data)} videos with label '{filter_label}'")
        return filtered_data
    
    def load_transcript(self, video_id: str) -> str:
        """Load transcript for a video"""
        transcript_file = self.data_dir / "transcript.jsonl"
        
        if not transcript_file.exists():
            return ""
        
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if item.get('vid') == video_id:
                    return item.get('transcript', '')
        
        return ""
    
    def prepare_composite_frames(self, video_id: str) -> List[str]:
        """Load and encode composite frames as base64"""
        quads_dir = self.data_dir / "quads_4"
        encoded_images = []
        
        for i in range(4):
            image_path = quads_dir / f"{video_id}_quad_{i}.jpg"
            if image_path.exists():
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
    
    def format_timestamp(self, timestamp_ms: int) -> str:
        """Convert millisecond timestamp to YYYY-MM-DD format"""
        if timestamp_ms:
            return datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d')
        return ""
    
    def create_prompt(self, video_data: Dict, transcript: str) -> str:
        """Create the prompt for GPT-4o-mini"""
        title = video_data.get('title', '')
        keywords = video_data.get('keywords', '')
        label = video_data.get('annotation', '')
        publish_time = self.format_timestamp(video_data.get('publish_time_norm', 0))
        
        prompt = f"""分析这个短视频的内容，提取知识图谱信息。

视频信息：
- 标题：{title}
- 关键词：{keywords}  
- 音频转录：{transcript if transcript else "无音频转录"}
- 标签：{label}
- 发布时间：{publish_time}

视频帧时序信息：
这个视频包含4张复合帧图像，每张包含4个连续帧（2x2网格布局）：
- 第1张复合帧（帧0-3，视频开始部分）：
  左上(1,1): 帧0  |  右上(1,2): 帧1
  左下(2,1): 帧2  |  右下(2,2): 帧3
- 第2张复合帧（帧4-7，视频前中部分）：
  左上(1,1): 帧4  |  右上(1,2): 帧5
  左下(2,1): 帧6  |  右下(2,2): 帧7
- 第3张复合帧（帧8-11，视频后中部分）：
  左上(1,1): 帧8  |  右上(1,2): 帧9
  左下(2,1): 帧10 |  右下(2,2): 帧11
- 第4张复合帧（帧12-15，视频结束部分）：
  左上(1,1): 帧12 |  右上(1,2): 帧13
  左下(2,1): 帧14 |  右下(2,2): 帧15

注意：如果音频转录看起来没有意义，可能是因为音频只是背景音乐而没有人声。

请提取以下信息（用中文回答）：

1. 实体及其相关声称（Entity-Claim Mapping）：
   为每个主要实体列出与之直接相关的事实性声称
   
   每个声称必须遵循5W原则（Who/What/When/Where/Why），包含具体信息：
   - WHO（谁）：涉及的人物或组织
   - WHAT（什么）：发生了什么事件或行动
   - WHEN（何时）：具体时间或时间范围
   - WHERE（何地）：具体地点
   - WHY/HOW（为何/如何）：原因或方式（如适用）
   
   示例：
   - 错误："消防员牺牲" 
   - 正确："2019年3月21日，两名消防员在上海某化工厂爆炸事故救援中牺牲"

2. 视频描述（Description）：
   简要描述这个视频讲述了什么（50-100字）

3. 时序演变（Temporal Evolution）：
   描述内容如何在4个复合帧中演变

返回JSON格式：
{{
  "entity_claims": {{
    "实体1": [
      "包含5W信息的具体声称1",
      "包含5W信息的具体声称2"
    ],
    "实体2": [
      "包含5W信息的具体声称3"
    ]
  }},
  "description": "视频描述",
  "temporal_evolution": "时序演变描述"
}}"""
        
        return prompt
    
    async def extract_from_video(self, video_data: Dict, images: List[str]) -> Optional[Dict]:
        """Extract entities and claims from a single video"""
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
            
            # Process entity-claim mapping
            entity_claims = result.get("entity_claims", {})
            
            # Extract entities list and all claims
            entities = list(entity_claims.keys())
            all_claims = []
            for entity, claims in entity_claims.items():
                all_claims.extend(claims)
            
            # Add metadata
            extraction = {
                "video_id": video_id,
                "annotation": video_data.get('annotation'),
                "title": video_data.get('title'),
                "entity_claims": entity_claims,
                "entities": entities,
                "claims": all_claims,
                "description": result.get("description", ""),
                "temporal_evolution": result.get("temporal_evolution", ""),
                "metadata": {
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "has_images": len(images) > 0,
                    "has_transcript": bool(transcript)
                }
            }
            
            # Update statistics
            self.stats["successful"] += 1
            self.stats["total_entities"] += len(entities)
            self.stats["total_claims"] += len(all_claims)
            
            return extraction
            
        except Exception as e:
            logger.error(f"Failed to process video {video_id}: {e}")
            self.stats["failed"] += 1
            return None
    
    async def process_batch(self, videos: List[Dict], batch_size: int = 10, max_concurrent: int = 10) -> List[Dict]:
        """Process videos with controlled concurrency and checkpoint support"""
        semaphore = asyncio.Semaphore(max_concurrent)
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
                    self.save_temp_extraction(result)
                    
                    # Save checkpoint periodically
                    if len(self.processed_videos) % checkpoint_interval == 0:
                        self.save_checkpoint()
                
                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)
                
                # Update progress
                self.stats["total_processed"] += 1
                if self.stats["total_processed"] % 10 == 0:
                    logger.info(f"Processed {self.stats['total_processed']}/{len(videos)} videos")
                
                return result
        
        # Filter videos that haven't been processed
        videos_to_process = [v for v in videos if v.get('video_id') not in self.processed_videos]
        logger.info(f"Skipping {len(videos) - len(videos_to_process)} already processed videos")
        
        # Process all videos with controlled concurrency
        tasks = []
        for video in videos_to_process:
            if self.shutdown_requested:
                break
            tasks.append(process_with_semaphore(video))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and None results
            for r in results:
                if isinstance(r, Exception):
                    logger.error(f"Task failed with exception: {r}")
                    self.stats["failed"] += 1
        
        return self.all_extractions
    
    def normalize_entity(self, entity: str) -> str:
        """Normalize entity names to reduce redundancy"""
        entity = entity.strip()
        
        # Define normalization rules
        normalization_rules = [
            # People names with titles
            (r'消防员(.+)', r'\1'),  # 消防员孙络络 -> 孙络络
            (r'消防战士(.+)', r'\1'),  # 消防战士孙络络 -> 孙络络
            (r'医生(.+)', r'\1'),
            (r'护士(.+)', r'\1'),
            (r'警察(.+)', r'\1'),
            (r'警官(.+)', r'\1'),
            
            # Location normalization
            (r'(.+)省(.+)市', r'\2'),  # 江苏省徐州市 -> 徐州市
            (r'(.+)市(.+)区', r'\1市\2区'),  # Ensure consistent format
            
            # Organization normalization
            (r'(.+)公司', r'\1'),
            (r'(.+)集团', r'\1'),
            (r'(.+)有限公司', r'\1'),
        ]
        
        import re
        normalized = entity
        for pattern, replacement in normalization_rules:
            normalized = re.sub(pattern, replacement, normalized)
        
        # Manual mapping for common variations
        entity_mapping = {
            '新冠肺炎': '新冠疫情',
            'COVID-19': '新冠疫情',
            '新型冠状病毒': '新冠疫情',
            '武汉肺炎': '新冠疫情',
        }
        
        if normalized in entity_mapping:
            normalized = entity_mapping[normalized]
        
        return normalized
    
    def merge_similar_entities(self, entity_kb: Dict) -> Dict:
        """Merge similar entities based on string similarity"""
        from difflib import SequenceMatcher
        
        merged_kb = {}
        processed = set()
        
        entities = list(entity_kb.keys())
        
        for i, entity1 in enumerate(entities):
            if entity1 in processed:
                continue
            
            # Normalize the entity
            normalized_entity = self.normalize_entity(entity1)
            
            # Start with this entity's data
            if normalized_entity not in merged_kb:
                merged_kb[normalized_entity] = entity_kb[entity1].copy()
                merged_kb[normalized_entity]["aliases"] = [entity1]
            else:
                # Merge with existing normalized entity
                self.merge_entity_data(merged_kb[normalized_entity], entity_kb[entity1])
                merged_kb[normalized_entity]["aliases"].append(entity1)
            
            processed.add(entity1)
            
            # Look for similar entities to merge
            for j, entity2 in enumerate(entities[i+1:], i+1):
                if entity2 in processed:
                    continue
                
                normalized_entity2 = self.normalize_entity(entity2)
                
                # Check if they normalize to the same thing
                if normalized_entity == normalized_entity2:
                    self.merge_entity_data(merged_kb[normalized_entity], entity_kb[entity2])
                    merged_kb[normalized_entity]["aliases"].append(entity2)
                    processed.add(entity2)
                # Check string similarity for potential merging
                elif SequenceMatcher(None, normalized_entity, normalized_entity2).ratio() > 0.85:
                    # High similarity, consider merging
                    self.merge_entity_data(merged_kb[normalized_entity], entity_kb[entity2])
                    merged_kb[normalized_entity]["aliases"].append(entity2)
                    processed.add(entity2)
        
        return merged_kb
    
    def merge_entity_data(self, target: Dict, source: Dict):
        """Merge source entity data into target"""
        # Merge claims
        for claim in source.get("correct_claims", []):
            # Check if claim already exists
            existing = False
            for existing_claim in target["correct_claims"]:
                if existing_claim["claim"] == claim["claim"]:
                    # Merge video_ids
                    for vid in claim["video_ids"]:
                        if vid not in existing_claim["video_ids"]:
                            existing_claim["video_ids"].append(vid)
                    existing = True
                    break
            if not existing:
                target["correct_claims"].append(claim)
        
        # Merge video counts
        for label in ["真", "假", "辟谣"]:
            target["video_count"][label] += source["video_count"].get(label, 0)
        
        # Update timestamps
        if source.get("first_seen") and (not target.get("first_seen") or source["first_seen"] < target["first_seen"]):
            target["first_seen"] = source["first_seen"]
        if source.get("last_seen") and (not target.get("last_seen") or source["last_seen"] > target["last_seen"]):
            target["last_seen"] = source["last_seen"]
    
    async def generate_entity_description(self, entity: str, claims: List[str]) -> str:
        """Generate entity description using LLM based on all claims"""
        if not claims:
            return "暂无相关声称"
        
        prompt = f"""基于以下关于"{entity}"的事实性声称，生成一个简洁、准确的描述（50-100字）：

声称列表：
{chr(10).join(f'- {claim}' for claim in claims)}  # Include all claims

要求：
1. 综合所有声称中的关键信息
2. 突出最重要的事实（时间、地点、事件）
3. 保持客观、准确
4. 不要简单罗列声称，而是形成连贯的描述

返回格式（纯文本，不要JSON）：
[实体的综合描述]"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use gpt-4o-mini for better quality descriptions
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            
            description = response.choices[0].message.content.strip()
            return description
        except Exception as e:
            logger.warning(f"Failed to generate description for {entity}: {e}")
            # Fallback to simple concatenation
            return f"基于以往真实视频中的正确声称：{'; '.join(claims[:3])}"
    
    async def enhance_entity_descriptions(self, entity_kb: Dict):
        """Generate LLM-based descriptions for all entities with checkpoint support"""
        logger.info("Generating entity descriptions using LLM...")
        
        # Load or create description checkpoint
        desc_checkpoint_file = self.output_dir / "description_checkpoint.json"
        processed_entities = set()
        
        if desc_checkpoint_file.exists():
            with open(desc_checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                processed_entities = set(checkpoint.get('processed_entities', []))
                logger.info(f"Resuming description generation: {len(processed_entities)} entities already processed")
        
        # Save partial KB periodically
        save_interval = 10  # Save every N entities
        entities_to_process = [(entity, data) for entity, data in entity_kb.items() 
                               if entity not in processed_entities]
        
        if not entities_to_process:
            logger.info("All entity descriptions already generated")
            return
        
        logger.info(f"Generating descriptions for {len(entities_to_process)} entities...")
        
        # Use tqdm for progress tracking
        with tqdm(total=len(entities_to_process), desc="Generating descriptions", 
                  initial=len(processed_entities)) as pbar:
            
            for idx, (entity, data) in enumerate(entities_to_process):
                if self.shutdown_requested:
                    logger.info("Description generation interrupted. Saving progress...")
                    break
                
                # Generate correct description
                correct_claims = [c["claim"] for c in data.get("correct_claims", [])]
                if correct_claims:
                    try:
                        data["correct_description"] = await self.generate_entity_description(entity, correct_claims)
                        await asyncio.sleep(0.5)  # Rate limiting
                    except Exception as e:
                        logger.warning(f"Failed to generate description for {entity}: {e}")
                        data["correct_description"] = f"基于以往真实视频中的正确声称：{'; '.join(correct_claims[:3])}"
                else:
                    data["correct_description"] = "暂无正确声称"
                
                # False description will be generated when processing fake videos
                data["false_description"] = "[待后续从假新闻视频中提取]"
                
                # Mark as processed
                processed_entities.add(entity)
                pbar.update(1)
                
                # Save checkpoint periodically
                if (idx + 1) % save_interval == 0:
                    # Save checkpoint
                    checkpoint = {
                        'processed_entities': list(processed_entities),
                        'timestamp': datetime.now().isoformat(),
                        'total_entities': len(entity_kb)
                    }
                    with open(desc_checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
                    
                    # Save partial KB
                    partial_kb_file = self.output_dir / "entity_knowledge_base_partial.json"
                    with open(partial_kb_file, 'w', encoding='utf-8') as f:
                        json.dump(entity_kb, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"Checkpoint saved: {len(processed_entities)}/{len(entity_kb)} entities processed")
        
        # Final checkpoint save
        if processed_entities:
            checkpoint = {
                'processed_entities': list(processed_entities),
                'timestamp': datetime.now().isoformat(),
                'total_entities': len(entity_kb)
            }
            with open(desc_checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            
            # Remove checkpoint file if all entities processed
            if len(processed_entities) == len(entity_kb):
                desc_checkpoint_file.unlink()
                logger.info("All descriptions generated, checkpoint removed")
                
                # Remove partial KB file
                partial_kb_file = self.output_dir / "entity_knowledge_base_partial.json"
                if partial_kb_file.exists():
                    partial_kb_file.unlink()
    
    # ============== FAKE NEWS PROCESSING METHODS ==============
    
    def create_fake_description_prompt(self, video_data: Dict, transcript: str) -> str:
        """Create prompt for extracting only description and temporal evolution from fake videos"""
        title = video_data.get('title', '')
        keywords = video_data.get('keywords', '')
        label = video_data.get('annotation', '')
        publish_time = self.format_timestamp(video_data.get('publish_time_norm', 0))
        
        prompt = f"""分析这个短视频的内容，提取视频描述信息。

视频信息：
- 标题：{title}
- 关键词：{keywords}  
- 音频转录：{transcript if transcript else "无音频转录"}
- 标签：{label}
- 发布时间：{publish_time}

视频帧时序信息：
这个视频包含4张复合帧图像，每张包含4个连续帧（2x2网格布局）：
- 第1张复合帧（帧0-3，视频开始部分）
- 第2张复合帧（帧4-7，视频前中部分）
- 第3张复合帧（帧8-11，视频后中部分）
- 第4张复合帧（帧12-15，视频结束部分）

请提取以下信息（用中文回答）：

1. 视频描述（Description）：
   简要描述这个视频讲述了什么（50-100字）

2. 时序演变（Temporal Evolution）：
   描述内容如何在4个复合帧中演变

注意：不要提取任何实体或声称，只需要描述视频内容。

返回JSON格式：
{{
  "description": "视频的详细描述",
  "temporal_evolution": "时序演变描述"
}}"""
        
        return prompt
    
    async def extract_fake_description(self, video_data: Dict, images: List[str]) -> Optional[Dict]:
        """Extract only description and temporal evolution from fake videos"""
        video_id = video_data.get('video_id')
        
        try:
            # Load transcript
            transcript = self.load_transcript(video_id)
            
            # Create prompt
            prompt = self.create_fake_description_prompt(video_data, transcript)
            
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
            
            return {
                "video_id": video_id,
                "annotation": video_data.get('annotation'),
                "title": video_data.get('title'),
                "keywords": video_data.get('keywords'),
                "description": result.get("description", ""),
                "temporal_evolution": result.get("temporal_evolution", ""),
                "metadata": {
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "has_images": len(images) > 0,
                    "has_transcript": bool(transcript)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to process fake video {video_id}: {e}")
            return None
    
    def encode_text_with_clip(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts using Chinese-CLIP model"""
        logger.info("Loading Chinese-CLIP model for text encoding...")
        
        # Load model and processor
        model_name = "OFA-Sys/chinese-clip-vit-large-patch14"
        processor = ChineseCLIPProcessor.from_pretrained(model_name)
        model = ChineseCLIPModel.from_pretrained(model_name)
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
                batch_texts = texts[i:i+batch_size]
                
                # Process texts
                inputs = processor(text=batch_texts, padding=True, truncation=True, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get text features
                text_features = model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize
                
                all_embeddings.append(text_features.cpu().numpy())
        
        # Concatenate all embeddings
        embeddings = np.vstack(all_embeddings)
        logger.info(f"Encoded {len(texts)} texts into {embeddings.shape} embeddings")
        
        return embeddings
    
    def find_similar_true_news(self, fake_descriptions: List[Dict], true_extractions: List[Dict], top_k: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """Find most similar true news for each fake news using Chinese-CLIP"""
        logger.info("Finding similar true news for fake videos...")
        
        # Prepare texts for encoding
        fake_texts = []
        fake_video_ids = []
        
        for desc in fake_descriptions:
            text = f"{desc['title']} {desc.get('keywords', '')} {desc['description']} {desc['temporal_evolution']}"
            fake_texts.append(text)
            fake_video_ids.append(desc['video_id'])
        
        true_texts = []
        true_video_ids = []
        
        for ext in true_extractions:
            text = f"{ext['title']} {ext.get('keywords', '')} {ext.get('description', '')} {ext.get('temporal_evolution', '')}"
            true_texts.append(text)
            true_video_ids.append(ext['video_id'])
        
        # Encode all texts
        logger.info(f"Encoding {len(fake_texts)} fake videos and {len(true_texts)} true videos...")
        fake_embeddings = self.encode_text_with_clip(fake_texts)
        true_embeddings = self.encode_text_with_clip(true_texts)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(fake_embeddings, true_embeddings)
        
        # Find top-k most similar true news for each fake news
        similar_videos = {}
        
        for i, fake_vid in enumerate(fake_video_ids):
            # Get similarity scores for this fake video
            sim_scores = similarities[i]
            
            # Get indices of top-k most similar
            top_indices = np.argsort(sim_scores)[-top_k:][::-1]
            
            # Store results
            similar_videos[fake_vid] = [
                (true_video_ids[idx], float(sim_scores[idx]))
                for idx in top_indices
            ]
            
        logger.info(f"Found top-{top_k} similar true news for {len(fake_video_ids)} fake videos")
        return similar_videos
    
    def create_false_claim_prompt(self, fake_video: Dict, similar_true_videos: List[Dict], entity_kb: Dict) -> str:
        """Create prompt for extracting false claims based on similar true news"""
        title = fake_video.get('title', '')
        keywords = fake_video.get('keywords', '')
        description = fake_video.get('description', '')
        temporal = fake_video.get('temporal_evolution', '')
        
        # Collect entities and their correct claims from similar true videos
        relevant_entities = {}
        for true_video in similar_true_videos:
            # Get entities mentioned in this true video
            for entity in true_video.get('entities', []):
                normalized_entity = self.normalize_entity(entity)
                if normalized_entity in entity_kb:
                    if normalized_entity not in relevant_entities:
                        relevant_entities[normalized_entity] = entity_kb[normalized_entity].get('correct_claims', [])
        
        # Format entities and claims for prompt
        entity_claim_text = ""
        for entity, claims in relevant_entities.items():
            entity_claim_text += f"\n【{entity}】的正确声称：\n"
            for claim_data in claims[:5]:  # Limit to 5 claims per entity
                entity_claim_text += f"  - {claim_data['claim']}\n"
        
        prompt = f"""分析这个假新闻视频，基于相关真实新闻的实体和声称，提取虚假声称。

当前假新闻视频信息：
- 标题：{title}
- 关键词：{keywords}
- 视频描述：{description}
- 时序演变：{temporal}
- 视频标签：假（这是一个假新闻视频）

相关真实新闻中的实体及其正确声称：
{entity_claim_text if entity_claim_text else "暂无相关实体"}

任务要求：
1. 针对上述已有实体，分析这个假新闻视频做出了哪些虚假声称
2. 每个虚假声称要具体、明确，包含细节信息
3. 如果视频中还涉及其他未列出的实体，也可以提取其虚假声称，但要标注为"新实体"

返回JSON格式：
{{
  "entity_false_claims": {{
    "已有实体1": [
      "关于该实体的虚假声称1",
      "关于该实体的虚假声称2"
    ],
    "已有实体2": [
      "关于该实体的虚假声称"
    ]
  }},
  "new_entity_false_claims": {{
    "新实体1": [
      "关于新实体的虚假声称"
    ]
  }}
}}

注意：
- 虚假声称应该与正确声称形成对比或矛盾
- 虚假声称要基于视频内容，不要凭空捏造
- 新实体的虚假声称会标记为"基于LLM推测"
"""
        
        return prompt
    
    async def extract_false_claims(self, fake_video: Dict, similar_true_videos: List[Dict], 
                                   entity_kb: Dict, images: List[str]) -> Optional[Dict]:
        """Extract false claims from fake video based on similar true news context"""
        video_id = fake_video.get('video_id')
        
        try:
            # Create prompt
            prompt = self.create_false_claim_prompt(fake_video, similar_true_videos, entity_kb)
            
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
            
            return {
                "video_id": video_id,
                "annotation": "假",
                "entity_false_claims": result.get("entity_false_claims", {}),
                "new_entity_false_claims": result.get("new_entity_false_claims", {}),
                "metadata": {
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "similar_true_videos": [v['video_id'] for v in similar_true_videos]
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to extract false claims from video {video_id}: {e}")
            return None
    
    async def process_fake_news_pipeline(self, sample_size: Optional[int] = None, 
                                         step1_only: bool = False,
                                         skip_step1: bool = False,
                                         max_concurrent: int = 10):
        """
        Complete pipeline for processing fake news videos
        
        Args:
            sample_size: Number of samples to process (None for all)
            step1_only: If True, only execute Step 1 (extract descriptions)
            skip_step1: If True, skip Step 1 and continue from Step 2
            max_concurrent: Maximum number of concurrent API calls (default: 10)
        """
        logger.info("Starting fake news processing pipeline...")
        
        if step1_only and skip_step1:
            logger.error("Cannot set both step1_only and skip_step1 to True")
            return
        
        # Load files if needed for later steps
        entity_kb = None
        true_extractions = []
        
        if not step1_only:
            # Step 0: Load existing entity knowledge base
            kb_file = self.output_dir / "entity_knowledge_base.json"
            if not kb_file.exists():
                logger.error("Entity knowledge base not found. Please run true news extraction first.")
                return
            
            with open(kb_file, 'r', encoding='utf-8') as f:
                entity_kb = json.load(f)
            logger.info(f"Loaded entity KB with {len(entity_kb)} entities")
            
            # Load true video extractions
            true_extractions_file = self.output_dir / "video_extractions.jsonl"
            with open(true_extractions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    true_extractions.append(json.loads(line))
            logger.info(f"Loaded {len(true_extractions)} true video extractions")
        
        # Step 1: Extract descriptions from fake videos
        fake_descriptions = []
        
        if not skip_step1:
            logger.info("Step 1: Extracting descriptions from fake videos...")
            fake_videos = self.load_video_data(filter_label='假')
            
            if sample_size:
                fake_videos = fake_videos[:sample_size]
                logger.info(f"Processing sample of {sample_size} fake videos")
            
            fake_desc_file = self.output_dir / "fake_video_descriptions.jsonl"
            
            # Checkpoint file for Step 1
            step1_checkpoint_file = self.output_dir / "step1_checkpoint.json"
            
            # Load existing descriptions if resuming
            existing_fake_ids = set()
            if fake_desc_file.exists():
                with open(fake_desc_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        desc = json.loads(line)
                        fake_descriptions.append(desc)
                        existing_fake_ids.add(desc['video_id'])
                logger.info(f"Loaded {len(fake_descriptions)} existing fake video descriptions")
            
            # Load checkpoint if exists
            if step1_checkpoint_file.exists():
                with open(step1_checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                    logger.info(f"Loaded Step 1 checkpoint: {checkpoint.get('processed', 0)} videos processed")
            
            # Process remaining fake videos with concurrency
            videos_to_process = [v for v in fake_videos if v['video_id'] not in existing_fake_ids]
            logger.info(f"{len(videos_to_process)} videos remaining to process")
            
            # Concurrent processing with semaphore
            semaphore = asyncio.Semaphore(max_concurrent)
            logger.info(f"Using {max_concurrent} concurrent connections")
            checkpoint_interval = 20  # Save checkpoint every N videos
            
            async def process_fake_video_with_semaphore(video, pbar):
                async with semaphore:
                    if self.shutdown_requested:
                        return None
                    
                    images = self.prepare_composite_frames(video['video_id'])
                    description = await self.extract_fake_description(video, images)
                    
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    return description
            
            # Process in batches for better checkpoint management
            batch_size = 50
            for batch_start in range(0, len(videos_to_process), batch_size):
                if self.shutdown_requested:
                    logger.info("Processing interrupted. Saving checkpoint...")
                    checkpoint = {
                        'processed': len(existing_fake_ids),
                        'total': len(fake_videos),
                        'timestamp': datetime.now().isoformat()
                    }
                    with open(step1_checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
                    logger.info("Checkpoint saved. Run again to continue.")
                    return
                
                batch_end = min(batch_start + batch_size, len(videos_to_process))
                batch_videos = videos_to_process[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start//batch_size + 1}: videos {batch_start+1}-{batch_end} of {len(videos_to_process)}")
                
                # Create tasks for this batch
                with tqdm(total=len(batch_videos), desc=f"Batch {batch_start//batch_size + 1}") as pbar:
                    tasks = [process_fake_video_with_semaphore(video, pbar) for video in batch_videos]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for video, result in zip(batch_videos, results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to process video {video['video_id']}: {result}")
                    elif result:
                        fake_descriptions.append(result)
                        existing_fake_ids.add(video['video_id'])
                        # Save immediately
                        with open(fake_desc_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                # Save checkpoint after each batch
                checkpoint = {
                    'processed': len(existing_fake_ids),
                    'total': len(fake_videos),
                    'timestamp': datetime.now().isoformat(),
                    'last_batch': batch_end
                }
                with open(step1_checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint, f, ensure_ascii=False, indent=2)
                logger.info(f"Batch complete. Processed {len(existing_fake_ids)}/{len(fake_videos)} total videos")
            
            # Clean up checkpoint file if all done
            if len(existing_fake_ids) >= len(fake_videos):
                if step1_checkpoint_file.exists():
                    step1_checkpoint_file.unlink()
                    logger.info("Step 1 completed, checkpoint removed")
            
            logger.info(f"Step 1 complete: Extracted descriptions for {len(fake_descriptions)} fake videos")
            
            if step1_only:
                logger.info("\n=== Step 1 Only Mode - Pipeline Stopped ===")
                logger.info(f"Descriptions saved to: {fake_desc_file}")
                logger.info(f"Total descriptions: {len(fake_descriptions)}")
                return
        else:
            # Load existing descriptions if skipping Step 1
            fake_desc_file = self.output_dir / "fake_video_descriptions.jsonl"
            if not fake_desc_file.exists():
                logger.error("fake_video_descriptions.jsonl not found. Please run Step 1 first.")
                return
            
            with open(fake_desc_file, 'r', encoding='utf-8') as f:
                for line in f:
                    fake_descriptions.append(json.loads(line))
            logger.info(f"Loaded {len(fake_descriptions)} existing fake video descriptions")
        
        # Step 2: Find similar true news
        logger.info("Step 2: Finding similar true news for each fake video...")
        similar_videos_map = self.find_similar_true_news(fake_descriptions, true_extractions, top_k=5)
        
        # Save similarity mapping
        similarity_file = self.output_dir / "fake_true_similarity.json"
        with open(similarity_file, 'w', encoding='utf-8') as f:
            json.dump(similar_videos_map, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved similarity mapping to {similarity_file}")
        
        # Step 3: Extract false claims based on similar true news
        logger.info("Step 3: Extracting false claims with true news context...")
        false_claims_file = self.output_dir / "fake_video_false_claims.jsonl"
        
        # Load existing false claims if resuming
        existing_false_claim_ids = set()
        all_false_claims = []
        if false_claims_file.exists():
            with open(false_claims_file, 'r', encoding='utf-8') as f:
                for line in f:
                    claim_data = json.loads(line)
                    all_false_claims.append(claim_data)
                    existing_false_claim_ids.add(claim_data['video_id'])
            logger.info(f"Loaded {len(all_false_claims)} existing false claim extractions")
        
        # Create lookup for true extractions
        true_video_lookup = {ext['video_id']: ext for ext in true_extractions}
        
        for fake_desc in tqdm(fake_descriptions, desc="Extracting false claims"):
            if fake_desc['video_id'] in existing_false_claim_ids:
                continue
            
            # Get similar true videos
            similar_vids = similar_videos_map.get(fake_desc['video_id'], [])
            similar_true_videos = [
                true_video_lookup[vid_id] 
                for vid_id, _ in similar_vids 
                if vid_id in true_video_lookup
            ]
            
            if not similar_true_videos:
                logger.warning(f"No similar true videos found for {fake_desc['video_id']}")
                continue
            
            # Get images
            images = self.prepare_composite_frames(fake_desc['video_id'])
            
            # Extract false claims
            false_claims = await self.extract_false_claims(
                fake_desc, similar_true_videos, entity_kb, images
            )
            
            if false_claims:
                all_false_claims.append(false_claims)
                # Save immediately
                with open(false_claims_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(false_claims, ensure_ascii=False) + '\n')
            
            await asyncio.sleep(self.rate_limit_delay)
        
        logger.info(f"Extracted false claims for {len(all_false_claims)} fake videos")
        
        # Step 4: Update entity knowledge base with false claims
        logger.info("Step 4: Updating entity knowledge base with false claims...")
        
        for claim_data in all_false_claims:
            video_id = claim_data['video_id']
            
            # Process existing entity false claims
            for entity, claims in claim_data.get('entity_false_claims', {}).items():
                normalized_entity = self.normalize_entity(entity)
                
                if normalized_entity not in entity_kb:
                    # Create new entry if entity doesn't exist
                    entity_kb[normalized_entity] = {
                        "correct_claims": [],
                        "false_claims": [],
                        "video_count": {"真": 0, "假": 0, "辟谣": 0},
                        "first_seen": datetime.now().isoformat(),
                        "last_seen": datetime.now().isoformat(),
                        "aliases": [entity] if entity != normalized_entity else []
                    }
                
                # Add false claims
                for claim in claims:
                    claim_entry = {
                        "claim": claim,
                        "video_ids": [video_id],
                        "source_label": "假"
                    }
                    
                    # Check if claim already exists
                    existing = False
                    for existing_claim in entity_kb[normalized_entity]["false_claims"]:
                        if existing_claim["claim"] == claim:
                            existing_claim["video_ids"].append(video_id)
                            existing = True
                            break
                    
                    if not existing:
                        entity_kb[normalized_entity]["false_claims"].append(claim_entry)
                
                # Update video count
                entity_kb[normalized_entity]["video_count"]["假"] += 1
            
            # Process new entity false claims (mark as LLM-generated)
            for entity, claims in claim_data.get('new_entity_false_claims', {}).items():
                normalized_entity = self.normalize_entity(entity)
                
                if normalized_entity not in entity_kb:
                    entity_kb[normalized_entity] = {
                        "correct_claims": [],
                        "false_claims": [],
                        "video_count": {"真": 0, "假": 0, "辟谣": 0},
                        "first_seen": datetime.now().isoformat(),
                        "last_seen": datetime.now().isoformat(),
                        "aliases": [entity] if entity != normalized_entity else [],
                        "llm_generated": True  # Mark as LLM-generated
                    }
                
                for claim in claims:
                    claim_entry = {
                        "claim": claim,
                        "video_ids": [video_id],
                        "source_label": "假",
                        "llm_generated": True  # Mark claim as LLM-generated
                    }
                    
                    # Check if claim already exists
                    existing = False
                    for existing_claim in entity_kb[normalized_entity]["false_claims"]:
                        if existing_claim["claim"] == claim:
                            existing_claim["video_ids"].append(video_id)
                            existing = True
                            break
                    
                    if not existing:
                        entity_kb[normalized_entity]["false_claims"].append(claim_entry)
                
                entity_kb[normalized_entity]["video_count"]["假"] += 1
        
        # Generate false claim summaries
        logger.info("Generating false claim summaries for entities...")
        await self.enhance_entity_false_descriptions(entity_kb)
        
        # Save updated entity knowledge base
        with open(kb_file, 'w', encoding='utf-8') as f:
            json.dump(entity_kb, f, ensure_ascii=False, indent=2)
        logger.info(f"Updated entity knowledge base saved to {kb_file}")
        
        # Print statistics
        false_claim_count = sum(
            len(data.get("false_claims", [])) 
            for data in entity_kb.values()
        )
        logger.info(f"\n=== Fake News Processing Complete ===")
        logger.info(f"Processed {len(fake_descriptions)} fake videos")
        logger.info(f"Added {false_claim_count} false claims to knowledge base")
        logger.info(f"Updated {len(entity_kb)} entities")
    
    async def enhance_entity_false_descriptions(self, entity_kb: Dict):
        """Generate false claim summaries for entities"""
        logger.info("Generating false claim summaries for entities...")
        
        entities_with_false_claims = [
            (entity, data) for entity, data in entity_kb.items()
            if data.get("false_claims") and not data.get("false_description") or data.get("false_description") == "[待后续从假新闻视频中提取]"
        ]
        
        for entity, data in tqdm(entities_with_false_claims, desc="Generating false descriptions"):
            false_claims = [c["claim"] for c in data.get("false_claims", [])]
            
            if false_claims:
                data["false_description"] = await self.generate_entity_description(
                    entity, false_claims, claim_type="false"
                )
                await asyncio.sleep(0.5)  # Rate limiting
    
    async def generate_entity_description(self, entity: str, claims: List[str], claim_type: str = "correct") -> str:
        """Generate entity description using LLM based on claims"""
        if not claims:
            return "暂无相关声称"
        
        claim_label = "正确" if claim_type == "correct" else "虚假"
        
        prompt = f"""基于以下关于"{entity}"的{claim_label}声称，生成一个简洁、准确的描述（50-100字）：

声称列表：
{chr(10).join(f'- {claim}' for claim in claims)}

要求：
1. 综合所有声称中的关键信息
2. 突出最重要的事实（时间、地点、事件）
3. 保持客观、准确
4. 不要简单罗列声称，而是形成连贯的描述
5. 明确指出这些是{claim_label}声称

返回格式（纯文本，不要JSON）：
[实体的综合描述]"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            
            description = response.choices[0].message.content.strip()
            return description
        except Exception as e:
            logger.warning(f"Failed to generate {claim_type} description for {entity}: {e}")
            return f"基于{claim_label}视频中的声称：{'; '.join(claims[:3])}"
    
    def aggregate_knowledge_base(self, extractions: List[Dict]) -> Dict:
        """Aggregate extractions into entity knowledge base"""
        entity_kb = defaultdict(lambda: {
            "correct_claims": [],
            "false_claims": [],
            "video_count": {"真": 0, "假": 0, "辟谣": 0},
            "first_seen": None,
            "last_seen": None,
            "aliases": []
        })
        
        claim_index = {
            "correct_claims": [],
            "false_claims": [],
            "statistics": {
                "total_correct_claims": 0,
                "total_false_claims": 0,
                "total_videos_processed": {"真": 0, "假": 0, "辟谣": 0}
            }
        }
        
        
        # Process each extraction
        for ext in extractions:
            video_id = ext["video_id"]
            annotation = ext["annotation"]
            entity_claims = ext.get("entity_claims", {})
            
            # Update video count
            claim_index["statistics"]["total_videos_processed"][annotation] += 1
            
            # Process entity-claim mappings
            for entity, claims in entity_claims.items():
                # Normalize entity name
                normalized_entity = self.normalize_entity(entity)
                
                if normalized_entity not in entity_kb:
                    entity_kb[normalized_entity]["first_seen"] = ext["metadata"]["timestamp"]
                entity_kb[normalized_entity]["last_seen"] = ext["metadata"]["timestamp"]
                entity_kb[normalized_entity]["video_count"][annotation] += 1
                
                # Track original entity name as alias
                if entity != normalized_entity and entity not in entity_kb[normalized_entity]["aliases"]:
                    entity_kb[normalized_entity]["aliases"].append(entity)
                
                # Add claims for this entity (now directly mapped)
                for claim in claims:
                    claim_entry = {
                        "claim": claim,
                        "video_ids": [video_id],
                        "source_label": annotation
                    }
                    
                    # Check if claim already exists
                    existing = False
                    claims_list = entity_kb[normalized_entity]["correct_claims"] if annotation == "真" else entity_kb[normalized_entity]["false_claims"]
                    
                    for existing_claim in claims_list:
                        if existing_claim["claim"] == claim:
                            existing_claim["video_ids"].append(video_id)
                            existing = True
                            break
                    
                    if not existing:
                        claims_list.append(claim_entry)
                    
                    # Also update claim index
                    claim_index_entry = {
                        "claim": claim,
                        "entities": [normalized_entity],  # Use normalized entity
                        "video_ids": [video_id],
                        "source_label": annotation
                    }
                    
                    if annotation == "真":
                        # Check if claim already in index
                        existing_in_index = False
                        for idx_claim in claim_index["correct_claims"]:
                            if idx_claim["claim"] == claim:
                                idx_claim["video_ids"].append(video_id)
                                if normalized_entity not in idx_claim["entities"]:
                                    idx_claim["entities"].append(normalized_entity)
                                existing_in_index = True
                                break
                        
                        if not existing_in_index:
                            claim_index["correct_claims"].append(claim_index_entry)
                            claim_index["statistics"]["total_correct_claims"] += 1
                    else:
                        # Similar for false claims
                        existing_in_index = False
                        for idx_claim in claim_index["false_claims"]:
                            if idx_claim["claim"] == claim:
                                idx_claim["video_ids"].append(video_id)
                                if normalized_entity not in idx_claim["entities"]:
                                    idx_claim["entities"].append(normalized_entity)
                                existing_in_index = True
                                break
                        
                        if not existing_in_index:
                            claim_index["false_claims"].append(claim_index_entry)
                            claim_index["statistics"]["total_false_claims"] += 1
        
        # Merge similar entities
        entity_kb = self.merge_similar_entities(dict(entity_kb))
        
        return entity_kb, claim_index
    
    def save_results(self, extractions: List[Dict], entity_kb: Dict, claim_index: Dict):
        """Save all results to files"""
        # Save raw extractions
        extractions_file = self.output_dir / "video_extractions.jsonl"
        with open(extractions_file, 'w', encoding='utf-8') as f:
            for ext in extractions:
                f.write(json.dumps(ext, ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(extractions)} extractions to {extractions_file}")
        
        # Save entity knowledge base
        kb_file = self.output_dir / "entity_knowledge_base.json"
        with open(kb_file, 'w', encoding='utf-8') as f:
            json.dump(entity_kb, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved entity knowledge base to {kb_file}")
        
        # Save claim index
        index_file = self.output_dir / "claim_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(claim_index, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved claim index to {index_file}")
        
        # Save statistics
        stats_file = self.output_dir / "extraction_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved statistics to {stats_file}")
    
    async def run(self, sample_size: Optional[int] = None, generate_descriptions: bool = True):
        """Run the extraction pipeline"""
        logger.info("Starting entity-claim extraction pipeline")
        
        # Load video data
        videos = self.load_video_data(filter_label='真')
        
        # Sample if requested
        if sample_size:
            videos = videos[:sample_size]
            logger.info(f"Processing sample of {sample_size} videos")
        
        try:
            # Process videos with concurrency
            logger.info(f"Processing {len(videos)} videos with concurrent API calls...")
            extractions = await self.process_batch(videos, max_concurrent=10)
            
            # Check if user requested shutdown
            if self.shutdown_requested:
                logger.info("Processing interrupted by user. Saving partial results...")
                # Save checkpoint before exit
                self.save_checkpoint()
                
                # Only process what we have so far
                if self.all_extractions:
                    # Aggregate results without generating descriptions
                    logger.info("Aggregating partial knowledge base...")
                    entity_kb, claim_index = self.aggregate_knowledge_base(self.all_extractions)
                    
                    # Save partial results WITHOUT descriptions
                    logger.info("Saving partial results (without descriptions)...")
                    self.save_results(self.all_extractions, entity_kb, claim_index)
                    logger.info("Partial results saved. Run again to continue processing.")
                
                return
            
            # If we completed all videos, proceed with full processing
            logger.info("All videos processed successfully!")
            
            # Aggregate results
            logger.info("Aggregating knowledge base...")
            entity_kb, claim_index = self.aggregate_knowledge_base(self.all_extractions)  # Use all_extractions instead of extractions
            
            # Generate LLM-based descriptions if all videos are processed
            # Check if we have processed all videos (including from previous runs)
            all_videos_processed = len(self.processed_videos) >= len(videos)
            
            if generate_descriptions and all_videos_processed:
                logger.info("Generating entity descriptions (this may take a few minutes)...")
                await self.enhance_entity_descriptions(entity_kb)
            elif not generate_descriptions:
                logger.info("Skipping description generation as requested")
            elif not all_videos_processed:
                logger.info(f"Not all videos processed yet ({len(self.processed_videos)}/{len(videos)}). Skipping description generation.")
            
            # Save final results
            logger.info("Saving final results...")
            self.save_results(self.all_extractions, entity_kb, claim_index)  # Use all_extractions to include everything
            
            # Clean up temporary files if everything succeeded
            if self.temp_extractions_file.exists():
                self.temp_extractions_file.unlink()
                logger.info("Cleaned up temporary extraction file")
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                logger.info("Cleaned up checkpoint file")
            
            # Clean up description checkpoint if exists
            desc_checkpoint_file = self.output_dir / "description_checkpoint.json"
            if desc_checkpoint_file.exists():
                desc_checkpoint_file.unlink()
                logger.info("Cleaned up description checkpoint file")
            
            # Clean up partial KB file if exists
            partial_kb_file = self.output_dir / "entity_knowledge_base_partial.json"
            if partial_kb_file.exists():
                partial_kb_file.unlink()
                logger.info("Cleaned up partial KB file")
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            # Save checkpoint on error
            self.save_checkpoint()
            logger.info("Checkpoint saved. Run again to continue from where it stopped.")
            raise
        
        # Print statistics
        logger.info("\n=== Extraction Statistics ===")
        logger.info(f"Total processed: {self.stats['total_processed']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Total entities: {self.stats['total_entities']}")
        logger.info(f"Total claims: {self.stats['total_claims']}")
        logger.info(f"Avg entities per video: {self.stats['total_entities'] / max(self.stats['successful'], 1):.2f}")
        logger.info(f"Avg claims per video: {self.stats['total_claims'] / max(self.stats['successful'], 1):.2f}")
        logger.info(f"Unique entities after merging: {len(entity_kb)}")
        
        # Log entity merging statistics
        total_aliases = sum(len(data.get("aliases", [])) for data in entity_kb.values())
        if total_aliases > 0:
            logger.info(f"Entity merging: {total_aliases} entity variations merged")
        

async def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description="Extract entities and claims from FakeSV videos")
    parser.add_argument("--sample", type=int, help="Process only N samples for testing")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--no-descriptions", action="store_true", help="Skip LLM-based description generation")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore checkpoint")
    parser.add_argument("--resume-descriptions", action="store_true", help="Resume only description generation from existing KB")
    parser.add_argument("--process-fake", action="store_true", help="Process fake news videos after true news")
    parser.add_argument("--fake-only", action="store_true", help="Only process fake news videos (requires existing true news KB)")
    parser.add_argument("--fake-step1-only", action="store_true", help="Only execute Step 1 of fake news processing (extract descriptions)")
    parser.add_argument("--fake-skip-step1", action="store_true", help="Skip Step 1 and continue from Step 2 in fake news processing")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Maximum number of concurrent API calls (default: 10)")
    args = parser.parse_args()
    
    # Create extractor
    extractor = EntityClaimExtractor(api_key=args.api_key, resume=not args.no_resume)
    
    # Special mode: process only fake news
    if args.fake_only or args.fake_step1_only:
        await extractor.process_fake_news_pipeline(
            sample_size=args.sample, 
            step1_only=args.fake_step1_only,
            skip_step1=args.fake_skip_step1,
            max_concurrent=args.max_concurrent
        )
        return
    
    # Special mode: resume only description generation
    if args.resume_descriptions:
        # Load existing KB
        kb_file = extractor.output_dir / "entity_knowledge_base.json"
        if not kb_file.exists():
            kb_partial_file = extractor.output_dir / "entity_knowledge_base_partial.json"
            if kb_partial_file.exists():
                kb_file = kb_partial_file
                logger.info("Loading partial KB for description generation...")
            else:
                logger.error("No entity knowledge base found. Run extraction first.")
                return
        
        with open(kb_file, 'r', encoding='utf-8') as f:
            entity_kb = json.load(f)
        
        logger.info(f"Loaded KB with {len(entity_kb)} entities")
        await extractor.enhance_entity_descriptions(entity_kb)
        
        # Save updated KB
        with open(extractor.output_dir / "entity_knowledge_base.json", 'w', encoding='utf-8') as f:
            json.dump(entity_kb, f, ensure_ascii=False, indent=2)
        logger.info("Description generation completed and KB saved")
        return
    
    # Run extraction for true news
    await extractor.run(sample_size=args.sample, generate_descriptions=not args.no_descriptions)
    
    # If requested, also process fake news
    if args.process_fake:
        logger.info("\n=== Starting fake news processing ===")
        await extractor.process_fake_news_pipeline(
            sample_size=args.sample, 
            step1_only=args.fake_step1_only,
            skip_step1=args.fake_skip_step1,
            max_concurrent=args.max_concurrent
        )


if __name__ == "__main__":
    asyncio.run(main())