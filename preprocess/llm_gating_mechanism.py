#!/usr/bin/env python3
"""
LLM Gating Mechanism for ExMRD: 
For each video, test LLM's response to both SLM predictions (true/fake) to create a gating mechanism.
"""

import json
import os
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import logging
import sys
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Add preprocess directory to path for utils import
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'preprocess'))
from cot.utils import ChatLLM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMGatingMechanism:
    def __init__(self, 
                 retrieval_results_file: str,
                 output_dir: str = "data/FakeSV/entity_claims/gating_predictions",
                 max_samples: int = None,
                 concurrency: int = 20,
                 batch_size: int = 50,
                 filter_k: int = None,
                 no_slm: bool = False):
        """
        Initialize LLM Gating Mechanism
        
        Args:
            retrieval_results_file: Path to full dataset retrieval results
            output_dir: Output directory for gating predictions
            max_samples: Maximum samples to process (None for all)
            concurrency: Maximum concurrent requests
            batch_size: Process N samples per batch for checkpointing
            filter_k: Filter parameter for using k-specific files (None for no filtering)
            no_slm: If True, use independent LLM prediction without SLM bias
        """
        self.filter_k = filter_k
        self.no_slm = no_slm
        self.retrieval_results_file = retrieval_results_file
        
        # Adjust output directory based on filter_k and no_slm
        if filter_k is not None:
            base_dir = f"{output_dir}_k{filter_k}"
        else:
            base_dir = output_dir
            
        if no_slm:
            self.output_dir = Path(f"{base_dir}_independent")
        else:
            self.output_dir = Path(base_dir)
            
        self.max_samples = max_samples
        self.concurrency = concurrency
        self.batch_size = batch_size
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load fake entity claims data
        self.load_fake_entity_claims()
        
        # Setup LLM client
        self.setup_llm_client()
        
        # Create gating prompt template
        self.create_prompt_template()
        
        logger.info(f"Initialized LLM Gating Mechanism")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_fake_entity_claims(self):
        """Load fake news entity claims from separate file"""
        fake_claims_file = Path("data/FakeSV/entity_claims/fake_entity_claims.jsonl")
        self.fake_entity_claims = {}
        
        # When k is specified, first load relevant video IDs from split files
        relevant_video_ids = None
        if self.filter_k is not None:
            relevant_video_ids = self.load_relevant_video_ids()
        
        if fake_claims_file.exists():
            with open(fake_claims_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    video_id = item.get('video_id')
                    entity_claims = item.get('entity_claims', {})
                    
                    if video_id:
                        # If k is specified, only load claims for relevant video IDs
                        if relevant_video_ids is None or video_id in relevant_video_ids:
                            self.fake_entity_claims[video_id] = entity_claims
            
            if self.filter_k is not None:
                logger.info(f"Loaded entity claims for {len(self.fake_entity_claims)} fake videos (filtered for k={self.filter_k})")
            else:
                logger.info(f"Loaded entity claims for {len(self.fake_entity_claims)} fake videos")
        else:
            logger.warning(f"Fake entity claims file not found: {fake_claims_file}")
            self.fake_entity_claims = {}
    
    def load_relevant_video_ids(self) -> set:
        """Load video IDs from train/valid/test split files when k is specified"""
        relevant_ids = set()
        
        # Define split file paths
        split_files = {
            'train': f'data/FakeSV/vids/vid_time3_train_k{self.filter_k}.txt',
            'valid': 'data/FakeSV/vids/vid_time3_valid.txt',
            'test': 'data/FakeSV/vids/vid_time3_test.txt'
        }
        
        for split_name, file_path in split_files.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        video_id = line.strip()
                        if video_id:
                            relevant_ids.add(video_id)
                logger.info(f"Loaded {len([line.strip() for line in open(file_path, 'r', encoding='utf-8') if line.strip()])} video IDs from {split_name} split")
            except FileNotFoundError:
                logger.warning(f"Split file not found: {file_path}")
        
        logger.info(f"Total relevant video IDs for k={self.filter_k}: {len(relevant_ids)}")
        return relevant_ids
        
    def setup_llm_client(self):
        """Setup AsyncOpenAI client"""
        load_dotenv(override=True)
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        base_url = os.getenv('OPENAI_BASE_URL')
        if base_url:
            self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.async_client = AsyncOpenAI(api_key=api_key)
        
        self.model = 'gpt-4o-mini'
        self.temperature = 0.1
        self.max_tokens = 2000
        
        logger.info("Initialized AsyncOpenAI client for GPT-4o-mini")
    
    def create_prompt_template(self):
        """Create the gating mechanism prompt template"""
        if self.no_slm:
            # Independent LLM prediction template (no SLM bias)
            self.gating_prompt_template = """你是一位专业的假新闻检测专家。一个多模态小模型对该新闻做出了初步预测，但是门控器对该预测结果存疑于是让你进行最后查验。

**当前视频信息**：
- 发布时间：{current_time}
- 标题：{current_title}
- 关键词：{current_keywords}
- 描述：{current_description}
- 时间演变：{current_temporal}

**当前视频的实体声称**：
{current_entity_claims}

**参考信息**：

**最相似的真新闻**：
- 标注：{similar_true_annotation}
- 发布时间：{similar_true_time}
- 标题：{similar_true_title}
- 关键词：{similar_true_keywords}
- 描述：{similar_true_description}
- 时间演变：{similar_true_temporal}

**最相似真新闻的实体声称**：
{similar_true_entity_claims}

**最相似的假新闻**：
- 标注：{similar_fake_annotation}
- 发布时间：{similar_fake_time}
- 标题：{similar_fake_title}
- 关键词：{similar_fake_keywords}
- 描述：{similar_fake_description}
- 时间演变：{similar_fake_temporal}

**最相似假新闻的实体声称**：
{similar_fake_entity_claims}

**请完成以下任务**：

1. **整合所有线索**：综合考虑实体声称、参考新闻等所有信息

2. **详细推理过程**：分析当前视频内容和实体声称，对比参考的真假新闻信息，从以下角度进行分析：
   - 实体声称的真实性和合理性分析
   - 与参考真新闻的相似性分析
   - 与参考假新闻的相似性分析
   - 时间逻辑性分析

3. **最终判断**：基于以上分析，判断当前视频是"真"还是"假"

**请按以下格式回答（必须返回JSON格式）**：

{{
  "reasoning": "详细的分析过程",
  "prediction": "真/假"
}}"""
        else:
            # Original gating mechanism template (with SLM bias)
            self.gating_prompt_template = """你是一位专业的假新闻检测专家。一个多模态小模型对该新闻预测结果为{slm_prediction}，但是门控器对该预测结果存疑于是让你进行最后查验。

**当前视频信息**：
- 发布时间：{current_time}
- 标题：{current_title}
- 关键词：{current_keywords}
- 描述：{current_description}
- 时间演变：{current_temporal}

**当前视频的实体声称**：
{current_entity_claims}

**参考信息**：

**最相似的真新闻**：
- 标注：{similar_true_annotation}
- 发布时间：{similar_true_time}
- 标题：{similar_true_title}
- 关键词：{similar_true_keywords}
- 描述：{similar_true_description}
- 时间演变：{similar_true_temporal}

**最相似真新闻的实体声称**：
{similar_true_entity_claims}

**最相似的假新闻**：
- 标注：{similar_fake_annotation}
- 发布时间：{similar_fake_time}
- 标题：{similar_fake_title}
- 关键词：{similar_fake_keywords}
- 描述：{similar_fake_description}
- 时间演变：{similar_fake_temporal}

**最相似假新闻的实体声称**：
{similar_fake_entity_claims}

**请完成以下任务**：

1. **整合所有线索**：综合考虑小模型判断、实体声称、参考新闻等所有信息

2. **详细推理过程**：分析当前视频内容和实体声称，对比参考的真假新闻信息，从以下角度进行分析：
   - 实体声称的真实性和合理性分析
   - 与参考真新闻的相似性分析
   - 与参考假新闻的相似性分析
   - 时间逻辑性分析
   - 综合以上分析，评估小模型预测"{slm_prediction}"的合理性

3. **最终判断**：基于以上分析，判断当前视频是"真"还是"假"

**请按以下格式回答（必须返回JSON格式）**：

{{
  "reasoning": "详细的分析过程",
  "prediction": "真/假"
}}"""

    def format_entity_claims(self, entity_claims_data: Dict) -> str:
        """Format entity claims for prompt"""
        if not entity_claims_data or 'entity_claims' not in entity_claims_data:
            return "无相关实体声称信息"
        
        entity_claims = entity_claims_data['entity_claims']
        if not entity_claims:
            return "无相关实体声称信息"
        
        formatted_claims = []
        for entity, claims in entity_claims.items():
            if claims:
                formatted_claims.append(f"**{entity}**:")
                for claim in claims:
                    formatted_claims.append(f"  - {claim}")
        
        return "\n".join(formatted_claims) if formatted_claims else "无相关实体声称信息"

    def create_gating_prompt(self, sample: Dict, slm_prediction: str = None) -> str:
        """Create prompt for gating mechanism"""
        # Handle different key formats (test_video vs query_video)
        video_key = 'query_video' if 'query_video' in sample else 'test_video'
        current_data = sample[video_key]
        similar_true = sample.get('similar_true', {}) or {}
        similar_fake = sample.get('similar_fake', {}) or {}
        
        # Format current entity claims - get from loaded data if empty
        current_claims = current_data.get('entity_claims', {})
        if not current_claims and current_data.get('video_id') and current_data.get('annotation') == '假':
            # For fake videos, try to get from loaded fake entity claims
            current_claims = self.fake_entity_claims.get(current_data['video_id'], {})
        current_entity_claims = self.format_entity_claims({'entity_claims': current_claims}) if current_claims else "无相关实体声称信息"
        
        # Format similar true entity claims
        similar_true_entity_claims = self.format_entity_claims(similar_true) if similar_true else "无相关信息"
        
        # Format similar fake entity claims - get from loaded data if empty in similar_fake
        similar_fake_claims = similar_fake.get('entity_claims', {})
        if not similar_fake_claims and similar_fake.get('video_id'):
            # Try to get from loaded fake entity claims
            similar_fake_claims = self.fake_entity_claims.get(similar_fake['video_id'], {})
        similar_fake_entity_claims = self.format_entity_claims({'entity_claims': similar_fake_claims}) if similar_fake_claims else "无相关信息"
        
        # Prepare format arguments
        format_args = {
            # Current video info
            'current_time': current_data.get('publish_time', 'Unknown'),
            'current_title': current_data.get('title', ''),
            'current_keywords': current_data.get('keywords', ''),
            'current_description': current_data.get('description', ''),
            'current_temporal': current_data.get('temporal_evolution', ''),
            'current_entity_claims': current_entity_claims,
            # Similar true news
            'similar_true_annotation': similar_true.get('annotation', 'N/A'),
            'similar_true_time': similar_true.get('publish_time', 'Unknown'),
            'similar_true_title': similar_true.get('title', ''),
            'similar_true_keywords': similar_true.get('keywords', ''),
            'similar_true_description': similar_true.get('description', ''),
            'similar_true_temporal': similar_true.get('temporal_evolution', ''),
            'similar_true_entity_claims': similar_true_entity_claims,
            # Similar fake news
            'similar_fake_annotation': similar_fake.get('annotation', 'N/A'),
            'similar_fake_time': similar_fake.get('publish_time', 'Unknown'),
            'similar_fake_title': similar_fake.get('title', ''),
            'similar_fake_keywords': similar_fake.get('keywords', ''),
            'similar_fake_description': similar_fake.get('description', ''),
            'similar_fake_temporal': similar_fake.get('temporal_evolution', ''),
            'similar_fake_entity_claims': similar_fake_entity_claims
        }
        
        # Add SLM prediction only for non-independent mode
        if not self.no_slm and slm_prediction is not None:
            format_args['slm_prediction'] = slm_prediction
        
        prompt = self.gating_prompt_template.format(**format_args)
        return prompt

    async def query_llm_async(self, prompt: str, max_retries: int = 3) -> Optional[Dict]:
        """Async query to LLM with retries"""
        for attempt in range(max_retries):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content.strip()
                
                # Parse JSON response directly (no markdown processing needed)
                result = json.loads(content)
                
                # Normalize prediction values
                prediction = result.get('prediction', '未知').strip()
                if '真' in prediction:
                    result['prediction'] = '真'
                elif '假' in prediction:
                    result['prediction'] = '假'
                else:
                    result['prediction'] = '未知'
                
                return result
            
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response (attempt {attempt + 1}): {e}")
                logger.error(f"Raw response: {content[:200]}...")
                if attempt == max_retries - 1:
                    return {
                        "reasoning": content,
                        "prediction": "真"  # Default fallback
                    }
            
            except Exception as e:
                logger.error(f"Error querying LLM (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(1)  # Brief pause before retry
        
        return None


    def load_retrieval_results(self) -> List[Dict]:
        """Load retrieval results"""
        logger.info(f"Loading retrieval results from {self.retrieval_results_file}")
        
        with open(self.retrieval_results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        logger.info(f"Loaded {len(results)} samples for processing")
        
        if self.max_samples:
            results = results[:self.max_samples]
            logger.info(f"Limited to {len(results)} samples for testing")
        
        return results

    def load_checkpoint(self) -> Tuple[List[Dict], int]:
        """Load existing checkpoint if available"""
        checkpoint_file = self.output_dir / "gating_checkpoint.json"
        
        if checkpoint_file.exists():
            logger.info(f"Loading checkpoint from {checkpoint_file}")
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                return checkpoint.get('processed_results', []), checkpoint.get('next_index', 0)
        
        return [], 0

    def save_checkpoint(self, processed_results: List[Dict], next_index: int):
        """Save checkpoint"""
        checkpoint_file = self.output_dir / "gating_checkpoint.json"
        
        checkpoint = {
            'processed_results': processed_results,
            'next_index': next_index,
            'timestamp': datetime.now().isoformat(),
            'total_processed': len(processed_results)
        }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Checkpoint saved: {len(processed_results)} samples processed")

    def save_final_results(self, results: List[Dict]):
        """Save final results"""
        # Save complete results in a single file
        output_file = self.output_dir / "gating_predictions.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Generate statistics
        self.generate_statistics(results)
        
        logger.info(f"Final results saved to {output_file}")
        logger.info(f"Total samples processed: {len(results)}")

    def generate_statistics(self, results: List[Dict]):
        """Generate processing statistics"""
        stats = {
            'total_samples': len(results),
            'successful_processing': 0,
            'failed_processing': 0,
            'splits': {},
            'ground_truth_distribution': {},
            'prediction_patterns': {
                'slm_true_llm_agrees': 0,
                'slm_true_llm_disagrees': 0,
                'slm_fake_llm_agrees': 0,
                'slm_fake_llm_disagrees': 0
            }
        }
        
        for result in results:
            # Count successful processing
            if (result['predictions']['slm_predicts_true'] and 
                result['predictions']['slm_predicts_fake']):
                stats['successful_processing'] += 1
                
                # Analyze prediction patterns
                slm_true_resp = result['predictions']['slm_predicts_true']
                slm_fake_resp = result['predictions']['slm_predicts_fake']
                
                if slm_true_resp.get('prediction') == '真':
                    stats['prediction_patterns']['slm_true_llm_agrees'] += 1
                else:
                    stats['prediction_patterns']['slm_true_llm_disagrees'] += 1
                
                if slm_fake_resp.get('prediction') == '假':
                    stats['prediction_patterns']['slm_fake_llm_agrees'] += 1
                else:
                    stats['prediction_patterns']['slm_fake_llm_disagrees'] += 1
            else:
                stats['failed_processing'] += 1
            
            # Count splits
            split = result['split']
            if split not in stats['splits']:
                stats['splits'][split] = 0
            stats['splits'][split] += 1
            
            # Count ground truth distribution
            gt = result['ground_truth']
            if gt not in stats['ground_truth_distribution']:
                stats['ground_truth_distribution'][gt] = 0
            stats['ground_truth_distribution'][gt] += 1
        
        # Save statistics
        stats_file = self.output_dir / "gating_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # Print summary
        logger.info("="*60)
        logger.info("GATING MECHANISM PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total samples: {stats['total_samples']}")
        logger.info(f"Successfully processed: {stats['successful_processing']}")
        logger.info(f"Failed processing: {stats['failed_processing']}")
        logger.info(f"Split distribution: {stats['splits']}")
        logger.info(f"Ground truth distribution: {stats['ground_truth_distribution']}")
        logger.info("Prediction patterns:")
        logger.info(f"  SLM→True, LLM agrees: {stats['prediction_patterns']['slm_true_llm_agrees']}")
        logger.info(f"  SLM→True, LLM disagrees: {stats['prediction_patterns']['slm_true_llm_disagrees']}")
        logger.info(f"  SLM→Fake, LLM agrees: {stats['prediction_patterns']['slm_fake_llm_agrees']}")
        logger.info(f"  SLM→Fake, LLM disagrees: {stats['prediction_patterns']['slm_fake_llm_disagrees']}")

    async def process_all_samples(self):
        """Process all samples with gating mechanism using concurrent batches"""
        logger.info("Starting gating mechanism processing...")
        
        # Load retrieval results
        retrieval_results = self.load_retrieval_results()
        
        # Load checkpoint (note: we'll modify checkpoint format for batch processing)
        processed_results, start_index = self.load_checkpoint()
        
        # Filter unprocessed samples
        remaining_samples = retrieval_results[start_index:]
        
        logger.info(f"Total samples: {len(retrieval_results)}")
        logger.info(f"Already processed: {start_index}")
        logger.info(f"Remaining to process: {len(remaining_samples)}")
        
        if not remaining_samples:
            logger.info("All samples already processed!")
            self.save_final_results(processed_results)
            return
        
        # Process in batches for better concurrency and checkpointing
        for batch_start in range(0, len(remaining_samples), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(remaining_samples))
            batch_samples = remaining_samples[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//self.batch_size + 1}: samples {batch_start+1}-{batch_end}")
            
            try:
                # Process current batch concurrently
                batch_results = await self.process_batch_concurrent(batch_samples)
                processed_results.extend(batch_results)
                
                # Update checkpoint
                new_index = start_index + batch_end
                self.save_checkpoint(processed_results, new_index)
                
                logger.info(f"Batch completed. Total processed: {len(processed_results)}")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_start//self.batch_size + 1}: {e}")
                # Create error entries for the entire batch
                for sample in batch_samples:
                    video_key = 'query_video' if 'query_video' in sample else 'test_video'
                    error_result = {
                        'video_id': sample[video_key]['video_id'],
                        'ground_truth': sample[video_key]['annotation'],
                        'split': sample[video_key].get('split', 'test'),
                        'predictions': {
                            'slm_predicts_true': None,
                            'slm_predicts_fake': None
                        },
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    processed_results.append(error_result)
                
                # Still save checkpoint on error
                new_index = start_index + batch_end
                self.save_checkpoint(processed_results, new_index)
        
        # Save final results
        self.save_final_results(processed_results)
        
        # Clean up checkpoint
        checkpoint_file = self.output_dir / "gating_checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info("Cleaned up checkpoint file")
        
        logger.info("Gating mechanism processing completed!")
    
    async def process_batch_concurrent(self, batch_samples: List[Dict]) -> List[Dict]:
        """Process a batch of samples concurrently"""
        if self.no_slm:
            # Independent mode: single query per sample
            logger.info(f"Processing {len(batch_samples)} samples with {self.concurrency} max concurrent requests (independent mode)...")
            return await self._process_batch_independent(batch_samples)
        else:
            # Original mode: two queries per sample (SLM=True and SLM=False)
            logger.info(f"Processing {len(batch_samples)} samples with {self.concurrency} max concurrent requests (gating mode)...")
            return await self._process_batch_gating(batch_samples)
    
    async def _process_batch_independent(self, batch_samples: List[Dict]) -> List[Dict]:
        """Process batch in independent mode (single query per sample)"""
        # Setup semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.concurrency)
        
        async def process_single_sample(sample: Dict):
            """Process single sample with independent prediction"""
            async with semaphore:
                try:
                    prompt = self.create_gating_prompt(sample)  # No SLM prediction needed
                    result = await self.query_llm_async(prompt)
                    await asyncio.sleep(0.1)  # Rate limiting
                    return result
                except Exception as e:
                    logger.error(f"Error processing sample: {e}")
                    return None
        
        # Process all samples concurrently
        logger.info(f"Processing {len(batch_samples)} samples with independent LLM predictions...")
        tasks = [process_single_sample(sample) for sample in batch_samples]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results
        batch_results = []
        for i, sample in enumerate(batch_samples):
            video_key = 'query_video' if 'query_video' in sample else 'test_video'
            video_id = sample[video_key]['video_id']
            ground_truth = sample[video_key]['annotation']
            split = sample[video_key].get('split', 'test')
            
            # Handle results or exceptions
            result = results[i] if not isinstance(results[i], Exception) else None
            
            if isinstance(results[i], Exception):
                logger.error(f"Exception for {video_id}: {results[i]}")
            
            # Create final result (same prediction for both slm_predicts_true and slm_predicts_fake)
            final_result = {
                'video_id': video_id,
                'ground_truth': ground_truth,
                'split': split,
                'predictions': {
                    'slm_predicts_true': result,
                    'slm_predicts_fake': result  # Same result for both since it's independent
                },
                'retrieval_info': {
                    'similar_true': {
                        'video_id': sample.get('similar_true', {}).get('video_id'),
                        'similarity_score': sample.get('similar_true', {}).get('similarity_score')
                    } if sample.get('similar_true') else None,
                    'similar_fake': {
                        'video_id': sample.get('similar_fake', {}).get('video_id'),
                        'similarity_score': sample.get('similar_fake', {}).get('similarity_score')
                    } if sample.get('similar_fake') else None
                },
                'timestamp': datetime.now().isoformat()
            }
            
            batch_results.append(final_result)
        
        logger.info(f"Independent batch processing completed: {len(batch_results)} samples processed")
        return batch_results
    
    async def _process_batch_gating(self, batch_samples: List[Dict]) -> List[Dict]:
        """Process batch in original gating mode (two queries per sample)"""
        # Setup semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.concurrency)
        
        async def process_single_prediction(sample: Dict, slm_prediction: str):
            """Process single sample with specific SLM prediction"""
            async with semaphore:
                try:
                    prompt = self.create_gating_prompt(sample, slm_prediction)
                    result = await self.query_llm_async(prompt)
                    await asyncio.sleep(0.1)  # Rate limiting
                    return result
                except Exception as e:
                    logger.error(f"Error processing sample with SLM={slm_prediction}: {e}")
                    return None
        
        # Phase 1: Process all samples with SLM=真 concurrently
        logger.info(f"Phase 1: Processing {len(batch_samples)} samples with SLM=真...")
        tasks_true = [
            process_single_prediction(sample, "真") 
            for sample in batch_samples
        ]
        results_true = await asyncio.gather(*tasks_true, return_exceptions=True)
        
        # Phase 2: Process all samples with SLM=假 concurrently  
        logger.info(f"Phase 2: Processing {len(batch_samples)} samples with SLM=假...")
        tasks_fake = [
            process_single_prediction(sample, "假") 
            for sample in batch_samples
        ]
        results_fake = await asyncio.gather(*tasks_fake, return_exceptions=True)
        
        # Merge results
        batch_results = []
        for i, sample in enumerate(batch_samples):
            video_key = 'query_video' if 'query_video' in sample else 'test_video'
            video_id = sample[video_key]['video_id']
            ground_truth = sample[video_key]['annotation']
            split = sample[video_key].get('split', 'test')
            
            # Handle results or exceptions
            result_true = results_true[i] if not isinstance(results_true[i], Exception) else None
            result_fake = results_fake[i] if not isinstance(results_fake[i], Exception) else None
            
            if isinstance(results_true[i], Exception):
                logger.error(f"Exception in SLM=真 for {video_id}: {results_true[i]}")
            if isinstance(results_fake[i], Exception):
                logger.error(f"Exception in SLM=假 for {video_id}: {results_fake[i]}")
            
            # Create final result
            final_result = {
                'video_id': video_id,
                'ground_truth': ground_truth,
                'split': split,
                'predictions': {
                    'slm_predicts_true': result_true,
                    'slm_predicts_fake': result_fake
                },
                'retrieval_info': {
                    'similar_true': {
                        'video_id': sample.get('similar_true', {}).get('video_id'),
                        'similarity_score': sample.get('similar_true', {}).get('similarity_score')
                    } if sample.get('similar_true') else None,
                    'similar_fake': {
                        'video_id': sample.get('similar_fake', {}).get('video_id'),
                        'similarity_score': sample.get('similar_fake', {}).get('similarity_score')
                    } if sample.get('similar_fake') else None
                },
                'timestamp': datetime.now().isoformat()
            }
            
            batch_results.append(final_result)
        
        logger.info(f"Gating batch processing completed: {len(batch_results)} samples processed")
        return batch_results

async def main():
    parser = argparse.ArgumentParser(description='LLM Gating Mechanism for ExMRD')
    parser.add_argument('--retrieval-file', type=str, default=None,
                       help='Path to retrieval results file (auto-determined from k if not specified)')
    parser.add_argument('--output-dir', type=str,
                       default='data/FakeSV/entity_claims/gating_predictions',
                       help='Output directory for gating predictions')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples to process (for testing)')
    parser.add_argument('--concurrency', type=int, default=20,
                       help='Maximum concurrent requests')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Process N samples per batch for checkpointing')
    parser.add_argument('--k', type=int, default=None,
                       help='Filter k parameter (None for no filtering)')
    parser.add_argument('--no-slm', action='store_true',
                       help='Use independent LLM prediction without SLM bias')
    
    args = parser.parse_args()
    
    # Auto-determine retrieval file if not specified
    if args.retrieval_file is None:
        if args.k is not None:
            retrieval_file = f'text_similarity_results/full_dataset_retrieval_chinese-clip-vit-large-patch14_k{args.k}.json'
        else:
            retrieval_file = 'text_similarity_results/full_dataset_retrieval_chinese-clip-vit-large-patch14.json'
    else:
        retrieval_file = args.retrieval_file
    
    # Initialize gating mechanism
    gating = LLMGatingMechanism(
        retrieval_results_file=retrieval_file,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        filter_k=args.k,
        no_slm=args.no_slm
    )
    
    # Process all samples
    await gating.process_all_samples()

if __name__ == "__main__":
    asyncio.run(main())