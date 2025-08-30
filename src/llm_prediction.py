#!/usr/bin/env python3
"""
GPT-4o-mini based video fake news detection using text similarity retrieval results
"""

import json
import os
import base64
import argparse
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
from dotenv import load_dotenv
import logging
import sys
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

class LLMVideoPredictor:
    def __init__(self, 
                 retrieval_results_file: str,
                 output_dir: str = "result/FakeSV/llm_prediction_results",
                 max_samples: int = None,
                 checkpoint_interval: int = 10,
                 use_slm_prediction: bool = False,
                 slm_prediction_file: str = None,
                 use_entity_claims: bool = False,
                 entity_claims_true_file: str = None,
                 entity_claims_fake_file: str = None,
                 no_image: bool = False):
        """
        Initialize LLM Video Predictor
        
        Args:
            retrieval_results_file: Path to text similarity retrieval results
            output_dir: Output directory for results
            max_samples: Maximum samples to process (None for all)
            checkpoint_interval: Save checkpoint every N samples
            use_slm_prediction: Whether to use small model predictions
            slm_prediction_file: Path to small model prediction file
            use_entity_claims: Whether to use entity claims information
            entity_claims_true_file: Path to true news entity claims file
            entity_claims_fake_file: Path to fake news entity claims file
            no_image: Whether to disable image input (text-only mode)
        """
        self.retrieval_results_file = retrieval_results_file
        self.output_dir = Path(output_dir)
        self.max_samples = max_samples
        self.checkpoint_interval = checkpoint_interval
        self.use_slm_prediction = use_slm_prediction
        self.slm_prediction_file = slm_prediction_file
        self.use_entity_claims = use_entity_claims
        self.entity_claims_true_file = entity_claims_true_file
        self.entity_claims_fake_file = entity_claims_fake_file
        self.no_image = no_image
        
        # Load SLM predictions if enabled
        self.slm_predictions = {}
        if self.use_slm_prediction and self.slm_prediction_file:
            self.load_slm_predictions()
        
        # Load entity claims if enabled
        self.entity_claims_true = {}
        self.entity_claims_fake = {}
        if self.use_entity_claims:
            self.load_entity_claims()
        
        # Create output subdirectories
        self.predictions_dir = self.output_dir / "predictions"
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.metrics_dir = self.output_dir / "metrics"
        
        # Create directories if they don't exist
        for dir_path in [self.predictions_dir, self.checkpoints_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Data paths
        self.quads_dir = Path("data/FakeSV/quads_4")
        
        # Load environment variables
        load_dotenv(override=True)
        
        # Initialize LLM client
        self.setup_llm_client()
        
        logger.info(f"Initialized LLM Video Predictor")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Max samples: {self.max_samples}")
        logger.info(f"Use SLM prediction: {self.use_slm_prediction}")
        if self.use_slm_prediction:
            logger.info(f"SLM predictions loaded: {len(self.slm_predictions)} videos")
        logger.info(f"Use entity claims: {self.use_entity_claims}")
        if self.use_entity_claims:
            logger.info(f"Entity claims loaded: {len(self.entity_claims_true)} true videos, {len(self.entity_claims_fake)} fake videos")
    
    def load_slm_predictions(self):
        """Load small model predictions from file"""
        try:
            slm_path = Path(self.slm_prediction_file)
            if slm_path.exists():
                logger.info(f"Loading SLM predictions from: {self.slm_prediction_file}")
                with open(self.slm_prediction_file, 'r', encoding='utf-8') as f:
                    slm_data = json.load(f)
                
                # Convert to dictionary for fast lookup: {video_id: prediction}
                for item in slm_data:
                    video_id = item['video_id']
                    prediction = item['prediction']  # "真" or "假"
                    self.slm_predictions[video_id] = prediction
                
                logger.info(f"Loaded {len(self.slm_predictions)} SLM predictions")
            else:
                logger.warning(f"SLM prediction file not found: {self.slm_prediction_file}")
            
        except Exception as e:
            logger.error(f"Error loading SLM predictions: {e}")
            raise
    
    def load_entity_claims(self):
        """Load entity claims from true and fake news files"""
        try:
            # Load true news entity claims
            if self.entity_claims_true_file:
                true_path = Path(self.entity_claims_true_file)
                if true_path.exists():
                    logger.info(f"Loading true news entity claims from: {self.entity_claims_true_file}")
                    with open(self.entity_claims_true_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            item = json.loads(line)
                            video_id = item['video_id']
                            self.entity_claims_true[video_id] = item
                    logger.info(f"Loaded {len(self.entity_claims_true)} true news entity claims")
                else:
                    logger.warning(f"True news entity claims file not found: {self.entity_claims_true_file}")
            
            # Load fake news entity claims  
            if self.entity_claims_fake_file:
                fake_path = Path(self.entity_claims_fake_file)
                if fake_path.exists():
                    logger.info(f"Loading fake news entity claims from: {self.entity_claims_fake_file}")
                    with open(self.entity_claims_fake_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            item = json.loads(line)
                            video_id = item['video_id']
                            self.entity_claims_fake[video_id] = item
                    logger.info(f"Loaded {len(self.entity_claims_fake)} fake news entity claims")
                else:
                    logger.warning(f"Fake news entity claims file not found: {self.entity_claims_fake_file}")
            
        except Exception as e:
            logger.error(f"Error loading entity claims: {e}")
            raise
        
    def setup_llm_client(self):
        """Setup AsyncOpenAI client for GPT-4o-mini"""
        
        # Original prompt template without SLM prediction
        self.prompt_template = """你是一位专业的假新闻检测专家。请分析以下信息来判断当前视频是真新闻还是假新闻。

**当前视频复合帧图片**：
[4个完整复合帧，每个复合帧包含4个关键帧（2x2网格布局）]

**当前视频信息**：
- 发布时间：{current_time}
- 标题：{current_title}
- 关键词：{current_keywords}
- 描述：{current_description}
- 时间演变：{current_temporal}

**参考信息**：

**最相似的真新闻**：
- 标注：{similar_true_annotation}
- 发布时间：{similar_true_time}
- 标题：{similar_true_title}
- 关键词：{similar_true_keywords}
- 描述：{similar_true_description}
- 时间演变：{similar_true_temporal}

**最相似的假新闻**：
- 标注：{similar_fake_annotation}
- 发布时间：{similar_fake_time}
- 标题：{similar_fake_title}
- 关键词：{similar_fake_keywords}
- 描述：{similar_fake_description}
- 时间演变：{similar_fake_temporal}

**请完成以下任务**：

1. **详细推理过程**：分析当前视频内容，对比参考的真假新闻信息，从以下角度进行分析：
   - 检测新闻内容分析
   - 与参考真新闻的相似性分析
   - 与参考假新闻的相似性分析
   - 时间逻辑性分析
   - 内容可信度分析

2. **最终判断**：基于以上分析，判断当前视频是"真"还是"假"

**请按以下格式回答（必须返回JSON格式）**：

{{
  "reasoning": "详细的分析过程",
  "prediction": "真/假"
}}"""

        # No-image version of basic prompt template
        self.prompt_template_no_image = """你是一位专业的假新闻检测专家。请分析以下信息来判断当前视频是真新闻还是假新闻。

**当前视频信息**：
- 发布时间：{current_time}
- 标题：{current_title}
- 关键词：{current_keywords}
- 描述：{current_description}
- 时间演变：{current_temporal}

**参考信息**：

**最相似的真新闻**：
- 标注：{similar_true_annotation}
- 发布时间：{similar_true_time}
- 标题：{similar_true_title}
- 关键词：{similar_true_keywords}
- 描述：{similar_true_description}
- 时间演变：{similar_true_temporal}

**最相似的假新闻**：
- 标注：{similar_fake_annotation}
- 发布时间：{similar_fake_time}
- 标题：{similar_fake_title}
- 关键词：{similar_fake_keywords}
- 描述：{similar_fake_description}
- 时间演变：{similar_fake_temporal}

**请完成以下任务**：

1. **详细推理过程**：分析当前视频内容，对比参考的真假新闻信息，从以下角度进行分析：
   - 检测新闻内容分析
   - 与参考真新闻的相似性分析
   - 与参考假新闻的相似性分析
   - 时间逻辑性分析
   - 内容可信度分析

2. **最终判断**：基于以上分析，判断当前视频是"真"还是"假"

**请按以下格式回答（必须返回JSON格式）**：

{{
  "reasoning": "详细的分析过程",
  "prediction": "真/假"
}}"""

        # Enhanced prompt template with SLM prediction integration
        self.prompt_template_with_slm = """你是一位专业的假新闻检测专家。请分析以下信息来判断当前视频是真新闻还是假新闻。

**多模态模型（文字+视频+音频）初步判断**：
- 预测结果：{slm_prediction}
- 说明：该多模态模型在大规模数据集上表现良好，准确率达到较高水平
- **重要**：该模型在识别假新闻方面表现极为出色，具有很强的模式识别能力
- **重要**：如需推翻该模型的判断，必须提供非常令人信服的证据和充分理由

**当前视频复合帧图片**：
[4个完整复合帧，每个复合帧包含4个关键帧（2x2网格布局）]

**当前视频信息**：
- 发布时间：{current_time}
- 标题：{current_title}
- 关键词：{current_keywords}
- 描述：{current_description}
- 时间演变：{current_temporal}

**参考信息**：

**最相似的真新闻**：
- 标注：{similar_true_annotation}
- 发布时间：{similar_true_time}
- 标题：{similar_true_title}
- 关键词：{similar_true_keywords}
- 描述：{similar_true_description}
- 时间演变：{similar_true_temporal}

**最相似的假新闻**：
- 标注：{similar_fake_annotation}
- 发布时间：{similar_fake_time}
- 标题：{similar_fake_title}
- 关键词：{similar_fake_keywords}
- 描述：{similar_fake_description}
- 时间演变：{similar_fake_temporal}

**请完成以下任务**：

1. **整合所有线索**：综合考虑小模型判断、视觉内容、参考新闻等所有信息

2. **详细推理过程**：分析当前视频内容，对比参考的真假新闻信息，从以下角度进行分析：
   - 小模型判断的合理性分析
   - 检测新闻内容分析
   - 与参考真新闻的相似性分析
   - 与参考假新闻的相似性分析
   - 时间逻辑性分析
   - 内容可信度分析

3. **最终判断**：基于以上分析，判断当前视频是"真"还是"假"
   **重要提醒**：多模态模型在假新闻检测方面极其擅长，推翻其判断需要提供极其充分和令人信服的证据

**请按以下格式回答（必须返回JSON格式）**：

{{
  "reasoning": "详细的分析过程",
  "prediction": "真/假"
}}"""

        # No-image version of SLM prompt template
        self.prompt_template_with_slm_no_image = """你是一位专业的假新闻检测专家。请分析以下信息来判断当前视频是真新闻还是假新闻。

**多模态模型（文字+视频+音频）初步判断**：
- 预测结果：{slm_prediction}
- 说明：该多模态模型在大规模数据集上表现良好，准确率达到较高水平
- **重要**：该模型在识别假新闻方面表现极为出色，具有很强的模式识别能力
- **重要**：如需推翻该模型的判断，必须提供非常令人信服的证据和充分理由

**当前视频信息**：
- 发布时间：{current_time}
- 标题：{current_title}
- 关键词：{current_keywords}
- 描述：{current_description}
- 时间演变：{current_temporal}

**参考信息**：

**最相似的真新闻**：
- 标注：{similar_true_annotation}
- 发布时间：{similar_true_time}
- 标题：{similar_true_title}
- 关键词：{similar_true_keywords}
- 描述：{similar_true_description}
- 时间演变：{similar_true_temporal}

**最相似的假新闻**：
- 标注：{similar_fake_annotation}
- 发布时间：{similar_fake_time}
- 标题：{similar_fake_title}
- 关键词：{similar_fake_keywords}
- 描述：{similar_fake_description}
- 时间演变：{similar_fake_temporal}

**请完成以下任务**：

1. **整合所有线索**：综合考虑小模型判断、参考新闻等所有信息

2. **详细推理过程**：分析当前视频内容，对比参考的真假新闻信息，从以下角度进行分析：
   - 小模型判断的合理性分析
   - 检测新闻内容分析
   - 与参考真新闻的相似性分析
   - 与参考假新闻的相似性分析
   - 时间逻辑性分析
   - 内容可信度分析

3. **最终判断**：基于以上分析，判断当前视频是"真"还是"假"
   **重要提醒**：多模态模型在假新闻检测方面极其擅长，推翻其判断需要提供极其充分和令人信服的证据

**请按以下格式回答（必须返回JSON格式）**：

{{
  "reasoning": "详细的分析过程",
  "prediction": "真/假"
}}"""

        # Enhanced prompt template with Entity-Claims integration
        self.prompt_template_with_entity_claims = """你是一位专业的假新闻检测专家。请分析以下信息来判断当前视频是真新闻还是假新闻。

**当前视频复合帧图片**：
[4个完整复合帧，每个复合帧包含4个关键帧（2x2网格布局）]

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

1. **详细推理过程**：分析当前视频内容，对比参考的真假新闻信息，从以下角度进行分析：
   - 检测新闻内容分析
   - 与参考真新闻的相似性分析：
     a. 视频主题及内容对比
     b. 实体声称point by point分析
   - 与参考假新闻的相似性分析：
     a. 视频主题及内容对比  
     b. 实体声称point by point分析
   - 时间逻辑性分析
   - 内容可信度分析

2. **最终判断**：基于以上分析，判断当前视频是"真"还是"假"

**请按以下格式回答（必须返回JSON格式）**：

{{
  "reasoning": "详细的分析过程",
  "prediction": "真/假"
}}"""

        # No-image version of entity claims prompt template
        self.prompt_template_with_entity_claims_no_image = """你是一位专业的假新闻检测专家。请分析以下信息来判断当前视频是真新闻还是假新闻。

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

1. **详细推理过程**：分析当前视频内容和实体声称，对比参考的真假新闻信息，从以下角度进行分析：
   - 实体声称的真实性和合理性分析
   - 检测新闻内容分析
   - 与参考真新闻的相似性分析
   - 与参考假新闻的相似性分析
   - 时间逻辑性分析
   - 内容可信度分析

2. **最终判断**：基于以上分析，判断当前视频是"真"还是"假"

**请按以下格式回答（必须返回JSON格式）**：

{{
  "reasoning": "详细的分析过程",
  "prediction": "真/假"
}}"""

        # Enhanced prompt template with both SLM and Entity-Claims
        self.prompt_template_with_both = """你是一位专业的假新闻检测专家。请分析以下信息来判断当前视频是真新闻还是假新闻。

**多模态模型（文字+视频+音频）初步判断**：
- 预测结果：{slm_prediction}
- 说明：该多模态模型在大规模数据集上表现良好，准确率达到较高水平
- **重要**：该模型在识别假新闻方面表现极为出色，具有很强的模式识别能力
- **重要**：如需推翻该模型的判断，必须从真假新闻分析或是你自己的可靠世界知识中找到充分且信服的证据

**当前视频复合帧图片**：
[4个完整复合帧，每个复合帧包含4个关键帧（2x2网格布局）]

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

1. **整合所有线索**：综合考虑小模型判断、视觉内容、参考新闻等所有信息

2. **详细推理过程**：分析当前视频内容，对比参考的真假新闻信息，从以下角度进行分析：
   - 检测新闻内容分析
   - 与参考真新闻的相似性分析：
     a. 视频主题及内容对比
     b. 实体声称point by point分析
   - 与参考假新闻的相似性分析：
     a. 视频主题及内容对比
     b. 实体声称point by point分析
   - 时间逻辑性分析
   - 内容可信度分析
   - 小模型判断的合理性分析

3. **最终判断**：基于以上分析，判断当前视频是"真"还是"假"
   **重要提醒**：多模态模型在假新闻检测方面极其擅长，推翻其判断需要从真假新闻分析/或是你自己的可靠世界知识中找到充分且信服的证据

**请按以下格式回答（必须返回JSON格式）**：

{{
  "reasoning": "详细的分析过程",
  "prediction": "真/假"
}}"""

        # No-image version of both SLM and entity claims prompt template
        self.prompt_template_with_both_no_image = """你是一位专业的假新闻检测专家。请分析以下信息来判断当前视频是真新闻还是假新闻。

**多模态模型（文字+视频+音频）初步判断**：
- 预测结果：{slm_prediction}
- 说明：该多模态模型在大规模数据集上表现良好，准确率达到较高水平
- **重要**：该模型在识别假新闻方面表现极为出色，具有很强的模式识别能力
- **重要**：如需推翻该模型的判断，必须提供非常令人信服的证据和充分理由

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
   - 小模型判断的合理性分析
   - 实体声称的真实性和合理性分析
   - 检测新闻内容分析
   - 与参考真新闻的相似性分析
   - 与参考假新闻的相似性分析
   - 时间逻辑性分析
   - 内容可信度分析

3. **最终判断**：基于以上分析，判断当前视频是"真"还是"假"
   **重要提醒**：多模态模型在假新闻检测方面极其擅长，推翻其判断需要提供极其充分和令人信服的证据

**请按以下格式回答（必须返回JSON格式）**：

{{
  "reasoning": "详细的分析过程",
  "prediction": "真/假"
}}"""

        # Get API settings
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Setup AsyncOpenAI client
        base_url = os.getenv('OPENAI_BASE_URL')
        if base_url:
            self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.async_client = AsyncOpenAI(api_key=api_key)
        self.model = 'gpt-4o-mini'
        self.temperature = 0.1
        self.max_tokens = 2000
        
        logger.info("Initialized AsyncOpenAI client for GPT-4o-mini")
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save to bytes
                import io
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                
                # Encode to base64
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return img_base64
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None
    
    def load_video_frames(self, video_id: str) -> List[str]:
        """Load 4 quad images for a video and convert to base64"""
        frames = []
        
        for i in range(4):
            quad_path = self.quads_dir / f"{video_id}_quad_{i}.jpg"
            if quad_path.exists():
                base64_img = self.encode_image_to_base64(str(quad_path))
                if base64_img:
                    frames.append(base64_img)
                else:
                    logger.warning(f"Failed to encode image: {quad_path}")
            else:
                logger.warning(f"Missing quad image: {quad_path}")
        
        return frames
    
    def format_entity_claims(self, entity_claims_data: Dict) -> str:
        """Format entity claims for prompt"""
        if not entity_claims_data or 'entity_claims' not in entity_claims_data:
            return "无相关实体声称信息"
        
        entity_claims = entity_claims_data['entity_claims']
        if not entity_claims:
            return "无相关实体声称信息"
        
        formatted_claims = []
        for entity, claims in entity_claims.items():
            formatted_claims.append(f"【{entity}】")
            for claim in claims:
                formatted_claims.append(f"  - {claim}")
            formatted_claims.append("")  # Add empty line between entities
        
        return "\n".join(formatted_claims)
    
    def load_retrieval_results(self) -> List[Dict]:
        """Load text similarity retrieval results"""
        logger.info(f"Loading retrieval results from: {self.retrieval_results_file}")
        
        with open(self.retrieval_results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        logger.info(f"Loaded {len(results)} retrieval results")
        
        # Limit samples if specified
        if self.max_samples and len(results) > self.max_samples:
            results = results[:self.max_samples]
            logger.info(f"Limited to {len(results)} samples")
        
        return results
    
    def load_checkpoint(self) -> Tuple[List[str], List[Dict]]:
        """Load checkpoint if exists"""
        checkpoint_file = self.checkpoints_dir / "prediction_checkpoint.json"
        
        if checkpoint_file.exists():
            logger.info(f"Loading checkpoint from: {checkpoint_file}")
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            processed_ids = checkpoint.get('processed_video_ids', [])
            results = checkpoint.get('results', [])
            
            logger.info(f"Checkpoint loaded: {len(processed_ids)} processed, {len(results)} results")
            return processed_ids, results
        
        return [], []
    
    def save_checkpoint(self, processed_ids: List[str], results: List[Dict]):
        """Save checkpoint"""
        checkpoint = {
            'processed_video_ids': processed_ids,
            'results': results,
            'total_processed': len(processed_ids)
        }
        
        checkpoint_file = self.checkpoints_dir / "prediction_checkpoint.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Checkpoint saved: {len(processed_ids)} processed")
    
    async def predict_single_video(self, retrieval_result: Dict) -> Dict:
        """Predict single video using async LLM call"""
        video_id = retrieval_result['test_video']['video_id']
        
        try:
            # Load video frames (skip if no_image mode)
            frames = []
            if not self.no_image:
                frames = self.load_video_frames(video_id)
                if not frames:
                    logger.warning(f"No frames found for video: {video_id}")
                    return {
                        'video_id': video_id,
                        'ground_truth': retrieval_result['test_video']['annotation'],
                        'prediction': '错误',
                        'reasoning': '视频帧加载失败',
                        'error': 'No frames found'
                    }
            
            # Prepare prompt data
            test_video = retrieval_result['test_video']
            similar_true = retrieval_result['most_similar_true_news']
            similar_fake = retrieval_result['most_similar_fake_news']
            
            # Get SLM prediction if available
            slm_prediction = None
            if self.use_slm_prediction:
                slm_prediction = self.slm_predictions.get(video_id, "未知")
                if slm_prediction == "未知":
                    logger.warning(f"No SLM prediction found for video: {video_id}")
            
            # Get entity claims if available
            current_entity_claims = ""
            similar_true_entity_claims = ""
            similar_fake_entity_claims = ""
            
            if self.use_entity_claims:
                # Get current video's entity claims
                current_claims_data = self.entity_claims_true.get(video_id) or self.entity_claims_fake.get(video_id)
                current_entity_claims = self.format_entity_claims(current_claims_data)
                
                # Get similar true news entity claims
                similar_true_video_id = similar_true.get('video_id')
                if similar_true_video_id:
                    similar_true_claims_data = self.entity_claims_true.get(similar_true_video_id)
                    similar_true_entity_claims = self.format_entity_claims(similar_true_claims_data)
                
                # Get similar fake news entity claims
                similar_fake_video_id = similar_fake.get('video_id')
                if similar_fake_video_id:
                    similar_fake_claims_data = self.entity_claims_fake.get(similar_fake_video_id)
                    similar_fake_entity_claims = self.format_entity_claims(similar_fake_claims_data)
            
            # Choose prompt template based on enabled features
            use_slm = self.use_slm_prediction and slm_prediction != "未知"
            use_ec = self.use_entity_claims
            
            if use_slm and use_ec:
                # Both SLM and entity claims
                if self.no_image:
                    prompt = self.prompt_template_with_both_no_image.format(
                    slm_prediction=slm_prediction,
                    current_time=test_video['publish_time'],
                    current_title=test_video['title'],
                    current_keywords=test_video['keywords'],
                    current_description=test_video['description'],
                    current_temporal=test_video['temporal_evolution'],
                    current_entity_claims=current_entity_claims,
                    
                    similar_true_annotation=similar_true['annotation'],
                    similar_true_time=similar_true['publish_time'],
                    similar_true_title=similar_true['title'],
                    similar_true_keywords=similar_true['keywords'],
                    similar_true_description=similar_true['description'],
                    similar_true_temporal=similar_true['temporal_evolution'],
                    similar_true_entity_claims=similar_true_entity_claims,
                    
                    similar_fake_annotation=similar_fake['annotation'],
                    similar_fake_time=similar_fake['publish_time'],
                    similar_fake_title=similar_fake['title'],
                    similar_fake_keywords=similar_fake['keywords'],
                    similar_fake_description=similar_fake['description'],
                    similar_fake_temporal=similar_fake['temporal_evolution'],
                    similar_fake_entity_claims=similar_fake_entity_claims
                )
                else:
                    prompt = self.prompt_template_with_both.format(
                        slm_prediction=slm_prediction,
                        current_time=test_video['publish_time'],
                        current_title=test_video['title'],
                        current_keywords=test_video['keywords'],
                        current_description=test_video['description'],
                        current_temporal=test_video['temporal_evolution'],
                        current_entity_claims=current_entity_claims,
                        
                        similar_true_annotation=similar_true['annotation'],
                        similar_true_time=similar_true['publish_time'],
                        similar_true_title=similar_true['title'],
                        similar_true_keywords=similar_true['keywords'],
                        similar_true_description=similar_true['description'],
                        similar_true_temporal=similar_true['temporal_evolution'],
                        similar_true_entity_claims=similar_true_entity_claims,
                        
                        similar_fake_annotation=similar_fake['annotation'],
                        similar_fake_time=similar_fake['publish_time'],
                        similar_fake_title=similar_fake['title'],
                        similar_fake_keywords=similar_fake['keywords'],
                        similar_fake_description=similar_fake['description'],
                        similar_fake_temporal=similar_fake['temporal_evolution'],
                        similar_fake_entity_claims=similar_fake_entity_claims
                    )
            elif use_slm:
                # Only SLM prediction
                if self.no_image:
                    prompt = self.prompt_template_with_slm_no_image.format(
                    slm_prediction=slm_prediction,
                    current_time=test_video['publish_time'],
                    current_title=test_video['title'],
                    current_keywords=test_video['keywords'],
                    current_description=test_video['description'],
                    current_temporal=test_video['temporal_evolution'],
                    
                    similar_true_annotation=similar_true['annotation'],
                    similar_true_time=similar_true['publish_time'],
                    similar_true_title=similar_true['title'],
                    similar_true_keywords=similar_true['keywords'],
                    similar_true_description=similar_true['description'],
                    similar_true_temporal=similar_true['temporal_evolution'],
                    
                    similar_fake_annotation=similar_fake['annotation'],
                    similar_fake_time=similar_fake['publish_time'],
                    similar_fake_title=similar_fake['title'],
                    similar_fake_keywords=similar_fake['keywords'],
                    similar_fake_description=similar_fake['description'],
                    similar_fake_temporal=similar_fake['temporal_evolution']
                )
                else:
                    prompt = self.prompt_template_with_slm.format(
                        slm_prediction=slm_prediction,
                        current_time=test_video['publish_time'],
                        current_title=test_video['title'],
                        current_keywords=test_video['keywords'],
                        current_description=test_video['description'],
                        current_temporal=test_video['temporal_evolution'],
                        
                        similar_true_annotation=similar_true['annotation'],
                        similar_true_time=similar_true['publish_time'],
                        similar_true_title=similar_true['title'],
                        similar_true_keywords=similar_true['keywords'],
                        similar_true_description=similar_true['description'],
                        similar_true_temporal=similar_true['temporal_evolution'],
                        
                        similar_fake_annotation=similar_fake['annotation'],
                        similar_fake_time=similar_fake['publish_time'],
                        similar_fake_title=similar_fake['title'],
                        similar_fake_keywords=similar_fake['keywords'],
                        similar_fake_description=similar_fake['description'],
                        similar_fake_temporal=similar_fake['temporal_evolution']
                    )
            elif use_ec:
                # Only entity claims
                if self.no_image:
                    prompt = self.prompt_template_with_entity_claims_no_image.format(
                    current_time=test_video['publish_time'],
                    current_title=test_video['title'],
                    current_keywords=test_video['keywords'],
                    current_description=test_video['description'],
                    current_temporal=test_video['temporal_evolution'],
                    current_entity_claims=current_entity_claims,
                    
                    similar_true_annotation=similar_true['annotation'],
                    similar_true_time=similar_true['publish_time'],
                    similar_true_title=similar_true['title'],
                    similar_true_keywords=similar_true['keywords'],
                    similar_true_description=similar_true['description'],
                    similar_true_temporal=similar_true['temporal_evolution'],
                    similar_true_entity_claims=similar_true_entity_claims,
                    
                    similar_fake_annotation=similar_fake['annotation'],
                    similar_fake_time=similar_fake['publish_time'],
                    similar_fake_title=similar_fake['title'],
                    similar_fake_keywords=similar_fake['keywords'],
                    similar_fake_description=similar_fake['description'],
                    similar_fake_temporal=similar_fake['temporal_evolution'],
                    similar_fake_entity_claims=similar_fake_entity_claims
                )
                else:
                    prompt = self.prompt_template_with_entity_claims.format(
                        current_time=test_video['publish_time'],
                        current_title=test_video['title'],
                        current_keywords=test_video['keywords'],
                        current_description=test_video['description'],
                        current_temporal=test_video['temporal_evolution'],
                        current_entity_claims=current_entity_claims,
                        
                        similar_true_annotation=similar_true['annotation'],
                        similar_true_time=similar_true['publish_time'],
                        similar_true_title=similar_true['title'],
                        similar_true_keywords=similar_true['keywords'],
                        similar_true_description=similar_true['description'],
                        similar_true_temporal=similar_true['temporal_evolution'],
                        similar_true_entity_claims=similar_true_entity_claims,
                        
                        similar_fake_annotation=similar_fake['annotation'],
                        similar_fake_time=similar_fake['publish_time'],
                        similar_fake_title=similar_fake['title'],
                        similar_fake_keywords=similar_fake['keywords'],
                        similar_fake_description=similar_fake['description'],
                        similar_fake_temporal=similar_fake['temporal_evolution'],
                        similar_fake_entity_claims=similar_fake_entity_claims
                    )
            else:
                # Basic template (no SLM, no entity claims)
                if self.no_image:
                    prompt = self.prompt_template_no_image.format(
                    current_time=test_video['publish_time'],
                    current_title=test_video['title'],
                    current_keywords=test_video['keywords'],
                    current_description=test_video['description'],
                    current_temporal=test_video['temporal_evolution'],
                    
                    similar_true_annotation=similar_true['annotation'],
                    similar_true_time=similar_true['publish_time'],
                    similar_true_title=similar_true['title'],
                    similar_true_keywords=similar_true['keywords'],
                    similar_true_description=similar_true['description'],
                    similar_true_temporal=similar_true['temporal_evolution'],
                    
                    similar_fake_annotation=similar_fake['annotation'],
                    similar_fake_time=similar_fake['publish_time'],
                    similar_fake_title=similar_fake['title'],
                    similar_fake_keywords=similar_fake['keywords'],
                    similar_fake_description=similar_fake['description'],
                    similar_fake_temporal=similar_fake['temporal_evolution']
                )
                else:
                    prompt = self.prompt_template.format(
                        current_time=test_video['publish_time'],
                        current_title=test_video['title'],
                        current_keywords=test_video['keywords'],
                        current_description=test_video['description'],
                        current_temporal=test_video['temporal_evolution'],
                        
                        similar_true_annotation=similar_true['annotation'],
                        similar_true_time=similar_true['publish_time'],
                        similar_true_title=similar_true['title'],
                        similar_true_keywords=similar_true['keywords'],
                        similar_true_description=similar_true['description'],
                        similar_true_temporal=similar_true['temporal_evolution'],
                        
                        similar_fake_annotation=similar_fake['annotation'],
                        similar_fake_time=similar_fake['publish_time'],
                        similar_fake_title=similar_fake['title'],
                        similar_fake_keywords=similar_fake['keywords'],
                        similar_fake_description=similar_fake['description'],
                        similar_fake_temporal=similar_fake['temporal_evolution']
                    )
            
            # Prepare messages (with or without images)
            if self.no_image:
                # Text-only mode
                messages = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            else:
                # With images mode
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                
                # Add images
                for img_base64 in frames:
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}",
                            "detail": "high"
                        }
                    })
            
            # Call async API with JSON format
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            llm_response = response.choices[0].message.content
            
            # Parse response
            prediction = self.parse_llm_response(llm_response)
            
            # Create result
            result = {
                'video_id': video_id,
                'ground_truth': test_video['annotation'],
                'prediction': prediction.get('prediction', '未知'),
                'reasoning': prediction.get('reasoning', ''),
                'llm_response': llm_response,
                'similar_true_news': similar_true,
                'similar_fake_news': similar_fake,
                'test_video_info': test_video
            }
            
            # Add SLM prediction info if available
            if self.use_slm_prediction:
                result['slm_prediction'] = slm_prediction
                result['used_slm_prompt'] = slm_prediction != "未知"
            
            # Add entity claims info if available
            if self.use_entity_claims:
                result['used_entity_claims'] = True
                result['current_entity_claims'] = current_entity_claims != "无相关实体声称信息"
                result['similar_true_entity_claims'] = similar_true_entity_claims != "无相关实体声称信息"
                result['similar_fake_entity_claims'] = similar_fake_entity_claims != "无相关实体声称信息"
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting video {video_id}: {e}")
            return {
                'video_id': video_id,
                'ground_truth': retrieval_result['test_video']['annotation'],
                'prediction': '错误',
                'reasoning': f'预测过程出错: {str(e)}',
                'error': str(e)
            }
    
    def parse_llm_response(self, response: str) -> Dict:
        """Parse LLM JSON response to extract reasoning and prediction"""
        try:
            # Parse JSON response
            result = json.loads(response.strip())
            
            # Extract prediction and reasoning
            prediction = result.get('prediction', '未知').strip()
            reasoning = result.get('reasoning', '').strip()
            
            # Normalize prediction values
            if '真' in prediction:
                prediction = '真'
            elif '假' in prediction:
                prediction = '假'
            else:
                prediction = '未知'
            
            return {
                'reasoning': reasoning,
                'prediction': prediction
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response}")
            
            # Fallback to text parsing for malformed JSON
            try:
                # Try to extract prediction from text
                prediction = "未知"
                if '真' in response and '假' not in response:
                    prediction = '真'
                elif '假' in response and '真' not in response:
                    prediction = '假'
                elif '假' in response and '真' in response:
                    # If both present, look for the final judgment
                    lines = response.lower().split('\n')
                    for line in lines:
                        if 'prediction' in line or '判断' in line or '结论' in line:
                            if '假' in line:
                                prediction = '假'
                                break
                            elif '真' in line:
                                prediction = '真'
                                break
                
                return {
                    'reasoning': response,  # Return full response as reasoning
                    'prediction': prediction
                }
            except Exception as fallback_e:
                logger.error(f"Fallback parsing also failed: {fallback_e}")
                return {
                    'reasoning': response,
                    'prediction': '未知'
                }
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return {
                'reasoning': response,
                'prediction': '未知'
            }
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate evaluation metrics"""
        # Filter out error cases
        valid_results = [r for r in results if r.get('prediction') not in ['错误', '未知']]
        
        if not valid_results:
            logger.warning("No valid predictions found")
            return {}
        
        # Prepare labels
        ground_truth = [r['ground_truth'] for r in valid_results]
        predictions = [r['prediction'] for r in valid_results]
        
        # Calculate metrics
        accuracy = accuracy_score(ground_truth, predictions)
        macro_f1 = f1_score(ground_truth, predictions, average='macro')
        macro_precision = precision_score(ground_truth, predictions, average='macro')
        macro_recall = recall_score(ground_truth, predictions, average='macro')
        
        # Per-class metrics
        labels = ['真', '假']
        per_class_f1 = f1_score(ground_truth, predictions, labels=labels, average=None)
        per_class_precision = precision_score(ground_truth, predictions, labels=labels, average=None)
        per_class_recall = recall_score(ground_truth, predictions, labels=labels, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(ground_truth, predictions, labels=labels)
        
        metrics = {
            'total_samples': len(results),
            'valid_predictions': len(valid_results),
            'accuracy': float(accuracy),
            'macro_metrics': {
                'f1': float(macro_f1),
                'precision': float(macro_precision),
                'recall': float(macro_recall)
            },
            'per_class_metrics': {
                '真': {
                    'f1': float(per_class_f1[0]),
                    'precision': float(per_class_precision[0]),
                    'recall': float(per_class_recall[0])
                },
                '假': {
                    'f1': float(per_class_f1[1]),
                    'precision': float(per_class_precision[1]),
                    'recall': float(per_class_recall[1])
                }
            },
            'confusion_matrix': {
                'labels': labels,
                'matrix': cm.tolist()
            }
        }
        
        return metrics
    
    def save_results(self, results: List[Dict], metrics: Dict):
        """Save all results"""
        # Save full results
        full_results_file = self.predictions_dir / "full_responses.jsonl"
        with open(full_results_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Save metrics
        metrics_file = self.metrics_dir / "evaluation_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        # Save prediction summary
        summary_data = []
        for result in results:
            summary_data.append({
                'video_id': result['video_id'],
                'ground_truth': result['ground_truth'],
                'prediction': result['prediction'],
                'correct': result['ground_truth'] == result['prediction'],
                'similar_true_id': result.get('similar_true_news', {}).get('video_id', ''),
                'similar_fake_id': result.get('similar_fake_news', {}).get('video_id', ''),
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.predictions_dir / "prediction_summary.csv"
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        
        logger.info(f"Results saved:")
        logger.info(f"  Full responses: {full_results_file}")
        logger.info(f"  Metrics: {metrics_file}")
        logger.info(f"  Summary: {summary_file}")
        
        # Print metrics summary
        if metrics:
            logger.info(f"\n=== Evaluation Metrics ===")
            logger.info(f"Total samples: {metrics['total_samples']}")
            logger.info(f"Valid predictions: {metrics['valid_predictions']}")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Macro F1: {metrics['macro_metrics']['f1']:.4f}")
            logger.info(f"Macro Precision: {metrics['macro_metrics']['precision']:.4f}")
            logger.info(f"Macro Recall: {metrics['macro_metrics']['recall']:.4f}")
            
            logger.info(f"\nPer-class metrics:")
            for label in ['真', '假']:
                class_metrics = metrics['per_class_metrics'][label]
                logger.info(f"  {label}: F1={class_metrics['f1']:.4f}, P={class_metrics['precision']:.4f}, R={class_metrics['recall']:.4f}")
    
    async def run_prediction(self):
        """Run the complete prediction process asynchronously"""
        logger.info("Starting LLM video prediction...")
        
        # Load retrieval results
        retrieval_results = self.load_retrieval_results()
        
        # Load checkpoint
        processed_ids, results = self.load_checkpoint()
        processed_set = set(processed_ids)
        
        # Filter unprocessed results
        unprocessed_results = [
            r for r in retrieval_results 
            if r['test_video']['video_id'] not in processed_set
        ]
        
        logger.info(f"Total samples: {len(retrieval_results)}")
        logger.info(f"Already processed: {len(processed_ids)}")
        logger.info(f"Remaining to process: {len(unprocessed_results)}")
        
        # Process unprocessed samples with async concurrency
        max_concurrent = 100  # 同时最多10个请求
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(retrieval_result, pbar):
            async with semaphore:
                result = await self.predict_single_video(retrieval_result)
                await asyncio.sleep(0.5)  # Rate limiting
                pbar.update(1)
                return result
        
        # Process in batches for checkpoint management
        batch_size = 20  # Checkpoint every 20 videos
        
        with tqdm(total=len(unprocessed_results), desc="Predicting videos") as pbar:
            for i in range(0, len(unprocessed_results), batch_size):
                batch_end = min(i + batch_size, len(unprocessed_results))
                batch = unprocessed_results[i:batch_end]
                
                logger.info(f"Processing batch {i//batch_size + 1}: videos {i+1}-{batch_end}")
                
                # Create async tasks for this batch
                tasks = [process_with_semaphore(retrieval_result, pbar) for retrieval_result in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for retrieval_result, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to process video {retrieval_result['test_video']['video_id']}: {result}")
                        # Create error result
                        error_result = {
                            'video_id': retrieval_result['test_video']['video_id'],
                            'ground_truth': retrieval_result['test_video']['annotation'],
                            'prediction': '错误',
                            'reasoning': f'处理异常: {str(result)}',
                            'error': str(result)
                        }
                        results.append(error_result)
                        processed_ids.append(error_result['video_id'])
                    elif result:
                        results.append(result)
                        processed_ids.append(result['video_id'])
                
                # Save checkpoint after each batch
                self.save_checkpoint(processed_ids, results)
                logger.info(f"Batch {i//batch_size + 1} completed. Total processed: {len(processed_ids)}")
        
        logger.info("All videos processed!")
        
        # Final save
        if unprocessed_results:
            self.save_checkpoint(processed_ids, results)
        
        # Calculate metrics
        metrics = self.calculate_metrics(results)
        
        # Save all results
        self.save_results(results, metrics)
        
        logger.info("LLM video prediction completed!")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="LLM-based video fake news prediction")
    parser.add_argument("--retrieval-file", type=str, 
                       default="text_similarity_results/text_similarity_retrieval_chinese-clip-vit-large-patch14.json",
                       help="Path to text similarity retrieval results")
    parser.add_argument("--output-dir", type=str, 
                       default="result/FakeSV/llm_prediction_results",
                       help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process (for testing)")
    parser.add_argument("--checkpoint-interval", type=int, default=10,
                       help="Save checkpoint every N samples")
    parser.add_argument("--use-slm-prediction", action="store_true",
                       help="Use small model predictions as input to LLM")
    parser.add_argument("--slm-prediction-file", type=str, 
                       default="result/FakeSV/slm_prediction.json",
                       help="Path to small model prediction file")
    parser.add_argument("--use-entity-claims", action="store_true",
                       help="Use entity claims information as input to LLM")
    parser.add_argument("--entity-claims-true-file", type=str, 
                       default="data/FakeSV/entity_claims/video_extractions.jsonl",
                       help="Path to true news entity claims file")
    parser.add_argument("--entity-claims-fake-file", type=str, 
                       default="data/FakeSV/entity_claims/fake_entity_claims.jsonl",
                       help="Path to fake news entity claims file")
    parser.add_argument("--no-image", action="store_true",
                       help="Disable image input to LLM (text-only mode)")
    
    args = parser.parse_args()
    
    # Modify output directory based on enabled features
    output_dir = args.output_dir
    suffixes = []
    
    if args.use_slm_prediction:
        suffixes.append("slm")
    
    if args.use_entity_claims:
        suffixes.append("entity_claims")
    
    if args.no_image:
        suffixes.append("no_image")
    
    if suffixes:
        output_dir = output_dir.rstrip('/') + "_with_" + "_".join(suffixes)
    
    # Create predictor
    predictor = LLMVideoPredictor(
        retrieval_results_file=args.retrieval_file,
        output_dir=output_dir,
        max_samples=args.max_samples,
        checkpoint_interval=args.checkpoint_interval,
        use_slm_prediction=args.use_slm_prediction,
        slm_prediction_file=args.slm_prediction_file,
        use_entity_claims=args.use_entity_claims,
        entity_claims_true_file=args.entity_claims_true_file,
        entity_claims_fake_file=args.entity_claims_fake_file,
        no_image=args.no_image
    )
    
    # Run prediction asynchronously
    await predictor.run_prediction()


if __name__ == "__main__":
    asyncio.run(main())