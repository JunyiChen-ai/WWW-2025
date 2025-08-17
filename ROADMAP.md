# Development ROADMAP

## Task 1: Cross-Modal Transformer Enhancement

### Objective
Add `cross_modal` hyperparameter to enable transformer-based fusion of visual and text tokens before classification, while maintaining backward compatibility with current averaging-based fusion.

### Implementation TODOs

#### 1.1 [✅] Add Cross-Modal Transformer Module
**File**: `src/model/ExMRD.py`
- Create `CrossModalTransformer` class:
  ```python
  class CrossModalTransformer(nn.Module):
      def __init__(self, d_model=256, n_heads=8, n_layers=4, d_ff=1024, dropout=0.1):
          # Transformer encoder with nn.TransformerEncoderLayer
          # cls_token: nn.Parameter(torch.zeros(1, 1, d_model))
          # modality_embeddings: nn.Embedding(2, d_model)  # 0=visual, 1=text
          # position_embeddings: nn.Parameter(torch.zeros(21, d_model))  # CLS + 16 vis + 4 text
  ```
  - Input processing:
    - Concatenate: [CLS] + visual_tokens[B,16,256] + text_tokens[B,4,256] → [B,21,256]
    - Add modality embeddings: visual tokens get 0, text tokens get 1
    - Add learnable position embeddings
  - Output: Extract [CLS] token at position 0 → [B,256]

#### 1.2 [✅] Update ExMRD Model Forward Pass
**File**: `src/model/ExMRD.py` (lines ~91-134)
- Modify token processing to preserve sequence dimension:
  ```python
  if self.cross_modal:
      # Visual: [B,16,1024] → temporal_pe → linear → [B,16,256] (no mean pooling)
      fea_frames = self.temporal_pe(fea_frames)
      visual_tokens = self.linear_video(fea_frames)  # Keep [B,16,256]
      
      # Text: Stack 4 CoT components → [B,4,256] (no mean pooling)
      text_tokens = torch.stack([fea_ocr, fea_caption, fea_comsense, fea_causal], dim=1)
      
      # Cross-modal fusion
      fused_features = self.cross_modal_transformer(visual_tokens, text_tokens)  # [B,256]
      output = self.classifier(fused_features)
  else:
      # Current approach: mean pooling then averaging (lines 130-133)
  ```

#### 1.3 [✅] Update Configuration System
**Files**: `src/config/ExMRD_*.yaml`
- Add under `para` section (after line 11):
  ```yaml
  para:
    hid_dim: 256
    dropout: 0.3
    text_encoder: "OFA-Sys/chinese-clip-vit-large-patch14"
    num_frozen_layers: 4
    # New cross-modal parameters
    cross_modal: false  # Toggle transformer fusion
    cross_modal_layers: 4
    cross_modal_heads: 8
    cross_modal_dim_ff: 1024
    cross_modal_dropout: 0.1
  ```

#### 1.4 [✅] Update Model Initialization
**File**: `src/main.py`
- Locate model initialization (search for `model = ExMRD`)
- Pass new parameters from config:
  ```python
  model = ExMRD(
      hid_dim=cfg.para.hid_dim,
      dropout=cfg.para.dropout,
      text_encoder=cfg.para.text_encoder,
      num_frozen_layers=cfg.para.num_frozen_layers,
      ablation=cfg.get('ablation', 'No'),
      ablation_no_cot=cfg.data.get('ablation_no_cot', False),
      # New parameters
      cross_modal=cfg.para.get('cross_modal', False),
      cross_modal_config={
          'n_layers': cfg.para.get('cross_modal_layers', 4),
          'n_heads': cfg.para.get('cross_modal_heads', 8),
          'd_ff': cfg.para.get('cross_modal_dim_ff', 1024),
          'dropout': cfg.para.get('cross_modal_dropout', 0.1)
      }
  )
  ```

#### 1.5 [✅] Test Implementation
- **Backward Compatibility Test**:
  ```bash
  python src/main.py --config-name ExMRD_FakeSV para.cross_modal=false
  # Should produce identical results to current implementation
  ```
- **New Feature Test**:
  ```bash
  python src/main.py --config-name ExMRD_FakeSV para.cross_modal=true
  # Monitor: GPU memory (+~30%), training time (+~40%), F1 score improvement
  ```
- **Ablation Study**:
  ```bash
  # Test different transformer configurations
  python src/main.py para.cross_modal=true para.cross_modal_layers=2
  python src/main.py para.cross_modal=true para.cross_modal_heads=4
  ```

### Technical Notes
- Memory complexity: O(n²d) where n=21 tokens, d=256 dims
- FLOPs increase: ~21² × 256 × 4 layers = ~450K per sample
- Attention weights can be extracted for explainability analysis
- Consider gradient checkpointing if OOM occurs with batch_size=32

---
**Status**: Awaiting approval to proceed

## Task 2: Pre-trained Multimodal Model Integration

### Objective
Replace the current text encoder (Chinese-CLIP BERT) and naive cross-modal transformer with a pre-trained multimodal transformer model for end-to-end optimization. The multimodal model will encode text as token sequences and process both visual and text tokens together.

### Architecture Analysis

#### Current Flow:
```
Text (4 CoT components) → Chinese-CLIP BERT → Linear layers → 4×[B,256] → Average → [B,256]
Visual (16 frames) → Temporal PE → Linear → [B,16,256] → Mean → [B,256]
Fusion: Simple average or CrossModalTransformer → [B,256] → MLP classifier
```

#### Proposed Flow:
```
Text (4 CoT components) → Multimodal model tokenizer → Text tokens [B,4,seq_len]
Visual (16 frames) → Multimodal model vision encoder → Visual tokens [B,16,768/1024]
Joint processing: Multimodal transformer → Fused representation [B,d_model] → MLP classifier
```

### Implementation TODOs

#### 2.1 [ ] Select and Analyze Pre-trained Multimodal Model
**Target Models** (in priority order):
1. **CLIP-based models** (OpenAI CLIP, Chinese-CLIP multimodal)
   - Good text+vision understanding
   - Existing Chinese-CLIP familiarity
   - Lightweight integration
2. **BLIP-2/InstructBLIP** (Salesforce)
   - Strong multimodal reasoning
   - Q-Former architecture for cross-modal fusion
   - Good for Chinese text with appropriate tokenizer
3. **LLaVA/Chinese-LLaVA**
   - Instruction-following multimodal model
   - Strong reasoning capabilities
   - May be overkill for classification task

**Key Requirements**:
- Support for both Chinese and English text
- Efficient batch processing
- Reasonable model size (<10B parameters)
- Available pre-trained weights

#### 2.2 [ ] Design Multimodal Model Wrapper
**File**: `src/model/MultimodalTransformer.py`
```python
class PretrainedMultimodalModel(nn.Module):
    def __init__(self, model_name, freeze_layers=None, output_dim=256):
        # Load pre-trained multimodal model
        # Configure layer freezing for efficiency
        # Add projection layer to match ExMRD dimensions
        
    def encode_text_tokens(self, text_inputs):
        # Tokenize and encode text to token sequences
        # Return: [batch_size, num_texts, seq_len, d_model]
        
    def encode_visual_tokens(self, visual_features):
        # Process visual features through vision encoder
        # Return: [batch_size, num_frames, d_model]
        
    def forward(self, text_inputs, visual_features):
        # Joint multimodal processing
        # Return: [batch_size, d_model]
```

#### 2.3 [ ] Update ExMRD Architecture
**File**: `src/model/ExMRD.py`
- Add new hyperparameter: `pretrained_multimodal`
- Conditional model selection:
  ```python
  if self.pretrained_multimodal:
      # Use pre-trained multimodal model
      self.multimodal_encoder = PretrainedMultimodalModel(...)
  elif self.cross_modal:
      # Use naive CrossModalTransformer (current Task 1)
  else:
      # Use original averaging approach
  ```
- Update forward pass with three pathways

#### 2.4 [ ] Configuration Parameters
**Files**: `src/config/ExMRD_*.yaml`
```yaml
para:
  # Existing parameters...
  
  # Pre-trained multimodal model parameters
  pretrained_multimodal: false  # Toggle pre-trained multimodal model
  multimodal_model_name: "openai/clip-vit-large-patch14"  # or "Salesforce/blip2-opt-2.7b"
  multimodal_freeze_layers: 8  # Number of layers to freeze
  multimodal_output_dim: 256   # Output projection dimension
  multimodal_max_text_len: 128  # Max text sequence length
```

#### 2.5 [ ] Data Pipeline Updates
**File**: `src/data/ExMRD_data.py`
- Update collator to handle multimodal model tokenization
- Preserve text as raw strings for multimodal tokenizer
- Ensure visual features are compatible with multimodal vision encoder

#### 2.6 [ ] Training Optimizations
**Considerations**:
- **Memory Management**: Pre-trained models are large; consider gradient checkpointing
- **Learning Rate Scheduling**: Different LR for pre-trained vs. new layers
- **Mixed Precision**: Use FP16/BF16 for efficiency
- **Layer Freezing Strategy**: Freeze early layers, fine-tune later ones

#### 2.7 [ ] Integration Testing
- **Backward Compatibility**: Ensure `pretrained_multimodal=false` works unchanged
- **Parameter Counting**: Verify significant parameter increase with multimodal model
- **Performance Comparison**: 
  - Baseline (averaging)
  - CrossModalTransformer (Task 1)
  - Pre-trained multimodal (Task 2)
- **Memory Profiling**: Monitor GPU memory usage

### Expected Benefits
1. **Stronger Representations**: Pre-trained multimodal features vs. simple averaging
2. **Better Cross-modal Understanding**: Joint optimization vs. separate encoders
3. **Transfer Learning**: Leverage large-scale pre-training for rumor detection
4. **End-to-end Optimization**: Unified model vs. pipeline approach

### Technical Challenges
1. **Model Size**: Pre-trained models may be 2-10x larger than current setup
2. **Tokenization Alignment**: Handling Chinese CoT text with different tokenizers
3. **Visual Feature Compatibility**: Current CLIP features vs. multimodal vision encoder
4. **Training Stability**: Balancing pre-trained and task-specific components

### Success Metrics
- No regression when `pretrained_multimodal=false`
- Significant parameter increase (expected: +100M to +2B parameters)
- F1 score improvement over naive cross-modal transformer
- Training remains stable and converges within reasonable time

---
**Status**: ✅ Completed

## Task 3: Enhanced Video Preprocessing Pipeline with Frame-Level Audio Processing

### Objective
Create a comprehensive preprocessing pipeline that extracts multimodal features at both frame-level and global-level for FakeSV dataset, enabling fine-grained temporal alignment between visual, audio, and text modalities.

### Implementation TODOs

#### 3.1 [✅] Extract and Organize Raw Video Data
**Target**: Extract videos from downloaded zip file to proper directory structure
- [✅] Unzip `data/FakeSV/raw_videos.zip` to `data/FakeSV/videos/`
- [✅] Verify video file structure matches expected format (*.mp4 files)
- [✅] Check video count against dataset metadata: 5,495 videos extracted (22GB)

#### 3.2 [✅] Video Frame Extraction with Temporal Alignment
**Target**: Extract 16 frames per video with precise temporal sampling
- [✅] Run frame extraction: `python preprocess/extract_frame.py`
- [✅] Output location: `data/FakeSV/frames_16/` (all videos completed)
- [✅] Record frame timestamps for audio alignment
- [✅] Frame format: JPG images with temporal metadata
- [✅] Output: Frame timestamp mapping for audio synchronization

#### 3.3 [✅] Frame-Aligned Audio Processing (In-Memory)
**Target**: Process audio segments corresponding to each video frame without saving intermediate files
- [✅] Extract full audio: `python preprocess/video_to_wav.py` (all 5,495 videos completed)
- [✅] Create new script: `preprocess/audio_frame_processing.py` with enhanced anomaly detection
- [✅] Fixed dtype compatibility issue: Whisper float16/float32 mismatch resolved
- [✅] Added incremental processing support: resume from existing results
- [✅] Output structure:
  ```
  data/FakeSV/audios/
  └── {video_id}_full.wav          # Global audio only (5,495 files)
  ```
- [✅] Frame-level audio features saved directly to feature tensors (no intermediate .wav files)

#### 3.4 [🔄] Frame-Level and Global Audio Transcription (In-Memory)
**Target**: Generate transcripts at both frame-level and global level
- [ ] **Global transcription**: `python preprocess/wav_to_transcript.py`
  - Input: `{video_id}_full.wav`
  - Output: `data/FakeSV/transcript_global.jsonl`
- [🔄] **Frame-level transcription**: Integrate into `preprocess/audio_frame_processing.py`
  - Process: In-memory audio segments → Whisper transcription
  - Output: `data/FakeSV/transcript_frames.jsonl` (currently in progress)
  - Format: `[{'vid': video_id, 'frame_transcripts': [text0, text1, ..., text15]}]`
  - Status: Currently running with fixed dtype issue

#### 3.5 [✅] OCR Text Extraction (Frame-Level)
**Target**: Extract on-screen text from individual video frames
- [✅] Create new script: `preprocess/frames_to_ocr.py` for FakeSV
- [✅] Process each of 16 frames individually using EasyOCR
- [✅] Output: `data/FakeSV/ocr_frames.jsonl` (processed available frames)
- [✅] Format: `[{'vid': video_id, 'frame_ocr': [ocr0, ocr1, ..., ocr15]}]`
- [✅] Also generate global OCR: `data/FakeSV/ocr_global.jsonl`

#### 3.6 [ ] Vision Feature Extraction (Frame-Level)
**Target**: Generate visual features for each frame using Chinese-CLIP
- [ ] Run visual feature extraction: `python preprocess/make_video_feature.py`
- [ ] Backbone: Chinese-CLIP ViT (`OFA-Sys/chinese-clip-vit-large-patch14`)
- [ ] Output: `data/FakeSV/fea/visual_features.pt`
- [ ] Feature dimensions: `[num_videos, 16, 1024]` (frame-level features)

#### 3.7 [ ] Text Feature Extraction (Frame-Level and Global)
**Target**: Generate text embeddings at multiple granularities
- [ ] **Frame-level text features**:
  - Combine: `title + frame_ocr[i] + frame_transcript[i]` for each frame
  - Use Chinese-CLIP text encoder
  - Output: `data/FakeSV/fea/text_features_frames.pt`
  - Dimensions: `[num_videos, 16, 768]`
- [ ] **Global text features**:
  - Combine: `title + global_ocr + global_transcript`
  - Output: `data/FakeSV/fea/text_features_global.pt`
  - Dimensions: `[num_videos, 768]`

#### 3.8 [ ] Audio Feature Extraction with Wav2Vec (Frame-Level and Global)
**Target**: Extract audio features using wav2vec2-base-960h at multiple levels
- [ ] Install dependencies: `transformers`, `torchaudio`
- [ ] Integrate into: `preprocess/audio_frame_processing.py`
- [ ] Model: `facebook/wav2vec2-base-960h`
- [ ] **Frame-level audio features**:
  - Process: In-memory audio segments (aligned to frame timestamps)
  - Pipeline: Full audio → segment in memory → wav2vec → aggregate features
  - Output: `data/FakeSV/fea/audio_features_frames.pt`
  - Dimensions: `[num_videos, 16, 768]`
- [ ] **Global audio features**:
  - Process: `{video_id}_full.wav`
  - Output: `data/FakeSV/fea/audio_features_global.pt`
  - Dimensions: `[num_videos, 768]`

#### 3.9 [ ] Enhanced Data Organization Structure
**Target**: Organize all features with frame-level and global variants
```
data/FakeSV/
├── videos/                           # Raw MP4 files
├── frames_16/                        # Video frames with timestamps
│   ├── {video_id}/
│   │   ├── frame_000.jpg (+ timestamp)
│   │   └── ... (16 frames)
├── audios/                          # Audio files
│   └── {video_id}_full.wav         # Global audio only
├── fea/                            # Processed features
│   ├── visual_features.pt          # [N, 16, 1024] - frame visual
│   ├── text_features_frames.pt     # [N, 16, 768] - frame text  
│   ├── text_features_global.pt     # [N, 768] - global text
│   ├── audio_features_frames.pt    # [N, 16, 768] - frame audio
│   └── audio_features_global.pt    # [N, 768] - global audio
├── ocr_frames.jsonl                # Frame-level OCR text
├── ocr_global.jsonl                # Global OCR text  
├── transcript_frames.jsonl         # Frame-level transcripts
├── transcript_global.jsonl         # Global transcripts
└── data.jsonl                      # Original metadata
```

#### 3.10 [ ] Audio-Frame Alignment Algorithm (In-Memory Processing)
**Target**: Precisely align audio segments with video frame timestamps for in-memory processing
- [ ] Integrate into `preprocess/audio_frame_processing.py`
- [ ] Algorithm:
  ```python
  def process_frame_aligned_audio(video_path, frame_timestamps):
      # Load full audio in memory
      # For each frame timestamp:
      #   - Calculate audio segment boundaries  
      #   - Extract audio chunk in memory (numpy array)
      #   - Process through wav2vec and whisper immediately
      #   - Store only final features/transcripts
      # Return: frame_features, frame_transcripts
  ```
- [ ] Support variable frame durations based on video length
- [ ] Process segments sequentially to minimize memory usage

#### 3.11 [ ] Raw Data Preservation
**Target**: Keep raw extracted data alongside processed features
- [ ] **Raw frame transcripts**: Store in `data/FakeSV/raw_transcripts_frames.json`
- [ ] **Raw global transcript**: Store in `data/FakeSV/raw_transcript_global.json`
- [ ] **Raw frame OCR**: Store in `data/FakeSV/raw_ocr_frames.json`
- [ ] **Raw global OCR**: Store in `data/FakeSV/raw_ocr_global.json`
- [ ] **Frame metadata**: Store timing and extraction info

#### 3.12 [ ] Multi-Granularity Data Loader
**Target**: Create flexible data loader supporting different temporal granularities
- [ ] Create `src/data/MultiGranularityDataLoader.py`
- [ ] Support modes:
  - `'frame'`: Load frame-level features `[B, 16, feat_dim]`
  - `'global'`: Load global features `[B, feat_dim]`
  - `'mixed'`: Load both frame and global features
- [ ] Modality selection: visual, text, audio, or combinations
- [ ] Temporal alignment validation across modalities

#### 3.13 [ ] Feature Quality Validation
**Target**: Comprehensive validation of extracted features
- [ ] **Temporal alignment check**: Verify audio-frame synchronization
- [ ] **Feature completeness**: Ensure all videos have all modalities
- [ ] **Dimension validation**: Check tensor shapes match expectations
- [ ] **Content quality**: Sample-check transcripts and OCR accuracy
- [ ] Generate processing report with statistics

### Technical Implementation Details

#### Frame-Level Audio Processing Script (In-Memory)
```python
# preprocess/audio_frame_processing.py
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model, WhisperProcessor, WhisperForConditionalGeneration

def process_frame_aligned_audio(video_path, frame_timestamps):
    """Process audio segments aligned with video frames in-memory"""
    # Load full audio
    audio, sr = librosa.load(video_path, sr=16000)
    
    frame_features = []
    frame_transcripts = []
    
    for start_time, end_time in frame_timestamps:
        # Extract segment in memory (numpy array)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = audio[start_sample:end_sample]
        
        # Process through wav2vec
        features = wav2vec_model(segment)  # Extract features
        frame_features.append(features)
        
        # Generate transcript with whisper
        transcript = whisper_model(segment)  # Get transcript
        frame_transcripts.append(transcript)
    
    return torch.stack(frame_features), frame_transcripts
```

#### Memory and Storage Requirements (Optimized)
- **Visual features**: `[16K, 16, 1024]` × 4 bytes ≈ 1GB
- **Text features (frame)**: `[16K, 16, 768]` × 4 bytes ≈ 800MB
- **Text features (global)**: `[16K, 768]` × 4 bytes ≈ 50MB
- **Audio features (frame)**: `[16K, 16, 768]` × 4 bytes ≈ 800MB  
- **Audio features (global)**: `[16K, 768]` × 4 bytes ≈ 50MB
- **Global audio files**: 16K videos × 5MB avg ≈ 80GB
- **Video frames**: 16K videos × 16 frames × 100KB ≈ 25GB
- **Text data (JSON)**: Raw transcripts, OCR ≈ 1GB
- **Total storage**: ~4GB for features + ~106GB for raw data = **~110GB total**

#### Processing Pipeline Order (Optimized)
1. Extract raw videos and frames (with timestamps)
2. Extract global audio files
3. **In-memory frame-level processing**: For each video:
   - Load audio → segment by timestamps → wav2vec features + whisper transcripts → save results
4. Extract OCR text (frame-level and global)  
5. Generate visual features (frame-level)
6. Generate text features (frame-level and global)
7. Process global audio features
8. Validate and organize all outputs

### Success Criteria
- [ ] Frame-level temporal alignment achieved across all modalities
- [ ] Both frame-level `[N, 16, feat_dim]` and global `[N, feat_dim]` features extracted
- [ ] Audio segments properly aligned with video frame timestamps
- [ ] Raw data preserved alongside processed features
- [ ] Multi-granularity data loader supports flexible feature access
- [ ] Complete processing pipeline documented and reproducible

---
**Status**: Awaiting approval to proceed

## Task 4: Entity-Claim Knowledge Base Extraction for FakeSV Videos

### Objective
Extract entities and their associated factual claims from FakeSV videos using GPT-4o-mini. Build a streamlined knowledge base for RAG-based fact-checking, where each entity has contextual descriptions and point-by-point claims with source attribution.

### Architecture Overview

#### Input Components per Video:
1. **Visual**: 4 composite frames (2x2 grid layout, covering 16 frames total)
2. **Text**: Title, keywords, overall transcript
3. **Metadata**: Video label (真/假/辟谣)

#### Output Components:
1. **Per Video**:
   - Entities mentioned/shown
   - Factual claims made (point by point)
   - Video description
   - Temporal evolution

2. **Aggregated Knowledge Base**:
   - Entity → Description with context
   - Entity → List of all factual claims
   - Claim → Source video IDs

### Implementation (Completed)

#### 4.1 [✅] Phase 1: Extract from True Videos (annotation='真')
**File**: `preprocess/extract_entity_claims.py`
- Implemented complete extraction pipeline with GPT-4o-mini
- Processes videos with checkpoint support and progress tracking
- Generates entity-claim mappings with 5W principle (Who/What/When/Where/Why)
- Includes entity merging and normalization
- Creates LLM-based entity descriptions from aggregated claims
- Outputs: `video_extractions.jsonl`, `entity_knowledge_base.json`, `claim_index.json`

#### 4.2 [🔄] Phase 2: Extract from Fake Videos (annotation='假')
**File**: `preprocess/extract_entity_claims.py`
**Implementation Plan**:

##### Step 1: Initial Description Extraction
- Input: Fake news videos with title, keywords, frames
- Process with GPT-4o-mini to extract ONLY:
  - Video description
  - Temporal evolution
- NO entity/claim extraction in this step

##### Step 2: Find Most Similar True News
- Concatenate: `title + keywords + description + temporal_evolution`
- Encode using Chinese-CLIP (`OFA-Sys/chinese-clip-vit-large-patch14`)
- Build vector bank from all true news (same concatenation + encoding)
- Calculate pairwise cosine similarity
- Extract top-k most similar true news videos

##### Step 3: Extract False Claims with Context
- Input: Fake video data + most similar true news
- Provide LLM with:
  - Current fake video information
  - All entities from similar true news
  - All correct claims for each entity
- Ask LLM to extract false claims for existing entities
- Allow new entity false claims (mark as "LLM-generated")
- Update `entity_knowledge_base.json` with false claims

##### Step 4: Summarize False Claims
- Merge entities with same normalization
- Generate comprehensive false claim summaries per entity
- Update final `entity_knowledge_base.json`

#### 4.3 [ ] Design Prompt Template
**Key Elements**:
```python
prompt = f"""
分析这个短视频的内容，提取知识图谱信息。

视频信息：
- 标题：{title}
- 关键词：{keywords}  
- 音频转录：{transcript}
- 标签：{label}
- 发布时间：{publish_time} # 格式化为YYYY-MM-DD

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

1. 实体（Entities）：
   列出视频中的主要实体（人物、地点、组织、事件、概念等）

2. 事实性声称（Factual Claims）：
   列出视频中所有的事实性声称，要具体、可验证
   格式：每个声称独立成行，简洁明了
   例如：
   - "武汉发现新型冠状病毒"
   - "疫苗有效率达到90%"
   - "某国确诊病例超过100万"

3. 视频描述（Description）：
   简要描述这个视频讲述了什么（50-100字）

4. 时序演变（Temporal Evolution）：
   描述内容如何在4个复合帧中演变

返回JSON格式：
{
  "entities": ["实体1", "实体2", ...],
  "claims": [
    "具体的事实性声称1",
    "具体的事实性声称2",
    ...
  ],
  "description": "视频描述",
  "temporal_evolution": "时序演变描述"
}
"""
```

#### 4.4 [ ] Implement Data Processing Pipeline
**Components**:
1. **Data Loading**:
   ```python
   def load_fakesv_data(filter_label='真'):
       # Load from data_complete_orig.jsonl
       # Filter by annotation
       # Load corresponding transcripts and OCR
   ```

2. **Image Processing**:
   ```python
   def prepare_composite_frames(video_id):
       # Load 4 quad images from data/FakeSV/quads_4/
       # Convert to base64 for GPT-4o-mini vision
   ```

3. **API Integration**:
   ```python
   def extract_knowledge_graph(video_data, images):
       # Call GPT-4o-mini with vision capability
       # Parse JSON response
       # Handle errors and retries
   ```

#### 4.5 [ ] Storage Format Design
**Output Structure**:
```
data/FakeSV/entity_claims/
├── video_extractions.jsonl   # Per-video extraction results
├── entity_knowledge_base.json # Aggregated entity descriptions and claims
└── claim_index.json          # Claim to video mapping
```

**Data Structure Definitions**:

1. **video_extractions.jsonl** (每行一个视频的提取结果):
```json
{
  "video_id": "3x4zj7hyemptkvm",
  "annotation": "真",
  "title": "视频标题",
  "entities": ["中国", "新冠疫苗", "武汉"],
  "claims": [
    "中国成功研发新冠疫苗",
    "疫苗有效率达到95%",
    "武汉是疫情首发地"
  ],
  "description": "该视频展示了中国新冠疫苗研发的过程和成果。",
  "temporal_evolution": "第1帧展示实验室，第2帧展示临床试验，第3帧展示数据分析，第4帧展示疫苗接种。"
}
```

2. **entity_knowledge_base.json** (实体知识库):
```json
{
  "中国": {
    "correct_description": "基于真实视频中的正确声称：中国在疫情期间实施了严格的防控措施，成功研发了多款新冠疫苗，疫苗接种率位居世界前列。",
    "false_description": "基于虚假视频中的错误声称：[待后续从假新闻视频中提取]",
    "correct_claims": [
      {
        "claim": "中国成功研发新冠疫苗",
        "video_ids": ["video1", "video3", "video7"],
        "source_label": "真"
      },
      {
        "claim": "中国疫情防控措施包括封城和全民核酸",
        "video_ids": ["video2", "video5"],
        "source_label": "真"
      }
    ],
    "false_claims": [
      {
        "claim": "[待从假新闻视频中提取]",
        "video_ids": [],
        "source_label": "假"
      }
    ],
    "video_count": {
      "真": 156,
      "假": 0,  // 待后续处理
      "辟谣": 0  // 待后续处理
    },
    "first_seen": "2020-01-15",
    "last_seen": "2023-12-20"
  },
  "新冠疫苗": {
    "correct_description": "基于真实视频：新冠疫苗经过三期临床试验，有效率在60-95%之间，可显著降低重症率和死亡率。",
    "false_description": "基于虚假视频：[待后续从假新闻视频中提取]",
    "correct_claims": [
      {
        "claim": "疫苗有效率达到95%",
        "video_ids": ["video1", "video4"],
        "source_label": "真"
      },
      {
        "claim": "疫苗可预防重症",
        "video_ids": ["video6", "video8", "video9"],
        "source_label": "真"
      }
    ],
    "false_claims": [],
    "video_count": {
      "真": 89,
      "假": 0,
      "辟谣": 0
    },
    "first_seen": "2020-03-10",
    "last_seen": "2023-11-15"
  }
}
```

3. **claim_index.json** (声称索引):
```json
{
  "correct_claims": [
    {
      "id": "C001",
      "claim": "中国成功研发新冠疫苗",
      "entities": ["中国", "新冠疫苗"],
      "video_ids": ["video1", "video3", "video7"],
      "frequency": 3,
      "source_label": "真"
    },
    {
      "id": "C002",
      "claim": "疫苗有效率达到95%",
      "entities": ["新冠疫苗"],
      "video_ids": ["video1", "video4"],
      "frequency": 2,
      "source_label": "真"
    }
  ],
  "false_claims": [
    // 待后续从假新闻视频中提取
  ],
  "statistics": {
    "total_correct_claims": 523,
    "total_false_claims": 0,  // 待后续处理
    "total_videos_processed": {
      "真": 1814,
      "假": 0,
      "辟谣": 0
    }
  }
}
```


#### 4.6 [ ] Quality Control & Validation
- **Entity Normalization**: Merge similar entities (e.g., "中国" and "中华人民共和国")
- **Claim Deduplication**: Merge similar claims across videos
- **Claim Validation**: Ensure claims are factual and verifiable
- **Coverage Metrics**: Track extraction success rate and completeness

#### 4.7 [ ] Description Generation from Claims
- **Correct Description**: Aggregate and summarize all correct claims for each entity
- **False Description**: [To be implemented when processing fake videos]
- **Contextual Enhancement**: Add temporal and source context to descriptions
- **Description Validation**: Ensure descriptions are coherent and informative

### Configuration Parameters
```yaml
entity_claim_extraction:
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4o-mini"
  temperature: 0.3
  max_tokens: 800
  batch_size: 10
  retry_attempts: 3
  filter_labels: ["真"]  # Start with real videos
  vision_detail: "high"
  rate_limit_delay: 1.0  # seconds between API calls
```

### Testing Plan

#### 4.8 [ ] Initial Testing (10 samples)
1. Select 10 diverse videos with annotation='真'
2. Run extraction pipeline
3. Manually validate results
4. Calculate metrics:
   - Entity extraction precision
   - Relation extraction accuracy
   - Description relevance
   - Temporal coherence

#### 4.9 [ ] Scale Testing (100 samples)
1. Process 100 videos after initial validation
2. Analyze common entity types and relations
3. Build entity/relation vocabulary
4. Identify edge cases and errors

#### 4.10 [ ] Full Dataset Processing
1. Process all 1,814 '真' labeled videos
2. Optional: Process '假' and '辟谣' videos
3. Generate statistics and insights
4. Create visualization of knowledge graph

### Expected Outputs

1. **Per Video**:
   - 5-10 entities on average
   - 3-8 factual claims
   - 1 general description
   - 1 temporal evolution description

2. **Dataset Level**:
   - Entity vocabulary (~500-1000 unique entities)
   - Unique claims (~2000-3000)
   - Entity knowledge base with contextual descriptions
   - Claim-to-video mapping for source attribution

### Technical Considerations

1. **API Costs**: 
   - ~$0.15 per 1M input tokens (GPT-4o-mini)
   - ~$0.60 per 1M output tokens
   - Estimated: ~$50 for full dataset

2. **Processing Time**:
   - ~2-3 seconds per video (API latency)
   - ~1.5 hours for full dataset (with rate limiting)

3. **Error Handling**:
   - Retry logic for API failures
   - Fallback to text-only if vision fails
   - Logging for debugging

### Success Criteria
- [ ] Successfully extract entities and claims for 95%+ videos
- [ ] Average entity count > 5 per video
- [ ] Average claim count > 3 per video
- [ ] Clear separation between correct and false claims
- [ ] Entity descriptions accurately summarize claims
- [ ] Database structure supports efficient retrieval

---
**Status**: Awaiting approval to proceed