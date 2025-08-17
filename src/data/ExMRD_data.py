import torch
from torch.utils.data import Dataset, DataLoader
from .baseline_data import FakeSVDataset, FakeTTDataset, FVCDataset
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

class FakeSVDataset_ExMRD(FakeSVDataset):
    def __init__(self, fold: int, split: str, lm: str='gpt-4o', ablation_no_cot: bool=False, 
                 description: bool=False, temp_evolution: bool=False,
                 use_text: bool=True, use_image: bool=True, use_audio: bool=True, **kwargs):
        super(FakeSVDataset_ExMRD, self).__init__(**kwargs)
        self.data = self._get_data(fold, split)
        self.ablation_no_cot = ablation_no_cot
        self.description = description
        self.temp_evolution = temp_evolution
        self.use_text = use_text
        self.use_image = use_image
        self.use_audio = use_audio
        
        # Load features based on modality controls
        self.fea_frames = None
        self.audio_features = None
        self.entity_data = None
        
        if self.use_image:
            self.fea_frames = torch.load('data/FakeSV/fea/vit_tensor.pt', weights_only=True)
        if self.use_audio:
            self.audio_features = torch.load('data/FakeSV/fea/audio_features_frames.pt', weights_only=True)
        
        # Load entity claims data for text features if needed
        if self.use_text:
            self.video_extractions = pd.read_json('data/FakeSV/entity_claims/video_extractions.jsonl', lines=True)
            self.fake_video_descriptions = pd.read_json('data/FakeSV/entity_claims/fake_video_descriptions.jsonl', lines=True)
            # Combine real and fake video descriptions
            self.entity_data = pd.concat([self.video_extractions, self.fake_video_descriptions], ignore_index=True)
        
        # Get available video IDs based on enabled modalities
        available_ids = set(self.data['video_id'])  # Start with all video IDs
        
        if self.use_image and self.fea_frames is not None:
            visual_ids = set(self.fea_frames.keys())
            available_ids &= visual_ids
            
        if self.use_audio and self.audio_features is not None:
            audio_ids = set(self.audio_features.keys())
            available_ids &= audio_ids
            
        if self.use_text and self.entity_data is not None:
            entity_ids = set(self.entity_data['video_id'])
            available_ids &= entity_ids
        
        # Filter dataset to only include videos with required features
        original_length = len(self.data)
        self.data = self.data[self.data['video_id'].isin(available_ids)].reset_index(drop=True)
        
        print(f"Filtered dataset: {original_length} -> {len(self.data)} samples (removed {original_length - len(self.data)} incomplete samples)")
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']
        label = torch.tensor(item['label'], dtype=torch.long)
        
        # Initialize features as None
        visual_features = None
        audio_features = None
        concatenated_text = None
        
        # Get visual features if enabled
        if self.use_image:
            if self.fea_frames is None or vid not in self.fea_frames:
                print(f"Warning: Video features not found for {vid}, skipping...")
                return self.__getitem__((idx + 1) % len(self.data))
            visual_features = torch.Tensor(self.fea_frames[vid])  # (16, 1024)
            
        # Get audio features if enabled
        if self.use_audio:
            if self.audio_features is None or vid not in self.audio_features:
                print(f"Warning: Audio features not found for {vid}, skipping...")
                return self.__getitem__((idx + 1) % len(self.data))
            audio_features = torch.Tensor(self.audio_features[vid])  # (seq_len, 768)
        
        # Get text features if enabled
        if self.use_text:
            if self.entity_data is None:
                print(f"Warning: Entity data not loaded, skipping...")
                return self.__getitem__((idx + 1) % len(self.data))
                
            entity_row = self.entity_data[self.entity_data['video_id'] == vid]
            if len(entity_row) == 0:
                print(f"Warning: Entity data not found for {vid}, skipping...")
                return self.__getitem__((idx + 1) % len(self.data))
            
            entity_row = entity_row.iloc[0]
        
        # Process text features if enabled
        if self.use_text:
            # Concatenate text features based on description and temp_evolution settings
            text_parts = []
            
            # Always include title and keywords
            if 'title' in entity_row and pd.notna(entity_row['title']):
                text_parts.append(str(entity_row['title']))
            if 'keywords' in entity_row and pd.notna(entity_row['keywords']):
                text_parts.append(str(entity_row['keywords']))
                
            # Include description if enabled
            if self.description and 'description' in entity_row and pd.notna(entity_row['description']):
                text_parts.append(str(entity_row['description']))
                
            # Include temporal_evolution if enabled  
            if self.temp_evolution and 'temporal_evolution' in entity_row and pd.notna(entity_row['temporal_evolution']):
                text_parts.append(str(entity_row['temporal_evolution']))
            
            concatenated_text = ' '.join(text_parts)
        
        return {
            'vid': vid,
            'label': label,
            'concatenated_text': concatenated_text,
            'visual_features': visual_features,
            'audio_features': audio_features,
            'use_text': self.use_text,
            'use_image': self.use_image,
            'use_audio': self.use_audio,
        }
    
class FakeSVCollator_ExMRD:
    def __init__(self, tokenizer_name, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def __call__(self, batch):
        vids = [item['vid'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        
        # Get modality flags from first item (should be same for all)
        use_text = batch[0]['use_text']
        use_image = batch[0]['use_image'] 
        use_audio = batch[0]['use_audio']
        
        # Initialize return dict
        result = {
            'vids': vids,
            'labels': labels,
            'use_text': use_text,
            'use_image': use_image,
            'use_audio': use_audio,
        }
        
        # Process text features if enabled
        if use_text:
            concatenated_texts = [item['concatenated_text'] for item in batch]
            max_len = 512  # Increased to accommodate concatenated text
            entity_text_input = self.tokenizer(
                concatenated_texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=max_len
            )
            result['entity_text_input'] = entity_text_input
            
        # Process visual features if enabled
        if use_image:
            visual_features = torch.stack([item['visual_features'] for item in batch])
            result['visual_features'] = visual_features
            
        # Process audio features if enabled
        if use_audio:
            audio_features = torch.stack([item['audio_features'] for item in batch])
            result['audio_features'] = audio_features
        
        return result

class FakeTTDataset_ExMRD(FakeTTDataset):
    def __init__(self, fold: int, split: str, lm: str='gpt-4o', **kwargs):
        super(FakeTTDataset_ExMRD, self).__init__()
        self.data = self._get_data(fold, split)
        
        self.ocr = pd.read_json('data/FakeTT/ocr.jsonl', lines=True, dtype={'vid': 'str'})
        self.lm_ocr = pd.read_json(f'data/FakeTT/CoT/{lm}/lm_text_refine.jsonl', lines=True, dtype={'vid': 'str'})
        self.caption = pd.read_json(f'data/FakeTT/CoT/{lm}/lm_visual_refine.jsonl', lines=True, dtype={'vid': 'str'})
     
        self.lm_comsense = pd.read_json(f'data/FakeTT/CoT/{lm}/lm_retrieve.jsonl', lines=True, dtype={'vid': 'str'})
        self.lm_causal = pd.read_json(f'data/FakeTT/CoT/{lm}/lm_reason.jsonl', lines=True, dtype={'vid': 'str'})
    
        self.fea_frames = torch.load('data/FakeTT/fea/vit_tensor.pt', weights_only=True)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']
        # response
        text_caption = self.caption[self.caption['vid']==vid]['text'].iloc[0]
        text_lm_ocr = self.lm_ocr[self.lm_ocr['vid']==vid]['text'].iloc[0]
        text_lm_ocr = f'{item['description']}\n{text_lm_ocr}'
        text_lm_comsense = self.lm_comsense[self.lm_comsense['vid']==vid]['text'].iloc[0]
        text_lm_causal = self.lm_causal[self.lm_causal['vid']==vid]['text'].iloc[0]
        
        label = torch.tensor(item['label'], dtype=torch.long)

        fea_frames = self.fea_frames[vid]
        fea_frames = torch.Tensor(fea_frames)

        
        return {
            'vid': vid,
            'label': label,
            'text_lm_ocr': text_lm_ocr,
            'text_caption': text_caption,
            'text_comsense': text_lm_comsense,
            'text_causal': text_lm_causal,
            'fea_frames': fea_frames,
        }
    
class FakeTTCollator_ExMRD:
    def __init__(self, tokenizer_name, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def __call__(self, batch):
        vids = [item['vid'] for item in batch]
        texts_lm_ocr = [item['text_lm_ocr'] for item in batch]
        texts_caption = [item['text_caption'] for item in batch]
        texts_comsense = [item['text_comsense'] for item in batch]
        texts_causal = [item['text_causal'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        fea_frames = torch.stack([item['fea_frames'] for item in batch])

        max_len = 77
        # texts_input = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        lm_ocr_input = self.tokenizer(list(texts_lm_ocr), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        caption_input = self.tokenizer(list(texts_caption), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        comsense_input = self.tokenizer(list(texts_comsense), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        causal_input = self.tokenizer(list(texts_causal), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        return {
            'vids': vids,
            'fea_frames': fea_frames,
            'labels': labels,
            'lm_ocr_input': lm_ocr_input,
            'caption_input': caption_input,
            'comsense_input': comsense_input,
            'causal_input': causal_input,
        }


class FVCDataset_ExMRD(FVCDataset):
    def __init__(self, fold: int, split: str, lm: str='gpt-4o', **kwargs):
        super(FVCDataset_ExMRD, self).__init__()
        self.data = self._get_data(fold, split)
        
        self.lm_ocr = pd.read_json(f'data/FVC/CoT/{lm}/lm_text_refine.jsonl', lines=True, dtype={'vid': 'str'})
        self.caption = pd.read_json(f'data/FVC/CoT/{lm}/lm_visual_refine.jsonl', lines=True, dtype={'vid': 'str'})
         
        self.lm_comsense = pd.read_json(f'data/FVC/CoT/{lm}/lm_retrieve.jsonl', lines=True, dtype={'vid': 'str'})
        self.lm_causal = pd.read_json(f'data/FVC/CoT/{lm}/lm_reason.jsonl', lines=True, dtype={'vid': 'str'})
    
        self.fea_frames = torch.load('data/FVC/fea/vit_tensor.pt', weights_only=True)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['vid']
        # response
        text_caption = self.caption[self.caption['vid']==vid]['text'].iloc[0]
        text_lm_ocr = self.lm_ocr[self.lm_ocr['vid']==vid]['text'].iloc[0]
        text_lm_ocr = f'{item['title']}\n{item['description']}\n{text_lm_ocr}'
        
        text_lm_comsense = self.lm_comsense[self.lm_comsense['vid']==vid]['text'].iloc[0]
        text_lm_causal = self.lm_causal[self.lm_causal['vid']==vid]['text'].iloc[0]
        
        label = torch.tensor(item['label'], dtype=torch.long)

        fea_frames = self.fea_frames[vid]
        fea_frames = torch.Tensor(fea_frames)
        
        return {
            'vid': vid,
            'label': label,
            'text_lm_ocr': text_lm_ocr,
            'text_caption': text_caption,
            'text_comsense': text_lm_comsense,
            'text_causal': text_lm_causal,
            'fea_frames': fea_frames,
        }
    
class FVCCollator_ExMRD:
    def __init__(self, tokenizer_name, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def __call__(self, batch):
        vids = [item['vid'] for item in batch]
        texts_lm_ocr = [item['text_lm_ocr'] for item in batch]
        texts_caption = [item['text_caption'] for item in batch]
        texts_comsense = [item['text_comsense'] for item in batch]
        texts_causal = [item['text_causal'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        fea_frames = torch.stack([item['fea_frames'] for item in batch])

        max_len = 77
        lm_ocr_input = self.tokenizer(list(texts_lm_ocr), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        caption_input = self.tokenizer(list(texts_caption), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        comsense_input = self.tokenizer(list(texts_comsense), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        causal_input = self.tokenizer(list(texts_causal), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        return {
            'vids': vids,
            'fea_frames': fea_frames,
            'labels': labels,
            'lm_ocr_input': lm_ocr_input,
            'caption_input': caption_input,
            'comsense_input': comsense_input,
            'causal_input': causal_input,
        }