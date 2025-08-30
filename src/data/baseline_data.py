from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image


class FakeSVDataset(Dataset):
    def __init__(self, include_piyao=False, filter_k=None, **kwargs):
        super(FakeSVDataset, self).__init__()
        self.include_piyao = include_piyao
        self.filter_k = filter_k
    
    def _get_complete_data(self):
        # Always use original dataset which contains '辟谣' labels
        data_complete = pd.read_json('./data/FakeSV/data_complete_orig.jsonl', orient='records', dtype=False, lines=True)
        
        if self.include_piyao:
            # Map '辟谣' (rumor debunking) to real/true (0), since it represents fact-checking content
            replace_values = {'辟谣': 0, '假': 1, '真':0}
            data_complete['label'] = data_complete['annotation'].replace(replace_values)
            # Use all data including '辟谣' entries (mapped to 0)
        else:
            # Filter out '辟谣' entries - map to 2 then remove
            replace_values = {'辟谣': 2, '假': 1, '真':0}
            data_complete['label'] = data_complete['annotation'].replace(replace_values)
            data_complete = data_complete[data_complete['label']!=2]
        
        data_complete['event'], _ = pd.factorize(data_complete['keywords'])
        return data_complete
    
    def _get_data(self, fold, split):
        if fold in [1, 2, 3, 4, 5]:
            data = self._get_fold_data(fold, split)
        elif fold in ['temporal']:
            data = self._get_temporal_data(split)
        else:
            raise NotImplementedError(f"Invalid fold: {fold}")
        return data
    
    def _get_fold_data(self, fold, split):
        if split == 'train':
            vid_path = f'data/FakeSV/vids/vid_fold_no_{fold}.txt'
        elif split == 'test':
            vid_path = f'data/FakeSV/vids/vid_fold_{fold}.txt'
        else:
            raise ValueError(f"Invalid split: {split}")
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['video_id'].isin(vids)]
        return data

    def _get_temporal_data(self, split: str):
        # Use filtered training file if filter_k is specified and split is 'train'
        if split == 'train' and self.filter_k is not None:
            vid_path = f'data/FakeSV/vids/vid_time3_train_k{self.filter_k}.txt'
        else:
            vid_path = f'data/FakeSV/vids/vid_time3_{split}.txt'
            
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['video_id'].isin(vids)]
        return data


class FakeTTDataset(Dataset):
    def __init__(self):
        super(FakeTTDataset, self).__init__()
    
    def _get_complete_data(self):
        data = pd.read_json('data/FakeTT/data.jsonl', orient='records', lines=True, dtype={'video_id': 'str'})
        replace_values = {'fake': 1, 'real': 0}
        data['label'] = data['annotation'].replace(replace_values)
        data['event'], _ = pd.factorize(data['event'])
        # set type of video_id to str
        return data
    
    def _get_data(self, fold, split):
        if fold in ['temporal']:
            data = self._get_temporal_data(split)
        elif fold in [1, 2, 3, 4, 5]:
            data = self._get_fold_data(fold, split)
        else:
            raise NotImplementedError(f"Invalid fold: {fold}")
        return data
    
    def _get_fold_data(self, fold, split):
        if split == 'train':
            vid_path = f'data/FakeTT/vids/vid_fold_no_{fold}.txt'
        elif split == 'test':
            vid_path = f'data/FakeTT/vids/vid_fold_{fold}.txt'
        else:
            raise ValueError(f"Invalid split: {split}")
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['video_id'].isin(vids)]
        return data

    def _get_temporal_data(self, split: str):
        vid_path = f'data/FakeTT/vids/vid_time3_{split}.txt'
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['video_id'].isin(vids)]
        return data
    
class FVCDataset(Dataset):
    def __init__(self):
        super(FVCDataset, self).__init__()
    
    def _get_complete_data(self):
        data = pd.read_json('data/FVC/data.jsonl', orient='records', lines=True, dtype={'vid': 'str'})
        data = data[data['label'].isin(['fake', 'real'])]
        replace_values = {'fake': 1, 'real': 0}
        data['label'] = data['label'].replace(replace_values)
        data['event'], _ = pd.factorize(data['event_id'])
        data['video_id'] = data['vid']
        return data
    
    def _get_data(self, fold, split):
        if fold in ['temporal']:
            data = self._get_temporal_data(split)
        elif fold in [1, 2, 3, 4, 5]:
            data = self._get_fold_data(fold, split)
        else:
            raise NotImplementedError(f"Invalid fold: {fold}")
        return data
    
    def _get_fold_data(self, fold, split):
        if split == 'train':
            vid_path = f'data/FVC/vids/vid_fold_no_{fold}.txt'
        elif split == 'test':
            vid_path = f'data/FVC/vids/vid_fold_{fold}.txt'
        else:
            raise ValueError(f"Invalid split: {split}")
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['vid'].isin(vids)]
        return data

    def _get_temporal_data(self, split: str):
        vid_path = f'data/FVC/vids/vid_time3_{split}.txt'
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['vid'].isin(vids)]
        return data
