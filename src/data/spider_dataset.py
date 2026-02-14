import os
import json
from torch.utils.data import Dataset

class SpiderDataset(Dataset):
    """Spider dataset for Text-to-SQL."""
    
    def __init__(self, data_path: str, db_root: str):
        """
        Initialize Spider dataset.
        
        Args:
            data_path: Path to spider data JSON file
            db_root: Root directory containing database folders
        """
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.db_root = db_root
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        return {
            'question': item['question'],
            'sql': item['query'],
            'db_id': item['db_id'],
            'db_path': os.path.join(self.db_root, item['db_id'], f"{item['db_id']}.sqlite"),
            'schema': item.get('schema', None)  # Optional
        }


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    return {
        'question': [item['question'] for item in batch],
        'sql': [item['sql'] for item in batch],
        'db_path': [item['db_path'] for item in batch],
        'schema': [item.get('schema') for item in batch]
    }
