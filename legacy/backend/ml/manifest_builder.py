import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from loguru import logger

class ManifestBuilder:
    """
    Builds stratified CSV manifests for model training.
    Ensures balanced distribution of real and fake samples.
    """
    
    def __init__(self, data_root: Path):
        self.data_root = data_root
        
    def build_from_directory(self, 
                             real_dir: str = "real", 
                             fake_dir: str = "fake",
                             output_csv: str = "manifest.csv") -> pd.DataFrame:
        """
        Scans directories and builds a manifest of all found images.
        """
        data = []
        
        # Process REAL
        real_path = self.data_root / real_dir
        if real_path.exists():
            for img in real_path.glob("*.jpg"):
                data.append({
                    "path": str(img.relative_to(self.data_root)),
                    "label": 0,  # 0 for Real
                    "class": "real"
                })
        
        # Process FAKE
        fake_path = self.data_root / fake_dir
        if fake_path.exists():
            for img in fake_path.glob("*.jpg"):
                data.append({
                    "path": str(img.relative_to(self.data_root)),
                    "label": 1,  # 1 for Fake
                    "class": "fake"
                })
                
        df = pd.DataFrame(data)
        out_path = self.data_root / output_csv
        df.to_csv(out_path, index=False)
        logger.info(f"Manifest built with {len(df)} samples: {out_path}")
        return df

    def split_manifest(self, 
                       manifest_csv: str = "manifest.csv",
                       train_ratio: float = 0.8,
                       val_ratio: float = 0.1,
                       test_ratio: float = 0.1):
        """
        Splits the manifest into train, val, and test stratified by label.
        """
        df = pd.read_csv(self.data_root / manifest_csv)
        
        # First split: Train vs Temp (Val + Test)
        train_df, temp_df = train_test_split(
            df, 
            train_size=train_ratio, 
            stratify=df['label'],
            random_state=42
        )
        
        # Second split: Val vs Test
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=relative_val_ratio,
            stratify=temp_df['label'],
            random_state=42
        )
        
        # Save splits
        train_df.to_csv(self.data_root / "train.csv", index=False)
        val_df.to_csv(self.data_root / "val.csv", index=False)
        test_df.to_csv(self.data_root / "test.csv", index=False)
        
        logger.info(f"Split complete: Train({len(train_df)}), Val({len(val_df)}), Test({len(test_df)})")
