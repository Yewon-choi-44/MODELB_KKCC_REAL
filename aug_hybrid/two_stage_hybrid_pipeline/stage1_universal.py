import os
import cv2
import numpy as np
from PIL import Image

class Stage1UniversalAugmentor:
    """
    Stage 1: Universal Baseline Data Augmentor
    전 품목(또는 지정 품목)에 대한 표준 3D Depth/Normal 기하 맵 생성 및 베이스라인 합성.
    """
    def __init__(self, target_category: str, output_base_dir: str):
        self.category = target_category
        self.output_dir = os.path.join(output_base_dir, "stage1_universal", target_category)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "masks"), exist_ok=True)

    def run_baseline_generation(self, count: int = 2):
        print(f"[Stage 1 - Universal] Generating {count} baseline samples for '{self.category}'...")
        
        w, h = 1024, 1024
        for i in range(count):
            sample_id = f"{self.category}_s1_{i+1:04d}"
            
            # Universal Planar/Cylindrical Base Map
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (w//2, h//2), (int(w*0.35), int(h*0.4)), 0, 0, 360, 255, -1)
            
            gt_mask = np.zeros((h, w), dtype=np.uint8)
            cx, cy = w//2 + np.random.randint(-100, 100), h//2 + np.random.randint(-100, 100)
            cv2.circle(gt_mask, (cx, cy), np.random.randint(40, 80), 255, -1)
            gt_mask = cv2.bitwise_and(gt_mask, mask)
            
            # Baseline Synth Image
            synth_np = np.stack([mask, mask, mask], axis=-1)
            synth_np[gt_mask > 0] = [200, 50, 50]  # Baseline Defect Color
            
            img_path = os.path.join(self.output_dir, "images", f"{sample_id}.png")
            mask_path = os.path.join(self.output_dir, "masks", f"{sample_id}_mask.png")
            
            Image.fromarray(synth_np).save(img_path)
            cv2.imwrite(mask_path, gt_mask)
            print(f"   -> [Stage 1] Saved {sample_id}.png")
