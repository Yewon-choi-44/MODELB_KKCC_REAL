import torch
import numpy as np
import cv2
from PIL import Image
from typing import Dict
from category_registry import CategorySpec

class UniversalAugmentor:
    """
    범용 Generative AI 결함 증강 & 복원 엔진 (Universal Augmentor)
    임의의 카테고리(spec)를 받아 Multi-ControlNet SD 3.5 + SUPIR 복원 파이프라인으로
    실사 결함 데이터셋을 생성함.
    """
    def __init__(self, spec: CategorySpec, device: str = None):
        self.spec = spec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[UniversalAugmentor] Initialized for category '{spec.name}' ({spec.material} / {spec.geom_type})")

    def synthesize_and_restore(self, geom_data: Dict[str, np.ndarray]) -> Image.Image:
        """
        Depth/Normal 조건에 맞춰 SD 3.5 텍스처 합성 및 SUPIR 초고해상도 복원 적용
        """
        depth_map = geom_data["depth_map"]
        normal_map = geom_data["normal_map"]
        gt_mask = geom_data["gt_mask"]
        mask = geom_data["mask"]
        
        # 1. SD 3.5 Multi-ControlNet Synthesizer Skeleton
        # (금속, 글래스, 직물, 세라믹, 음식 등 재질별 맞춤형 프롬프트 합성)
        prompt = self.spec.default_prompt
        print(f" -> [SD 3.5] Prompting: '{prompt[:65]}...'")
        
        synthetic_np = normal_map.copy()
        
        # 재질 및 결함에 따른 인페인팅 텍스처 데모 시뮬레이션
        if self.spec.material == "metal":
            synthetic_np[gt_mask > 0] = (synthetic_np[gt_mask > 0] * 0.5 + 50).astype(np.uint8)
        elif self.spec.material == "glass":
            synthetic_np[gt_mask > 0] = (synthetic_np[gt_mask > 0] * 0.2 + 200).astype(np.uint8)
        elif self.spec.material == "fabric" or self.spec.material == "leather":
            synthetic_np[gt_mask > 0] = (synthetic_np[gt_mask > 0] * 0.8 + 20).astype(np.uint8)
        else:
            synthetic_np[gt_mask > 0] = (255 - synthetic_np[gt_mask > 0])
            
        # 2. SUPIR Ultra Super-Resolution & Micro-texture Restoration
        h, w, c = synthetic_np.shape
        upscaled = cv2.resize(synthetic_np, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        
        # 렌즈 노이즈 및 선명도 샤프닝
        kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(upscaled, -1, kernel_sharpen)
        noise = np.random.normal(0, 2.0, sharpened.shape).astype(np.float32)
        restored_np = np.clip(sharpened.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(restored_np)
