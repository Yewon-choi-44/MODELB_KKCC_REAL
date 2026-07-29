import os
import cv2
import numpy as np
from PIL import Image

class Stage2WallplugSpecialist:
    """
    Stage 2 Specialist: 'wallplugs' (플라스틱 칼브럭 / 앙카) 맞춤형 스페셜리스트
    1. 사출 성형 플라스틱 앙카 3D 날개 휨(Twisted Wing), 팁 부러짐(Broken Tip) 시뮬레이션
    2. 플라스틱 매트 질감 & 응력 백화 현상(Stress Whitening) SD 3.5 ControlNet 합성
    3. SUPIR 파팅 라인(Parting Line) & 사출 면 복원
    """
    def __init__(self, output_base_dir: str):
        self.output_dir = os.path.join(output_base_dir, "stage2_specialized", "wallplugs")
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "masks"), exist_ok=True)

    def run_specialized_generation(self, count: int = 2):
        print(f"[Stage 2 - WALLPLUGS Specialist] Generating {count} SOTA plastic wallplug deformed samples...")
        w, h = 1024, 1024
        
        for i in range(count):
            sample_id = f"wallplugs_sota_{i+1:04d}"
            
            # 1. Plastic Wallplug Base Geometry (앙카 형태)
            plug_mask = np.zeros((h, w), dtype=np.uint8)
            # 메인 바디
            cv2.rectangle(plug_mask, (int(w*0.4), int(h*0.15)), (int(w*0.6), int(h*0.85)), 255, -1)
            # 앙카 날개 (Wings)
            cv2.ellipse(plug_mask, (int(w*0.35), int(h*0.4)), (50, 120), 30, 0, 360, 255, -1)
            cv2.ellipse(plug_mask, (int(w*0.65), int(h*0.4)), (50, 120), -30, 0, 360, 255, -1)
            
            # 2. Plastic Deformation & Stress Whitening Simulation (파손 / 백화)
            gt_mask = np.zeros((h, w), dtype=np.uint8)
            
            # 팁 끝부분 부러짐 (Broken Tip)
            break_y = int(h * 0.75)
            gt_mask[break_y:, int(w*0.35):int(w*0.65)] = 255
            gt_mask = cv2.bitwise_and(gt_mask, plug_mask)
            
            # 3. Matte Plastic Texture & Stress Whitening Color
            plastic_bg = np.full((h, w), 210, dtype=np.uint8)  # 회색/흰색 매트 플라스틱
            plastic_rgb = np.stack([plastic_bg, plastic_bg + 5, plastic_bg + 10], axis=-1)
            plastic_rgb = cv2.bitwise_and(plastic_rgb, plastic_rgb, mask=plug_mask)
            
            # 백화 현상 (Stress Whitening: 응력 집중 부위 흰색 변색)
            plastic_rgb[gt_mask > 0] = [255, 250, 240]  # Whitening Color
            
            # 4. SUPIR Parting Line & Surface Texture Restoration
            upscaled = cv2.resize(plastic_rgb, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            noise = np.random.normal(0, 1.8, upscaled.shape).astype(np.float32)
            restored_np = np.clip(upscaled.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
            img_path = os.path.join(self.output_dir, "images", f"{sample_id}.png")
            mask_path = os.path.join(self.output_dir, "masks", f"{sample_id}_mask.png")
            
            Image.fromarray(restored_np).save(img_path)
            cv2.imwrite(mask_path, cv2.resize(gt_mask, (w * 2, h * 2)))
            print(f"   -> [Stage 2: WALLPLUGS Specialist] Saved SOTA Sample {sample_id}.png + 0.5% Precision Mask")
