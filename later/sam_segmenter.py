# preprocessing/sam_extractor.py
import os
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor

"""SAM에는 두가지 방식이 있는데 transformers 라이브러리에 있는 SAM을 사용하거나,
 segment_anything 라이브러리에서 제공하는 사용자 편의용 래퍼(Wrapper) 클래스를 사용하는 것이다.
 아래 코드는 segment_anything 라이브러리에서 제공하는 클래스를 사용하는 코드이다."""

"""SAM(Segment Anything Model)을 활용하여 이미지 내 특정 객체의 마스크를 추출하는 모듈"""
class SAMMaskExtractor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. SAM 모델 및 Predictor 세팅
        self.model = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(self.device)
        self.predictor = SamPredictor(self.model)
        print(f"   => [SAMMaskExtractor] SAM 로드 완료 (Device: {self.device})")
        # 2. 기준이 되는 프롬프트 설정 (예: 이미지 중앙 근처의 점 좌표와 라벨)
        # 사진들의 해상도와 대상 위치가 거의 같으므로 동일한 좌표를 재사용합니다.
        self.input_point = np.array([[512, 512]])
        self.input_label = np.array([1]) # 1: 포인트를 포함한 마스크 생성 // 2: 포인트를 제외한 마스크 생성

        self.train_target_dir = None

    def extract_mask(self, train_target_dir, output_dir):
        """
        특정 픽셀 좌표(Point)를 기반으로 객체 마스크를 추출합니다.
        point_coords: [[x, y]] 형태의 리스트 (예: 제품의 중심점)
        """
        self.train_target_dir = train_target_dir
        self.output_dir = output_dir
        for img_name in os.listdir(self.train_target_dir):
            if not img_name.endswith(('.png')): continue

            img_path = os.path.join(self.train_target_dir, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            self.predictor.set_image(image)
            masks, scores, logits = self.predictor.predict(
                point_coords=self.input_point,
                point_labels=self.input_label,
                multimask_output=False  # 가장 신뢰도 높은 단일 마스크만 반환
            )
            mask_img = (masks[0] * 255).astype(np.uint8)
            mask_path = os.path.join(self.output_dir, img_name)
            cv2.imwrite(mask_path, mask_img)

# if __name__ == "__main__":
#     sample = SAMMaskExtractor()
#     sample.extract_mask(train_target_dir="/home/ai-engr/KKCC/modelA_first/0628/can_padded",
#                         output_dir="/home/ai-engr/KKCC/modelA_first/0628/can_mask")