import cv2
import numpy as np
import torch
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation

class CannyEdgeExtractor:
    """OpenCV 기반 Canny Edge 추출기"""
    def __init__(self, low_threshold=100, high_threshold=200):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def extract(self, pil_image: Image.Image) -> Image.Image:
        # PIL -> OpenCV (numpy array)
        cv_img = np.array(pil_image)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        
        # Canny edge 검출
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        
        # Edge 맵을 3채널 RGB 형태로 복제 (Stable Diffusion 입력용)
        edges_3ch = np.concatenate([edges[:, :, None]] * 3, axis=-1)
        return Image.fromarray(edges_3ch)


class DepthMapExtractor:
    """Hugging Face Transformers의 DPT 모델 기반 Depth Map 추출기"""
    def __init__(self, model_id="Intel/dpt-hybrid-midas", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.processor = DPTImageProcessor.from_pretrained(model_id)
        self.model = DPTForDepthEstimation.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def extract(self, pil_image: Image.Image) -> Image.Image:
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # 원래 이미지 크기로 보간(Interpolation)
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=pil_image.size[::-1], # (height, width)
            mode="bicubic",
            align_corners=False,
        )
        depth_output = prediction.squeeze().cpu().numpy()
        
        # 0 ~ 255 정규화
        formatted = (depth_output * 255 / np.max(depth_output)).astype(np.uint8)
        depth_3ch = np.concatenate([formatted[:, :, None]] * 3, axis=-1)
        return Image.fromarray(depth_3ch)