import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

class CannyEdgeExtractor:
    """이미지에서 윤곽선(외곽선) 가이드를 추출하는 클래스"""
    def __init__(self, low_threshold=100, high_threshold=200):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def extract(self, pil_image: Image.Image) -> Image.Image:
        open_cv_image = np.array(pil_image)
        gray_img = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_img, self.low_threshold, self.high_threshold)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb)


class DepthMapExtractor:
    """이미지에서 깊이(원근감) 정보를 분석하여 왜곡을 방지하는 클래스"""
    def __init__(self, model_name="LiheYoung/depth-anything-small-hf"):
        self.depth_estimator = pipeline("depth-estimation", model=model_name)

    def extract(self, pil_image: Image.Image) -> Image.Image:
        result = self.depth_estimator(pil_image)
        depth_image = result["depth"].convert("RGB")
        return depth_image