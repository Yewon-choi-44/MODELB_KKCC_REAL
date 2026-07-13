import os
import torch
from diffusers import StableDiffusion3ControlNetPipeline, ControlNetModel
from PIL import Image

class ControlNetImageGenerator:
    """가이드라인(Canny, Depth)과 DoRA를 융합하여 증강 이미지를 만드는 클래스"""
    def __init__(self, base_model="stabilityai/stable-diffusion-3.5-large"):
        self.base_model = base_model

    def generate(self, prompt: str, canny_img: Image.Image, depth_img: Image.Image, dora_path: str) -> Image.Image:
        print("🚀 [ControlNet 생성] SD 3.5 모델 및 컨트롤넷 로드 중...")
        
        controlnets = [
            ControlNetModel.from_pretrained("stabilityai/stable-diffusion-3.5-large-controlnet-canny", torch_dtype=torch.float16),
            ControlNetModel.from_pretrained("stabilityai/stable-diffusion-3.5-large-controlnet-depth", torch_dtype=torch.float16)
        ]
        
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            self.base_model, controlnet=controlnets, torch_dtype=torch.float16
        )
        
        if os.path.exists(dora_path):
            pipe.load_lora_weights(dora_path, adapter_name="dora_defect")
            print("💡 [DoRA 결합] 불량 특징 기억 장치 주입 완료.")
            
        pipe.to("cuda")

        augmented_output = pipe(
            prompt=prompt,
            control_image=[canny_img, depth_img],
            num_inference_steps=28,
            controlnet_conditioning_scale=[0.7, 0.6]
        ).images[0]

        return augmented_output