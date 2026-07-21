import torch
from PIL import Image
from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel

class ControlNetAugmentor:
    """SD3.5 + Multi-ControlNet (Canny + Depth) 기반 조건부 이미지 증강 생성기"""
    def __init__(self, base_model_id="stabilityai/stable-diffusion-3.5-large", 
                 canny_controlnet_id="InstantX/SD3.5-Large-ControlNet-Canny",
                 depth_controlnet_id="InstantX/SD3.5-Large-ControlNet-Depth"): # 가칭
                 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Canny & Depth ControlNet 모델 로드
        self.canny_control = SD3ControlNetModel.from_pretrained(canny_controlnet_id, torch_dtype=torch.float16)
        self.depth_control = SD3ControlNetModel.from_pretrained(depth_controlnet_id, torch_dtype=torch.float16)
        
        # 2. 통합 멀티 컨트롤넷 파이프라인 구성
        self.pipeline = StableDiffusion3ControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=[self.canny_control, self.depth_control],
            torch_dtype=torch.float16
        ).to(self.device)
        
        # 3. CPU Offload 및 메모리 최적화 기법 적용
        self.pipeline.enable_model_cpu_offload()

    def update_weights(self, tuned_transformer_dir: str):
        """파인튜닝된 SD3.5 Transformer 가중치로 덮어쓰기"""
        print(f"[Augmentor] 파인튜닝 가중치 적용 중: {tuned_transformer_dir}")
        self.pipeline.transformer.load_pretrained(tuned_transformer_dir)

    def generate(self, prompt: str, negative_prompt: str, canny_img: Image.Image, depth_img: Image.Image, 
                 control_scales=[0.6, 0.4], num_inference_steps=30, guidance_scale=7.5) -> Image.Image:
        """가이드를 바탕으로 새로운 정방형 이미지 1장 생성"""
        
        images = [canny_img, depth_img]
        
        generator = torch.Generator(device=self.device).manual_seed(42)
        
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_image=images,
            controlnet_conditioning_scale=control_scales, # 각 가이드 가중치 조율
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        
        return output.images[0]