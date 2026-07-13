import torch
import os
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from peft import LoraConfig, get_peft_model, PeftModel


"""

MVTec AD 데이터셋 학습 및 불량 이미지 생성을 전용으로 처리하는 DoRA 클래스 모듈

"""

class DoraWeightEvaluator:

    def __init__(self, base_model = "stabilityai/stable-diffusion-3.5-large"):
        self.base_model = base_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.pipeline = None


    def build_dora_model(self, r, alpha):
        transformer = SD3Transformer2DModel.from_pretrained(
            self.base_model,
            subfolder = "transformer",
            torch_dtype=torch.bfloat16
        )

        dora_config = LoraConfig(
            r = any,
            lora_alpha = any,
            target_modules = ["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout = 0.05,
            bias = "none",
            use_dora = True
        )

        self.model = get_peft_model(transformer, dora_config)
        self.model.to(self.device)
        return self.model
    

    def save_dora_weights(self, save_directory=any):
        if self.model is None:
            raise ValueError("저장할 모델 없음.")
        
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)


    def load_dora_model(self, dora_path):
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            self.base_model,
            torch_dtype=torch.bfloat16
        )
        
        self.pipeline.transformer = PeftModel.from_pretrained(
            self.pipeline.transformer,
            dora_path
        )

        print("DoRA 가중치 융합 완료.")