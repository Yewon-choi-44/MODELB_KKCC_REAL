import os
import json
import subprocess

class LoraWeightExtractor:
    def __init__(self):
        self.script_path = None
        self.category = None
        self.input_dir = None
        self.output_dir = None
    def extract_lora_weights(self, input_dir, output_dir):
        self.script_path = "/home/ai-engr/KKCC/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py"
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        cmd = [
            "accelerate", "launch", "/home/ai-engr/KKCC/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py",
            f"--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
            f"--train_data_dir={self.input_dir}",
            f"--output_dir={self.output_dir}",
            f"--caption_column=text",
            f"--resolution=1024",  # 정방형 규격 통일 반영
            f"--train_batch_size=4",
            f"--num_train_epochs=10",
            f"--checkpointing_steps=5000",
            f"--learning_rate=1e-4",
            f"--lr_scheduler=constant",
            f"--lr_warmup_steps=0",
            f"--mixed_precision=bf16", # 수치적 안정성이 높은 bf16 지정 (지원 안 될 시 fp16 변경)
            f"--use_8bit_adam",  # bitsandbytes 기반의 8bit AdamW를 활성화할 때 추가하는 인자
            f"--rank=64",  # LoRA 차원 크기, 표현력 향상을 위해 Rank 상향 (선택 사항)
            f"--seed=42",
            # f"--gradient_checkpointing"  # VRAM 절약 필수 옵션, ai-engr vram 48gb 이므로 옵션 제거
        ]

        try:
            subprocess.run(cmd, check=True)
            # print(f"[{self.category}] LoRA 가중치 추출 완료: {self.output_dir}")

        except subprocess.CalledProcessError as e:
            pass
            # print(f"[{self.category}] 학습 중 오류 발생: {e}")

    def get_lora_weights_category_dir(self):
        return self.output_dir

if __name__ == "__main__":
    sample = LoraWeightExtractor()
    sample.extract_lora_weights("/home/ai-engr/KKCC/Model_A_SDXL/0628/can_padded",
                                "/home/ai-engr/KKCC/Model_A_SDXL/0628/")