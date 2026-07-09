import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3ControlNetPipeline, ControlNetModel
from transformers import pipeline

# ==========================================
# 1. 준비 단계: 가이드라인 이미지 준비 및 특징 추출
# ==========================================

# 원본 이미지를 불러옵니다. (Canny와 Depth를 추출할 기준 이미지)
input_image_path = "input_image.png"
init_image = Image.open(input_image_path).convert("RGB")

# [1-A] Canny Edge (외곽선) 추출
# 이미지를 흑백 선으로 따서 형태의 기준을 만듭니다.
np_image = np.array(init_image)
canny_matrix = cv2.Canny(np_image, 100, 200)  # 100과 200은 선을 얼마나 촘촘하게 딸지 결정하는 문턱값입니다.
canny_matrix = canny_matrix[:, :, None]
canny_image = np.concatenate([canny_matrix, canny_matrix, canny_matrix], axis=2)
canny_image = Image.fromarray(canny_image)  # AI가 읽을 수 있는 이미지 형태로 변환

# [1-B] Depth Map (입체감/깊이) 추출
# 가깝고 먼 거리 정보를 담은 회색조 지도를 만듭니다.
depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
depth_image = depth_estimator(init_image)["depth"]


# ==========================================
# 2. 모델 로드 단계: AI 건축가와 설계도 배치하기
# ==========================================

# SD 3.5 버전에 맞는 ControlNet 모델들을 불러옵니다.
controlnet_canny = ControlNetModel.from_pretrained(
    "InstantX/SD3.5-Large-ControlNet-Canny", 
    torch_dtype=torch.float16
)
controlnet_depth = ControlNetModel.from_pretrained(
    "InstantX/SD3.5-Large-ControlNet-Depth", 
    torch_dtype=torch.float16
)

# Full Fine-tuning이 완료된 Stable Diffusion 3.5 Large 모델을 생성 파이프라인에 연결합니다.
# "my_finetuned_sd35_large_path" 부분에 학습 완료된 본인의 모델 경로를 적어줍니다.
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    "my_finetuned_sd35_large_path",
    controlnet=[controlnet_canny, controlnet_depth], # 두 가지 제어 모델을 동시에 사용
    torch_dtype=torch.float16
)
pipe.to("cuda") # 컴퓨터의 그래픽카드(GPU)를 사용하여 속도를 높입니다.


# ==========================================
# 3. 이미지 생성 단계: SD 3.5 + Multi-ControlNet
# ==========================================

prompt = "A cinematic, highly detailed masterwork photo, 8k resolution"

# 가이드라인(Canny, Depth)을 바탕으로 새 이미지를 그려냅니다.
generated_image = pipe(
    prompt=prompt,
    control_image=[canny_image, depth_image],        # 추출했던 두 지도를 주입
    controlnet_conditioning_scale=[0.6, 0.4],       # Canny 선 반영 비율 60%, Depth 입체감 반영 비율 40%
    num_inference_steps=30                          # 이미지를 얼마나 정성 들여 깎아낼지 (반복 횟수)
).images[0]

generated_image.save("step1_sd35_output.png")


# ==========================================
# 4. 마감 복원 단계: SUPIR를 통한 초고해상도 디테일 업그레이드
# ==========================================

# 💡 SUPIR는 대형 모델이므로 공식 깃허브 코드 구조를 활용해 호출하는 것이 일반적입니다.
# 아래는 SUPIR를 파이프라인 끝단에 연결하는 직관적인 흐름 예시입니다.
def run_supir_restoration(image, text_prompt):
    print("SUPIR 마감 정밀 복원 시작...")
    # 실제 환경에서는 SUPIR 리포지토리의 전용 로더를 사용합니다.
    # from SUPIR.util import create_SUPIR_model
    
    # 1. SUPIR 모델에 가공된 이미지와 텍스트 입력
    # 2. 미세한 스크래치 제거 및 초고해상도 픽셀 디테일 질감 복원 수행
    
    # 이해를 돕기 위한 개념적 반환 코드입니다.
    restored_image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS) 
    return restored_image

# 최종 결과물 완성
final_output_image = run_supir_restoration(generated_image, prompt)
final_output_image.save("final_pipeline_output.png")
print("모든 파이프라인이 성공적으로 완료되었습니다!")