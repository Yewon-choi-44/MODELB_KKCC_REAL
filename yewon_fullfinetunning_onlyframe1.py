"""
# 순정 아키텍처 보존형 풀 파인튜닝
    최종 역할
    - 원래 모델이 하던 일(이미지 생성)을 그대로 유지하되
        내부 가중치 행렬의 수치만 내 데이터에 맞춤형으로 고도화합니다.

    컴퓨터 내부의 동작
    - SD3Transformer2DModel이라는 기존의 컴퓨터 계산 레이어 뼈대를
        단 1밀리미터도 변형하지 않고 그대로 둡니다.

    - 입력되는 데이터의 형태([배치, 채널, 높이, 너비])와
        최종 출력되는 데이터의 형태가 순정 상태와 100% 동일합니다.

    - 오직 기존 레이어 내부에 채워져 있던 소수점 가중치 데이터들만
        역전파 연산으로 싹 다 새로 고침하여,
        특정 화풍이나 캐릭터를 더 잘 그리는 이미지 생성기를 만듭니다.

    손실 함수 (Loss) 기준
    - 원래 이미지와 노이즈 간의 평균제곱오차 (MSE)

    최종 목적
    - 기존 작업을 더 높은 품질로 수행하기 위함
"""

import torch
from accelerate import Accelerator
from transformers import CLIPTextModel
from diffusers import SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler
from torch.optim import AdamW


# 이미지를 만들땐 StableDiffusion3Pipeline
# 모델 학습/파인튜닝 시에는 SD3Transformer2DModel (=메인 신경망 MMDiT)


def run_full_finetuning():
    # 1. 가속화 엔진(Accelerator) 초기화
    # 앞서 말씀드린 bfloat16 하드웨어 가속을 총괄 제어하는 캡틴을 임명합니다.

    # B200 내부 텐서 코어(Tensor core)는 데이터 포맷을 bf16일 때 가장 빨리 돌아감.
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    # 2. 순정 모델 뼈대(구조)들 로드하기
    # SD 3.5 Large의 메인 두뇌(Transformer)와 글자 이해 장치(Text Encoder)를 불러옵니다.

    # https://huggingface.co/docs/diffusers/api/models/overview#diffusers.ModelMixin.from_pretrained

    transformer = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        subfolder="transformer")

    text_encoder = CLIPTextModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        subfolder="text_encoder")

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        subfolder="scheduler")

    # 🌟 [핵심] '풀 파인튜닝' 선언: 뇌세포 전체의 잠금장치를 해제합니다.
    # 모든 뉴런(매개변수)의 값을 수정할 수 있도록 허가(True)하는 단계입니다.
    transformer.requires_grad_(True)
    text_encoder.requires_grad_(True)

    # 3. 최적화 도구(Optimizer) 설정
    # VRAM이 넉넉한 B200이므로 성능 저하가 없는 32비트 순정 AdamW 최적화 도구를 연결합니다.
    optimizer = AdamW(
        list(transformer.parameters()) + list(text_encoder.parameters()),
        lr=1e-5
    )

    # 4. 준비된 부품들을 가속화 엔진에 일괄 등록
    # "이제부터 B200 하드웨어로 초고속 학습을 시작하겠다"고 세팅하는 함수입니다.
    transformer, text_encoder, noise_scheduler, optimizer = accelerator.prepare(
        transformer, text_encoder, noise_scheduler, optimizer
    )

    # 5. 무한 반복 학습 루프 (가장 핵심적인 훈련 과정)
    # 이미지와 캡션이 담긴 데이터셋에서 기출문제를 하나씩 꺼내옵니다.
    # (여기서는 예시를 위해 5,000번 반복한다고 가정합니다)
    for step in range(5000):
        optimizer.zero_grad()  # 예전 문제의 오답 기억을 지우고 새 마음으로 시작

        # [A] 가상의 이미지와 글자 데이터를 가져왔다고 가정 (Mock Data)
        mock_images = torch.randn(4, 16, 64, 64).to(device, dtype=torch.bfloat16)
        mock_input_ids = torch.randint(0, 1000, (4, 77)).to(device)

        # [B] 순방향 연산 (시험 치기)
        # AI에게 사진을 주고, 글자 인코더를 거쳐 그림을 그리게 시킵니다.
        encoder_hidden_states = text_encoder(mock_input_ids)[0]

        # 임의의 노이즈(배경 잡음)를 섞은 뒤 모델이 이를 얼마나 잘 복원하는지 계산합니다.
        noise = torch.randn_like(mock_images)
        timesteps = torch.randint(0, 1000, (4,), device=device).long()

        # AI가 예측한 결과물 출력
        model_pred = transformer(
            mock_images,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states).sample

        # [C] 손실(Loss) 계산: 정답지랑 비교해서 점수 매기기
        # 원래 이미지(noise)와 AI가 그린 결과물(model_pred)이 얼마나 닮았는지 '오차 점수'를 냅니다.
        loss = torch.nn.functional.mse_loss(model_pred, noise, reduction="mean")

        # [D] 역방향 연산 (오답노트 작성 및 뇌세포 수정)
        # 오차가 발생한 원인을 역추적하여 전 뉴런에 전달하고(backward),
        # 최적화 도구(optimizer)를 통해 뇌 가중치 파일의 숫자를 미세하게 고칩니다(step).
        accelerator.backward(loss)
        optimizer.step()

        if step % 100 == 0:
            print(f"현재 훈련 단계: {step}/5000 | 오차 점수(Loss): {loss.item():.4f}")

    # 6. 학습이 끝난 나만의 '새로운 완성형 뇌 파일'을 폴더에 저장합니다.
    # 이 결과물이 바로 우리가 앞서 이야기한 .safetensors 파일로 변환되는 데이터입니다.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer.save_pretrained("./my_b200_sd35_full_model")
        print("🎉 축하합니다! 풀 파인튜닝이 완료되어 새 모델 파일이 저장되었습니다.")


if __name__ == "__main__":
    run_full_finetuning()