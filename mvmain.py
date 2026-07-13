from PIL import Image

# 🛠️ 우리가 직접 만든 5개의 파일(모듈)에서 클래스를 가져옵니다.
from preprocessor import ImageSquarePadder
from extractor import CannyEdgeExtractor, DepthMapExtractor
from dora_trainer import DoRAWeightGenerator
from augmentor import ControlNetImageGenerator
from restorer import HighResDetailRestorer

def main():
    # 설정 경로 및 예시 파일명
    sample_image_path = "raw_defect_leak.png"
    dora_weight_output = "dora_defect_weights.safetensors"
    text_prompt = "An industrial oil-leak defect on a metallic surface, high resolution"

    # 테스트를 위한 가상 이미지 파일 생성
    Image.new("RGB", (800, 600), (120, 150, 80)).save(sample_image_path)

    print("=== [시스템 가동] 데이터 증강 파이프라인 시작 ===")

    # 1단계: 전처리 모듈 호출
    padder = ImageSquarePadder(target_size=1024)
    padded_img = padder.process(sample_image_path)

    # 2&3단계: 특징 추출 모듈 호출
    canny_extractor = CannyEdgeExtractor()
    depth_extractor = DepthMapExtractor()
    canny_guide = canny_extractor.extract(padded_img)
    depth_guide = depth_extractor.extract(padded_img)

    # 4단계: DoRA 가중치 훈련 모듈 호출
    dora_trainer = DoRAWeightGenerator()
    dora_trainer.train_and_save(dataset_folder="./train_dataset/leak", save_path=dora_weight_output)

    # 5단계: ControlNet 증강 생성 모듈 호출
    augmentor = ControlNetImageGenerator()
    # 실제 환경 가동 시 주석 해제 가능
    # generated_raw_img = augmentor.generate(text_prompt, canny_guide, depth_guide, dora_weight_output)
    generated_raw_img = padded_img # 시뮬레이션용 대체

    # 6단계: SUPIR 고해상도 질감 복원 모듈 호출
    restorer = HighResDetailRestorer()
    final_output = restorer.restore_details(generated_raw_img, prompt=text_prompt)

    # 최종 결과물 저장
    final_output.save("final_augmented_defect_data.png")
    print("\n🎉 모든 파일 모듈이 연계되어 증강 이미지 파일 생성을 완료했습니다!")

if __name__ == "__main__":
    main()