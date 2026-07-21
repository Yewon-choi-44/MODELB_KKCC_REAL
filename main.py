import os
import argparse
from preprocessor import ImageSquarePadder
from extractor import CannyEdgeExtractor, DepthMapExtractor
from augmentor import ControlNetAugmentor
from restorer import SUPIRImageRestorer

def run_augmentation_pipeline(args):
    # 0. 경로 및 파라미터 초기 설정
    src_img_path = args.input_image
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(">>> 1단계: 이미지 규격화 및 전처리 <<<")
    padder = ImageSquarePadder(target_size=1024)
    padded_img = padder.process(src_img_path)
    padded_img.save(os.path.join(output_dir, "step1_padded.png"))
    
    print(">>> 2, 3단계: Canny 외곽선 및 Depth 입체 구조 추출 <<<")
    canny_ext = CannyEdgeExtractor()
    depth_ext = DepthMapExtractor()
    
    canny_guide = canny_ext.extract(padded_img)
    canny_guide.save(os.path.join(output_dir, "step2_canny_guide.png"))
    
    depth_guide = depth_ext.extract(padded_img)
    depth_guide.save(os.path.join(output_dir, "step3_depth_guide.png"))
    
    # 4단계는 trainer.py를 통해 오프라인으로 완료되었다고 가정하고 튜닝 가중치 주입
    print(">>> 5단계: ControlNet 조건부 증강 생성 <<<")
    augmentor = ControlNetAugmentor()
    if args.finetuned_weights:
        augmentor.update_weights(args.finetuned_weights)
        
    generated_raw = augmentor.generate(
        prompt=args.prompt,
        negative_prompt="blurry, low quality, distorted structure, generic texture",
        canny_img=canny_guide,
        depth_img=depth_guide,
        control_scales=[0.7, 0.3]
    )
    generated_raw.save(os.path.join(output_dir, "step5_generated_raw.png"))
    
    print(">>> 6단계: SUPIR 미세 스크래치 질감 복원 <<<")
    restorer = SUPIRImageRestorer()
    final_output = restorer.restore(
        image=generated_raw,
        prompt="micro crack defect on metal surface, high-resolution, industrial camera detail",
        scale=2
    )
    
    final_output.save(os.path.join(output_dir, "step6_final_augmented.png"))
    print(f"🎉 전체 파이프라인 실행 완료! 결과물 저장 위치: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KKCC Model B Anomaly Defect Data Augmentation Pipeline")
    parser.add_argument("--input_image", type=str, required=True, help="가공할 원본 불량 이미지 파일 경로")
    parser.add_argument("--output_dir", type=str, default="./output_results", help="결과물이 저장될 폴더 경로")
    parser.add_argument("--finetuned_weights", type=str, default=None, help="파인튜닝된 SD3.5 Transformer 가중치 폴더 경로")
    parser.add_argument("--prompt", type=str, default="defect area with scratch, rust, and crack, photorealistic", help="생성형 AI 타겟 불량 설명 프롬프트")
    
    args = parser.parse_args()
    run_augmentation_pipeline(args)