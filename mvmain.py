from PIL import Image

from mvpreprocess import padding, cd_extractor, dora_trainer
from mvgenerator import augmentor, restorer

# from padding import ImageSquarePadder
# from cationer import DataCaptioner
# from cd_extractor import CannyEdgeExtractor, DepthMapExtractor
# from dora_trainer import DoRAWeightGenerator
# from augmentor import ControlNetImageGenerator
# from restorer import HighResDetailRestorer


def main(user_dataset_name, category_names):

    raw_dataset_path = "../" + user_dataset_name + "/" + category_names[0] + "/"    
    dora_weight_output = ""
    text_prompt = ""


    # # 테스트를 위한 가상 이미지 파일 생성
    # Image.new("RGB", (800, 600), (120, 150, 80)).save(raw_dataset_path)


    print("=== MODEL B 파이프라인 시작 ===")


    # 1. padding 모듈 호출
    padder = padding.ImageSquarePadder(target_size=1024)
    padded_img = padder.process(raw_dataset_path)


    # 2&3단계: canny, depth 모듈 호출
    canny = cd_extractor.CannyEdgeExtractor()
    depth = cd_extractor.DepthMapExtractor()
    canny_guide = canny.extract(padded_img) # padded_img가 카테고리 폴더 통째로인지 그 안에 이미지 한 장인지 확인 필요
    depth_guide = depth.extract(padded_img)


    # 4단계: DoRA 가중치 훈련 모듈 호출
    dora = dora_trainer.DoRAWeightGenerator()
    dora.train_and_save(dataset_folder="./train_dataset/leak", 
                        save_path=dora_weight_output)


    # 5단계: ControlNet 증강 생성 모듈 호출
    augmentor = augmentor.ControlNetImageGenerator()


    # 실제 환경 가동 시 주석 해제 가능
    # generated_raw_img = padded_img # 시뮬레이션용 대체
    gen_raw_img = augmentor.generate(text_prompt, 
                                    canny_guide, 
                                    depth_guide, 
                                    dora_weight_output)


    # 6단계: SUPIR 고해상도 질감 복원 모듈 호출
    restorer = restorer.HighResDetailRestorer()
    res = restorer.restore_details(gen_raw_img, 
                                   prompt=text_prompt)


    # 최종 결과물 저장
    res.save(f"gen_{category_names[0]}.png")
    print("\n 증강 이미지 파일 생성을 완료")



if __name__ == "__main__":
    
    user_dataset_name = "mvtec_ad_2"
    category_names = ["can",
                    "fabric",
                    "fruit_jelly",
                    "rice",
                    "sheet_metal",
                    "vial",
                    "wallplugs",
                    "walnuts"]

    main()