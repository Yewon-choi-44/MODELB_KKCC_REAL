import os
from PIL import Image
from transformers import pipeline

class DataCaptioner:
    """폴더 안의 이미지를 자동으로 읽어 영어 지시서(프롬프트) 베이스 문장을 만들어내는 AI 클래스"""

    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):

        
        # 이미지 투 텍스트(Image-to-Text) 파이프라인 구축
        self.caption_pipeline = pipeline("image-to-text", model=model_name)

    def generate_caption(self, folder_path: str) -> str:

        """
        지정된 폴더의 첫 번째 불량 이미지를 분석하여 
        Model B 파이프라인 전체에서 고정으로 사용할 '대표 뭉뚱그린 문장' 한 줄을 자동으로 추출합니다.
        """
        
        if not os.path.exists(folder_path):
            print(f"❌ [경로 에러] '{folder_path}' 폴더를 찾을 수 없습니다.")
            return "An industrial product image"

        # 읽어올 수 있는 이미지 확장자 필터링
        valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")
        all_files = os.listdir(folder_path)
        image_files = [f for f in all_files if f.lower().endswith(valid_extensions)]

        if not image_files:
            print(f"⚠️ '{folder_path}' 폴더 내에 이미지 파일이 존재하지 않습니다.")
            return "A product sample photograph"

        # 폴더 내 가장 첫 번째 샘플 이미지를 대표로 가져와 읽기
        sample_image_name = image_files[0]
        full_image_path = os.path.join(folder_path, sample_image_name)
        
        print(f"📂 [자동 캡션 가동] 샘플 이미지 '{sample_image_name}' 분석 시작...")
        img = Image.open(full_image_path).convert("RGB")
        
        # AI 모델 작동 (이미지를 넣으면 글이 나옴)
        outputs = self.caption_pipeline(img)
        generated_raw_caption = outputs[0]['generated_text']
        
        # 조금 더 고품질의 SD3.5 프롬프트가 되도록 '고화질(high resolution)' 수식어 단어 결합
        final_prompt = f"{generated_raw_caption}, high resolution, industrial defect sample"
        
        print(f"✨ [AI 자막 생성 완료]: \"{final_prompt}\"")
        return final_prompt