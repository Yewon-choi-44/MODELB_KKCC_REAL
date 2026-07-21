import os
import glob
from PIL import Image

class ImageSquarePadder:
    """이미지를 비율 유지하며 조정하고, 빈 공간을 흰색으로 채우는 클래스"""
    def __init__(self, target_size=1024):
        self.target_size = (target_size, target_size)

    def process(self, image_path: str) -> Image.Image:
        image = Image.open(image_path).convert("RGB")
        # 비율 유지하며 액자 크기 내로 축소
        image.thumbnail(self.target_size, Image.Resampling.LANCZOS)
        # 하얀색 도화지 생성
        white_canvas = Image.new("RGB", self.target_size, (255, 255, 255))
        # 정중앙 배치 좌표 계산
        paste_position = (
            (self.target_size[0] - image.size[0]) // 2,
            (self.target_size[1] - image.size[1]) // 2
        )
        white_canvas.paste(image, paste_position)
        return white_canvas

def preprocess_dataset(src_dir: str, dst_dir: str, target_size=1024):
    """지정된 디렉토리 내의 모든 원본 이미지를 정방형 패딩 처리하여 저장"""
    os.makedirs(dst_dir, exist_ok=True)
    padder = ImageSquarePadder(target_size=target_size)
    
    image_paths = glob.glob(os.path.join(src_dir, "**/*.png"), recursive=True) + \
                  glob.glob(os.path.join(src_dir, "**/*.jpg"), recursive=True)
                  
    print(f"[Preprocessor] 총 {len(image_paths)}개의 이미지를 변환합니다...")
    for idx, path in enumerate(image_paths):
        padded_img = padder.process(path)
        filename = f"padded_{idx:04d}.png"
        padded_img.save(os.path.join(dst_dir, filename))
    print(f"[Preprocessor] 변환 완료 -> 저장경로: {dst_dir}")

if __name__ == "__main__":
    # 단독 테스트 코드
    preprocess_dataset(
        src_dir="./raw_dataset/can/train/bad",
        dst_dir="./padded_dataset/can/train/bad"
    )