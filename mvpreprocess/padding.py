from PIL import Image

class ImageSquarePadder:
    """이미지를 비율 유지하며 조정하고, 빈 공간을 흰색으로 채우는 클래스"""
    def __init__(self, target_size=1024):
        self.target_size = (target_size, target_size)

    def process(self, image_path: str) -> Image.Image:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail(self.target_size, Image.Resampling.LANCZOS)
        
        white_canvas = Image.new("RGB", self.target_size, (255, 255, 255))
        paste_position = (
            (self.target_size[0] - image.size[0]) // 2,
            (self.target_size[1] - image.size[1]) // 2
        )
        white_canvas.paste(image, paste_position)
        return white_canvas