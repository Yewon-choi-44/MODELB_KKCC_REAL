from PIL import Image, ImageOps
import os

def pad_image(self, idx : int, target_size=1024):
    
    raw_image = self.RawDataset.get_image(idx)
    raw_image.thumbnail(target_size, target_size)

    delta_w = target_size - raw_image.width
    delta_h = target_size - raw_image.height

    padding = (delta_w//2 , delta_h//2 , delta_w-(delta_w//2), delta_h-(delta_h//2  ))
    padding_image = ImageOps.expand(raw_image, padding, fill = 'white')
    padding_image_name = self.object + "_" + str(idx).zfill(3) + ".png"
    image_path = os.path.join(
        self.PaddingImages.set_image_path(), padding_image_name)
    padding_image.save(image_path)

    
def padding_all_images_in_object(self):
    for idx in range(RawDataset.__len__(self.RawDataset)):
        self.pad_image(idx)
        print(f"인덱스 {idx+1}/{RawDataset.__len__(self.RawDataset)}번 사진 패딩 완료")
        
