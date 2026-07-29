import os
import json
import torch
from PIL import Image, ImageOps
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusion3Img2ImgPipeline
from huggingface_hub import login
import tokens

from data import RawDataset, PaddingImages, set_metadata_path, set_generated_image_path

class Processing:
    def __init__(self,
                 dataset_name = 'MVTEC_AD_2',
                 object_name = "can",
                 split = "train",
                 prompt = "A highly detailed, macro top-down view of an industrial [object name], "
                          "centrally positioned on a flat, neutral gray background. The surface of the object exhibits "
                          "[Defect Description / pristine condition]. Illuminated by uniform, diffused industrial studio "
                          "lighting to eliminate harsh shadows and specular highlights. "
                          "Professional industrial inspection photography, high resolution.",
                 negative_prompt = "Depth of field, bokeh, out of focus, harsh shadows, directional lighting, "
                                   "colorful background, multiple objects, text, watermark, distorted geometry, "
                                   "artistic rendering, low resolution, noise, artifacts.", ):
        # 클래스 composition(합성) : 객체가 다른 객체를 가지고(has-a) 있어 그 기능을 활용하게 하는 방식
        self.object = object_name
        self.RawDataset = RawDataset(dataset_name = dataset_name,
                                      object_name = object_name,
                                      split = split)
        self.PaddingImages = PaddingImages(object_name = object_name)
        self.prompt = prompt
        self.negative_prompt = negative_prompt

    def pad_image(self, idx : int, target_size=1024):
        """ def: 이미지 생성을 위해 letterbox padding을 적용함
            letter padding이란 원본 이미지의 비율을 유지해서 크기를 맞추고 여백은 정한 색으로 채움
        Args:
            self.raw_image (PIL) : MVTecAD2.get_image(idx:int)
            :param target_size: 기본값 1024로 설정. output_image 크기 1024*1024
            :param idx:
        """
        raw_image = self.RawDataset.get_image(idx)

        # 비율 유지하며 리사이즈
        raw_image.thumbnail((target_size, target_size))

        # 중앙 정렬 후 패딩 처리
        delta_w = target_size - raw_image.width
        delta_h = target_size - raw_image.height
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        padding_image = ImageOps.expand(raw_image, padding, fill='white') # 흰색으로 채움
        padding_image_name = self.object + "_" + str(idx).zfill(3) + ".png"
        image_path = os.path.join(self.PaddingImages.set_image_path(), padding_image_name)
        padding_image.save(image_path)

    def padding_all_images_in_object(self):
        for idx in range(RawDataset.__len__(self.RawDataset)):
            self.pad_image(idx)
            print(f"{self.object} : 인덱스 {idx+1}/{RawDataset.__len__(self.RawDataset)}번 사진 패딩 완료")

    def generate_caption(self, idx : int):
        """ def: pad_image 함수를 통해 패딩된 이미지에서 caption을 도출함
        Args:
            :param idx:
        """
        if torch.cuda.is_available():
            device = "cuda"
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

            # caption 생성
            padding_image = self.PaddingImages.get_image(idx)
            input_image = processor(padding_image, return_tensors="pt").to(device)
            output_1 = model.generate(**input_image, max_new_tokens = 50)
            output_2 = processor.decode(output_1[0], skip_special_tokens=True)
            caption = self.object + ',' + output_2
            return caption

    def create_diffusers_metadata(self):
        """ def: generate_caption 함수를 통해 생성된 caption을 diffusion모델에 넣을 수 있는 형식인 jsonl로 변환함
        Args:

        """
        metadata_path = set_metadata_path(self.object)
        with open(metadata_path, "w", encoding="utf-8") as f_jsonl:
            for idx in range(RawDataset.__len__(self.RawDataset)):
                image_name = self.object + "_" + str(idx).zfill(3) + ".png"
                caption = self.generate_caption(idx)
                # jsonl 작성
                line = {"file_name": image_name, "text": caption}
                f_jsonl.write(json.dumps(line, ensure_ascii=False) + "\n")
                print(f"인덱스 {idx}번 사진 caption 작성 완료")

    def Segmentation(self):
        pass

    def generate_image_SDXL(self):
        self.PaddingImages.image_paths
        if torch.cuda.is_available():
            device = "cuda"
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                torch_dtype=torch.float16,
                variant="fp16"
            ).to(device)
            model_name = "SDXL"
            generated_image_path = set_generated_image_path(model_name, self.object)
            for idx in range(RawDataset.__len__(self.RawDataset)):
                padding_image = self.PaddingImages.get_image(idx)
                generated_image = pipe(
                    prompt = self.prompt,
                    negative_prompt = self.negative_prompt,
                    image = padding_image,
                    height = 1024,
                    width = 1024,
                    num_inference_steps = 70,
                ).images[0]
                generated_image_name = "gen_" + self.object + "_" + str(idx).zfill(3) + ".png"
                image_path = os.path.join(generated_image_path, generated_image_name)
                generated_image.save(image_path)
                print(f"{self.object} : 인덱스 {idx+1}번 증강이미지 생성 완료")

    def generate_image_SD35(self):
        self.PaddingImages.image_paths
        if torch.cuda.is_available():
            device = "cuda"
            login(token=tokens.token)
            pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-3.5-large",
                torch_dtype=torch.float16,
                variant="fp16"
            ).to(device)

            shared_params = {
                "prompt" : self.prompt,
                "negative_prompt" : self.negative_prompt,
                "strength" : 0.3,  # 원본 이미지 변경 강도(0.0~1.0) 값이 클수록 프롬프트 반영률 높아지고 원본과 달라짐
                "guidance_scale" : 4.0,  # CFG Scale : SD35 권장값인 3.5~4.5 적용, 텍스트 프롬프트를 얼마나 강하게 반영할지.
                "height" : 1024,
                "width" : 1024,
                "num_inference_steps" : 50,
            }
            batch_metadata = {
                "shared_params" : shared_params,
                "history" : []
            }
            current_seed = 42
            generator = torch.Generator(device).manual_seed(current_seed)

            model_name = "SD35"
            generated_image_path = set_generated_image_path(model_name, self.object)
            for idx in range(RawDataset.__len__(self.RawDataset)):
                padding_image = self.PaddingImages.get_image(idx)
                input_image_paths = self.PaddingImages.image_paths
                input_image_path = input_image_paths[idx]
                generated_image = pipe(
                    prompt=shared_params["prompt"],
                    negative_prompt=shared_params["negative_prompt"],
                    image=padding_image,
                    strength=shared_params["strength"],  # 원본 이미지 변경 강도(0.0~1.0) 값이 클수록 프롬프트 반영률 높아지고 원본과 달라짐
                    guidance_scale=shared_params["guidance_scale"],  # CFG Scale : SD35 권장값인 3.5~4.5 적용, 텍스트 프롬프트를 얼마나 강하게 반영할지.
                    height=shared_params["height"],
                    width=shared_params["width"],
                    num_inference_steps=shared_params["num_inference_steps"],
                    generator = generator,
                ).images[0]
                generated_image_name = "gen_" + self.object + "_" + str(idx).zfill(3) + ".png"
                image_path = os.path.join(generated_image_path, generated_image_name)
                generated_image.save(image_path)
                batch_metadata["history"].append({
                    "input_file": input_image_path,
                    "output_file": image_path,
                    "seed": current_seed
                })
                print(f"{self.object} : 인덱스 {idx + 1}번 증강이미지 생성 완료")
            batch_metadata_name = f"{self.object}_batch_metadata.json"
            batch_metadata_path = os.path.join(generated_image_path, batch_metadata_name)
            with open(batch_metadata_path, "w", encoding = "utf-8") as f:
                json.dump(batch_metadata, f, indent = 4, ensure_ascii = False)
            print(f"{self.object} 증강 이미지 및 메타데이터 생성 완료")



