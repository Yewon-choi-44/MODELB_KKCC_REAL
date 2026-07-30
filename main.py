from process import Processing
dataset_name = ['MVTEC_AD_1', 'MVTEC_AD_2']

MVTEC_AD_1_OBJECTS = [ 'bottle', 'cable', 'capsule', 'carpet', 'grid',
                       'hazelnut', 'leather', 'metal_nut', 'pill',
                       'screw', 'tile', 'toothbrush', 'transistor',
                       'wood', 'zipper', ]

MVTEC_AD_2_OBJECTS = ['can', 'fabric', 'fruit_jelly', 'rice',
                      'sheet_metal', 'vial', 'wallplugs', 'walnuts', ]
PROMPT = ("An ultra-detailed, high-resolution macro photograph for industrial computer vision anomaly detection. "
          "A close-up top-down view of an industrial product, showing clear structural physical defects including "
          "sharp surface scratches, local mechanical dents, localized surface contamination, and structural fractures. "
          "The defect reveals realistic surface deformation and authentic material texture. Set against a clean, neutral, "
          "non-reflective background with uniform diffuse industrial inspection lighting, shadowless illumination, "
          "perfectly centered subject, strictly shot with a professional telecentric lens, "
          "sharp focus across the entire depth of field, 8k resolution.")
NEGATIVE_PROMPT = ("artistic rendering, stylized, conceptual, bokeh, blurry, depth of field blur, tilted camera angle, "
                   "perspective distortion, dramatic warm lighting, harsh shadows, dirty background, noisy background, "
                   "glares, lens flare, text, watermark, logo, hands, human fingers, out of frame, cropped subject, "
                   "low resolution, artifacts.")

if __name__ == "__main__":
    input_object_name = input("이미지 증강할 품목을 선택하시오.(all = 모든 품목 / object_name = 해당 품목")
    if input_object_name == "all":
        for object_1 in MVTEC_AD_1_OBJECTS:
            object_name = object_1
            preprocessor = Processing(dataset_name='MVTEC_AD_1',
                                      object_name=object_name,
                                      prompt=PROMPT,
                                      negative_prompt=NEGATIVE_PROMPT)
            # preprocessor.padding_all_images_in_object()
            preprocessor.generate_image_SD35()
        for object_2 in MVTEC_AD_2_OBJECTS:
            object_name = object_2
            preprocessor = Processing(dataset_name='MVTEC_AD_2',
                                      object_name=object_name,
                                      prompt=PROMPT,
                                      negative_prompt=NEGATIVE_PROMPT)
            # preprocessor.padding_all_images_in_object()
            preprocessor.generate_image_SD35()
    else :
        object_name = input_object_name
        preprocessor = Processing(dataset_name='MVTEC_AD_1', object_name=object_name, )
        preprocessor.padding_all_images_in_object()
        preprocessor.generate_image_SD35()
