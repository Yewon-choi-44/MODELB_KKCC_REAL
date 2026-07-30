<표준적인 프로젝트 구조와 정리 방식>

제공된 SDXL, SAM, LoRA 결합 파이프라인의 구성 요소들을 기반으로,</br>
실제 대학교 연구실이나 오픈소스 AI 연구 진영(예: Intel Anomalib, Hugging Face)에서 </br>
오픈소스화할 때 사용하는 표준적인 프로젝트 구조와 정리 방식을 제안합니다. </br>
* 전처리한 이미지 저장할 디렉토리명: prep_img
* 생성한 이미지 저장할 디렉토리명: agu_img </br>

## **1\. 표준 디렉토리 구조 (Directory Structure)**

기능별 역할을 명확히 분리하고 하드코딩을 방지하기 위해 설정을 외부로 빼는 구조입니다.  
`Data-Augmentation/`  
`├── configs/                   # 실험 하이퍼파라미터 및 경로 설정 (YAML)`  
`│   └── base_config.yaml`  
`├── src/                       # 메인 소스 코드 패키지`  
`│   ├── __init__.py`  
`│   ├── data/                  # 1~2단계 데이터 준비 및`  
`│   │   ├── preprocess.py      # 레터박스 패딩 및 캡셔닝 파일 생성 로직`  
`│   │   └── dataset.py         # MVTec 품목별 이미지 로더 정의`  
`│   ├── models/                # 3단계 개별 모델 객체 및 파이프라인 통합`  
`│   │   ├── __init__.py`  
`│   │   ├── sam_segmenter.py   # SAM 기반 마스크 추출 클래스`  
`│   │   └── sdxl_inpainter.py  # SDXL + LoRA 로드 및 추론 수행 클래스`  
`│   └── utils/                 # 4단계 최적화 및 공통 유틸리티`  
`│       ├── __init__.py`  
`│       └── hardware.py        # GPU VRAM 최적화 및 메모리 제어`  
`├── scripts/                   # 실행 가능한 스크립트 (CLI 엔트리포인트)`  
`│   ├── train_lora.sh          # train_text_to_image_lora_sdxl.py 실행 스크립트`  
`│   └── run_inference.py       # 전체 파이프라인 통합 실행 및 튜닝 스크립트`  
`├── requirements.txt           # 터미널 설치 라이브러리 목록 파일`  
`└── README.md                  # 깃헙 리포지토리 메인 설명 문서`

## 

## **2\. 주요 모듈별 클래스 및 함수 설계**

### **1\) 데이터 모듈 (src/data/preprocess.py)**

* **apply\_letterbox\_padding(image, target\_size=1024)**: 
* 이미지의 긴 쪽을 1024에 맞추고 빈 공간을 패딩하여 정방형 이미지로 변환합니다.  
* **generate\_caption\_file(image\_path, text\_content)**: 
* 이미지와 동일한 명칭의 설명 텍스트 파일(.txt)을 매칭하여 생성합니다.

### **2\) 모델 파이프라인 모듈 (src/models/)**

연구진들의 코드는 각 모델의 의존성을 줄이기 위해 핵심 기능을 클래스로 래핑(Wrapping)합니다.

* **SAMSegmenter 클래스 (sam\_segmenter.py)**:  
  * \_\_init\_\_(self, checkpoint\_path): sam\_vit\_h\_4b8939.pth 가중치를 로드하여 모델을 초기화합니다.  
  * get\_mask(self, image, points): 입력된 원본 이미지와 특정 좌표를 활용해 mask\_image를 추출합니다.  
* **SDXLInpainter 클래스 (sdxl\_inpainter.py)**:  
  * \_\_init\_\_(self, model\_id, device): StableDiffusionXLInpaintPipeline 베이스 모델을 로드합니다.  
  * load\_lora\_weights(self, lora\_path): 생성된 .safetensors 가중치를 파이프라인에 결합합니다.  
  * predict(self, prompt, image, mask\_image, scale, steps): 
  * 주입된 파라미터를 기반으로 최종 pipe()를 실행하여 이미지를 생성합니다.

### **3\) 하드웨어 유틸리티 모듈 (src/utils/hardware.py)**

* **enforce\_vram\_optimization(pipe)**: 
* 하드웨어 제약(VRAM 12GB\~16GB) 환경을 고려하여 pipe.enable\_model\_cpu\_offload() 등 메모리 최적화 코드를 제어합니다.

## 

## **3\. YAML 설정 파일 예시 (configs/base\_config.yaml)**

실험의 재현성(Reproducibility)을 위해 파라미터는 하드코딩하지 않고 아래와 같이 관리하는 것이 공식 관례입니다.  
`model_paths:`  
  `sdxl_base: "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"`  
  `sam_checkpoint: "sam_vit_h_4b8939.pth"`  
  `lora_weights: "C:/Users/사용자/projects/models/lora/my_custom_lora.safetensors"`

`inference_hyperparameters:`  
  `prompt: "A photo of [학습한 LoRA 트리거 단어], highly detailed, 4k"`  
  `num_inference_steps: 30`  
  `lora_scale: 0.8 # 4단계에서 조정할 가중치 비율`

## **4\. 참고 가능한 공식 레퍼런스 및 저장소 구조**

* **Intel Anomalib (openvinotoolkit/anomalib)**: 
* MVTec 데이터셋 처리 및 이미지 이상 탐지(Anomaly Detection) 분야에서 가장 구조화가 잘 된 공식 라이브러리입니다. 
* 데이터 로더 분할 방식을 벤치마킹하기 좋습니다.  
* 
* **Hugging Face Diffusers Examples (huggingface/diffusers/tree/main/examples)**: 
* 연구진들이 오픈소스로 공개하는 스크립트의 표준입니다. train\_text\_to\_image\_lora\_sdxl.py 등의 
* 구조가 독립적으로 깔끔하게 분리되어 있어 구조적 귀감이 됩니다.