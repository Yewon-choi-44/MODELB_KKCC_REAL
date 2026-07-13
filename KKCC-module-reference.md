---
type: Concept
title: DataAugmentation 모듈 레퍼런스
description: KKCC_AnomalyDetection/DataAgumentation 프로젝트의 모든 모듈에서 사용된 클래스명, 함수명, 변수명을 정리하고 각 항목의 역할을 주석으로 설명한 문서
tags: [python, kkcc, anomaly-detection, data-augmentation, reference]
timestamp: 2026-07-13
status: active
source: D:/KKCC_Project/KKCC_AnomalyDetection/DataAgumentation
---

# DataAugmentation 모듈 레퍼런스

**프로젝트 경로:** `D:/KKCC_Project/KKCC_AnomalyDetection/DataAgumentation`

이 문서는 이상 탐지(Anomaly Detection) 프로젝트의 데이터 증강 파이프라인에 사용된 모든 모듈의 클래스, 함수, 변수를 모듈별로 정리한 레퍼런스입니다.

---

## 📦 프로젝트 전체 구조 개요

```
DataAgumentation/
├── data.py          # 데이터셋 경로 관리 및 이미지 로드 담당
├── process.py       # 핵심 이미지 처리 파이프라인 (패딩 → 캡션 → 이미지 생성)
├── main.py          # 전체 파이프라인 실행 진입점
└── later/           # 추후 도입 예정 고급 모듈
    ├── sam_segmenter.py    # SAM 기반 객체 마스크 추출
    └── sdxl_inpainter.py   # SDXL + LoRA 기반 인페인팅 이미지 생성
```

---

## 1. `data.py`

> 데이터셋의 경로 설정, 이미지 파일 목록 관리, 이미지 로드를 담당하는 모듈.
> MVTec AD 1/2 데이터셋을 지원하며, 패딩 이미지 출력 경로도 함께 관리한다.

---

### 🔵 전역 변수 (Module-level Variables)

| 변수명 | 값 / 타입 | 역할 |
|---|---|---|
| `root` | `str` | 데이터셋 및 출력 폴더가 위치한 최상위 경로 |
| `PATH_TO_MVTEC_AD_1_FOLDER` | `str` | MVTec AD 1 데이터셋의 절대 경로 |
| `MVTEC_AD_1_OBJECTS` | `list[str]` | MVTec AD 1에 포함된 15개 품목 이름 목록 |
| `PATH_TO_MVTEC_AD_2_FOLDER` | `str` | MVTec AD 2 데이터셋의 절대 경로 |
| `MVTEC_AD_2_OBJECTS` | `list[str]` | MVTec AD 2에 포함된 8개 품목 이름 목록 |
| `PADDING_IMAGE_FOLDER` | `str` | 패딩 처리된 이미지를 저장하는 출력 폴더 경로 |
| `METADATA_FOLDER` | `str` | 캡션 메타데이터(.jsonl)를 저장하는 출력 폴더 경로 |
| `GENERATED_IMAGE_FOLDER` | `str` | Diffusion 모델로 생성된 증강 이미지를 저장하는 폴더 경로 |

---

### 🟠 클래스: `RawDataset`

> MVTec AD 1 또는 MVTec AD 2 데이터셋에서 원본 이미지 경로를 관리하고 개별 이미지를 로드하는 기반 클래스.

#### 인스턴스 변수 (Instance Variables)

| 변수명 | 역할 |
|---|---|
| `self.dataset` | 사용 중인 데이터셋 이름 (`'MVTEC_AD_1'` 또는 `'MVTEC_AD_2'`) |
| `self.object` | 분석할 품목 이름 (예: `'can'`, `'bottle'`) |
| `self.split` | 데이터 분할 기준 (`'train'`, `'test'`, `'validation'` 등) |
| `self._image_base_dir` | 데이터셋의 루트 폴더 경로 |
| `self._object_dir` | 특정 품목 폴더의 절대 경로 |
| `self._image_paths` | glob으로 수집된 이미지 파일 경로 목록 (정렬됨) |

#### 메서드 (Methods)

```python
def __init__(self, dataset_name='MVTEC_AD_2', object_name='can', split='train')
```
> 데이터셋 이름, 품목 이름, 분할 종류를 받아 이미지 경로 목록(`self._image_paths`)을 자동으로 초기화한다.

---

```python
def get_image_path(self) -> str
```
> 데이터셋 종류(`MVTEC_AD_1` / `MVTEC_AD_2`)와 `split` 값에 따라 이미지를 glob으로 탐색할 패턴 경로 문자열을 반환한다.
> 예: `train/good/*.png`, `test/*/*.png`

---

```python
def get_image(self, idx: int)
```
> 정렬된 이미지 경로 목록에서 `idx` 번째 이미지를 PIL Image 객체로 로드하여 반환한다.
>
> | 매개변수 | 설명 |
> |---|---|
> | `idx` | 로드할 이미지의 인덱스 |

---

```python
def __len__(self)
```
> 데이터셋에서 탐색된 전체 이미지 파일의 개수를 반환한다. (`len(dataset)` 형태로 사용)

---

```python
@property
def image_paths(self)
```
> `self._image_paths` (이미지 경로 목록)를 읽기 전용 프로퍼티로 외부에 공개한다.

---

### 🟠 클래스: `PaddingImages` ← `RawDataset` 상속

> `RawDataset`을 상속하여, 패딩 처리된 이미지가 저장되는 출력 폴더를 관리하는 클래스.
> 원본 데이터셋 경로 대신 `PADDING_IMAGE_FOLDER` 경로를 사용한다.

#### 인스턴스 변수 (Instance Variables)

| 변수명 | 역할 |
|---|---|
| `self.object` | 품목 이름 |
| `self._image_base_dir` | 패딩 이미지 저장 루트 폴더 (`PADDING_IMAGE_FOLDER`) |
| `self._object_dir` | 특정 품목의 패딩 이미지 폴더 경로 |
| `self._image_paths` | 초기에는 `None`, 처음 접근 시 lazy-load로 경로 목록을 채운다 |

#### 메서드 (Methods)

```python
def __init__(self, object_name, padding_image_folder=PADDING_IMAGE_FOLDER)
```
> 품목 이름과 패딩 이미지 저장 폴더 경로를 받아 객체를 초기화한다.

---

```python
def set_image_path(self)
```
> 패딩 이미지를 저장할 품목 폴더(`self._object_dir`)가 없으면 자동으로 생성(makedirs)하고, 그 경로를 반환한다.

---

```python
def get_image_path(self)
```
> 패딩 이미지 폴더 내의 모든 PNG 파일을 glob으로 탐색하는 패턴 경로를 반환한다. (`*.png`)

---

```python
@property
def image_paths(self)
```
> `self._image_paths`가 `None`이면 그 시점에 glob 탐색을 수행하여 경로 목록을 채우고 반환하는 **지연 로딩(Lazy Loading)** 프로퍼티.

---

### 🟢 독립 함수 (Module-level Functions)

```python
def set_metadata_path(object_name)
```
> 품목 이름을 받아 메타데이터 JSONL 파일의 전체 경로(`METADATA_FOLDER/{object_name}_metadata.jsonl`)를 생성하여 반환한다.

---

```python
def set_generated_image_path(object_name)
```
> 품목 이름을 받아 증강 이미지 저장 폴더(`GENERATED_IMAGE_FOLDER/{object_name}`)를 생성(makedirs)하고, 그 경로를 반환한다.

---

## 2. `process.py`

> 데이터 증강 파이프라인의 핵심 처리 로직을 담당하는 모듈.
> 패딩 처리 → BLIP 캡션 생성 → JSONL 메타데이터 저장 → SDXL Img2Img 증강 이미지 생성의 흐름을 수행한다.

---

### 🟠 클래스: `Processing`

> `data.py`의 `RawDataset`과 `PaddingImages`를 **Composition(합성)** 방식으로 내부에 포함하여, 전체 데이터 증강 파이프라인을 실행하는 주 클래스.

#### 인스턴스 변수 (Instance Variables)

| 변수명 | 역할 |
|---|---|
| `self.object` | 처리할 품목 이름 |
| `self.RawDataset` | `RawDataset` 인스턴스. 원본 이미지 로드에 사용 |
| `self.PaddingImages` | `PaddingImages` 인스턴스. 패딩 이미지 경로 관리에 사용 |
| `self.prompt` | SDXL 이미지 생성 시 사용할 텍스트 프롬프트 |

#### 메서드 (Methods)

```python
def __init__(self, dataset_name='MVTEC_AD_2', object_name='can', split='train', prompt='...')
```
> 데이터셋 이름, 품목, 분할 종류, 생성 프롬프트를 받아 `RawDataset`과 `PaddingImages` 객체를 Composition으로 초기화한다.

---

```python
def pad_image(self, idx: int, target_size=1024)
```
> `idx`번째 원본 이미지를 `target_size × target_size` 크기의 정사각형으로 **Letterbox 패딩** 처리하고, 결과를 PNG 파일로 저장한다.
>
> | 지역 변수 | 역할 |
> |---|---|
> | `raw_image` | 원본 이미지 PIL 객체 |
> | `delta_w` | 가로 방향 여백 총량 (`target_size - 원본가로`) |
> | `delta_h` | 세로 방향 여백 총량 (`target_size - 원본세로`) |
> | `padding` | 4방향 여백 픽셀 수를 담은 튜플 `(left, top, right, bottom)` |
> | `padding_image` | 흰색 패딩이 적용된 최종 PIL 이미지 객체 |
> | `padding_image_name` | 저장할 파일명 (`{object}_{idx:03d}.png`) |
> | `image_path` | 패딩 이미지 저장 전체 경로 |

---

```python
def padding_all_images_in_object(self)
```
> 해당 품목의 모든 원본 이미지에 대해 `pad_image()`를 순차적으로 실행하여 전량 패딩 처리하는 배치 함수.

---

```python
def generate_caption(self, idx: int)
```
> BLIP 모델(`blip-image-captioning-large`)을 사용하여 `idx`번째 패딩 이미지에 대한 텍스트 캡션을 생성하고 반환한다. CUDA 환경에서만 동작한다.
>
> | 지역 변수 | 역할 |
> |---|---|
> | `device` | 연산 장치 (`"cuda"`) |
> | `processor` | BLIP 이미지-텍스트 전처리기 |
> | `model` | BLIP 조건부 텍스트 생성 모델 |
> | `padding_image` | 캡션을 생성할 패딩된 PIL 이미지 |
> | `input_image` | 모델 입력용 텐서로 변환된 이미지 |
> | `output_1` | 모델이 생성한 토큰 ID 시퀀스 |
> | `output_2` | 토큰을 디코딩한 원시 캡션 문자열 |
> | `caption` | 품목 이름이 접두사로 붙은 최종 캡션 문자열 |

---

```python
def create_diffusers_metadata(self)
```
> 모든 패딩 이미지에 대해 `generate_caption()`을 실행하고, 파일명-캡션 쌍을 JSONL 형식으로 메타데이터 파일에 저장한다. Diffusion 모델 파인튜닝(LoRA 학습)에 사용된다.
>
> | 지역 변수 | 역할 |
> |---|---|
> | `metadata_path` | 저장할 JSONL 파일의 전체 경로 |
> | `f_jsonl` | 열린 JSONL 파일 핸들 |
> | `image_name` | JSONL에 기록할 이미지 파일명 |
> | `caption` | 해당 이미지의 BLIP 캡션 |
> | `line` | JSONL 한 줄에 해당하는 딕셔너리 `{"file_name": ..., "text": ...}` |

---

```python
def Segmentation(self)
```
> 세그멘테이션 처리를 위해 예약된 메서드. 현재는 `pass`로 비어있으며 추후 구현 예정.

---

```python
def generate_image(self)
```
> **SDXL Img2Img 파이프라인**을 사용하여 패딩 이미지를 기반으로 증강 이미지를 생성하고 저장한다. CUDA 환경에서만 동작한다.
>
> | 지역 변수 | 역할 |
> |---|---|
> | `device` | 연산 장치 (`"cuda"`) |
> | `pipe` | `StableDiffusionXLImg2ImgPipeline` 모델 인스턴스 |
> | `generated_image_path` | 증강 이미지를 저장할 폴더 경로 |
> | `padding_image` | SDXL에 입력할 패딩된 PIL 이미지 |
> | `generated_image` | SDXL이 생성한 증강 PIL 이미지 |
> | `generated_image_name` | 저장할 파일명 (`gen{object}_{idx:03d}.png`) |
> | `image_path` | 증강 이미지 저장 전체 경로 |

---

## 3. `main.py`

> 전체 데이터 증강 파이프라인의 **실행 진입점(Entry Point)**. `Processing` 클래스를 인스턴스화하여 품목별로 증강 이미지를 생성한다.

---

### 🔵 전역 변수 (Module-level Variables)

| 변수명 | 타입 | 역할 |
|---|---|---|
| `dataset_name` | `list[str]` | 지원하는 데이터셋 이름 목록 |
| `MVTEC_AD_1_OBJECTS` | `list[str]` | MVTec AD 1의 15개 품목 이름 목록 (실행 반복용) |
| `MVTEC_AD_2_OBJECTS` | `list[str]` | MVTec AD 2의 8개 품목 이름 목록 (실행 반복용) |
| `PROMPT` | `list[str]` | 이미지 생성에 사용할 프롬프트 목록 (미완성, 추후 추가 예정) |

### 🔵 실행 흐름 변수 (런타임 지역 변수)

| 변수명 | 역할 |
|---|---|
| `object_1` | MVTec AD 1 반복문의 현재 품목 이름 |
| `object_2` | MVTec AD 2 반복문의 현재 품목 이름 |
| `object_name` | 현재 처리 중인 품목 이름 (반복문 내부 임시 변수) |
| `preprocessor` | `Processing` 인스턴스. 품목별 파이프라인 실행 객체 |

> **참고:** `__name__ == "__main__"` 블록 내의 주석 처리된 코드는 `padding_all_images_in_object()` (패딩 전처리)를 먼저 실행하는 이전 단계 코드이며, 현재 활성 코드는 `generate_image()` (증강 이미지 생성)을 실행한다.

---

## 4. `later/sam_segmenter.py`

> **SAM(Segment Anything Model)**을 활용하여 이미지 내 특정 객체의 마스크를 포인트 기반으로 추출하는 모듈. 인페인팅 파이프라인의 전처리 단계로 사용 예정.

---

### 🟠 클래스: `SAMMaskExtractor`

> SAM 모델을 로드하고, 지정된 픽셀 좌표(포인트)를 기준으로 객체 마스크를 추출하여 PNG로 저장하는 클래스.

#### 인스턴스 변수 (Instance Variables)

| 변수명 | 역할 |
|---|---|
| `self.device` | 연산 장치 (`"cuda"` 또는 `"cpu"`) |
| `self.model` | SAM `vit_h` 체크포인트 기반 세그멘테이션 모델 |
| `self.predictor` | `SamPredictor` - 이미지에 마스크를 예측하는 래퍼 객체 |
| `self.input_point` | 마스크 추출 기준 픽셀 좌표 (`[[512, 512]]`, 이미지 중앙) |
| `self.input_label` | 포인트 라벨 (`[1]`: 해당 포인트를 포함하는 마스크 생성) |
| `self.train_target_dir` | 처리할 이미지가 위치한 폴더 경로 |

#### 메서드 (Methods)

```python
def __init__(self)
```
> SAM `vit_h` 모델과 `SamPredictor`를 로드하고, 기준 포인트 좌표(`[512, 512]`)와 라벨을 초기화한다.

---

```python
def extract_mask(self, train_target_dir, output_dir)
```
> 지정된 폴더 내의 모든 PNG 이미지에 대해 SAM 포인트 프롬프트 기반 마스크를 예측하고, 흑백 마스크 이미지로 저장한다.
>
> | 지역 변수 | 역할 |
> |---|---|
> | `img_name` | 반복 중인 이미지 파일명 |
> | `img_path` | 이미지 전체 경로 |
> | `image` | OpenCV로 읽은 후 BGR→RGB 변환된 numpy 배열 |
> | `masks` | SAM이 예측한 마스크 배열 |
> | `scores` | 각 마스크에 대한 신뢰도 점수 |
> | `logits` | SAM 예측 로짓 값 |
> | `mask_img` | 이진 마스크를 0-255 uint8로 변환한 numpy 배열 |
> | `mask_path` | 마스크 이미지 저장 전체 경로 |

---

## 5. `later/sdxl_inpainter.py`

> **SDXL Inpaint 파이프라인**과 품목별 **LoRA 가중치**를 결합하여, 마스크 기반으로 정상 또는 비정상 증강 이미지를 생성하는 모듈.

---

### 🟠 클래스: `SDXLInpaintGenerator`

> SDXL Inpainting 모델을 로드하고, 패딩 이미지와 SAM 마스크를 입력받아 LoRA 가중치가 적용된 인페인팅 증강 이미지를 생성하는 클래스.

#### 인스턴스 변수 (Instance Variables)

| 변수명 | 역할 |
|---|---|
| `self.device` | 연산 장치 (`"cuda"` 또는 `"cpu"`) |
| `self.pipe` | `StableDiffusionXLInpaintPipeline` 인페인팅 파이프라인 객체 |
| `self.category` | 처리 중인 품목 카테고리 (현재 미사용) |
| `self.lora_dir` | LoRA 가중치 파일 경로 |
| `self.input_dir` | 입력 이미지 폴더 경로 (현재 미사용) |
| `self.output_dir` | 생성된 이미지를 저장할 폴더 경로 |
| `self.prompt` | 이미지 생성에 사용할 텍스트 프롬프트 |

#### 메서드 (Methods)

```python
def __init__(self)
```
> `StableDiffusionXLInpaintPipeline`을 `float16` 정밀도로 로드하고 `self.device`로 이동한다.

---

```python
def generate_image(self, lora_dir, padded_dir, mask_dir, output_dir, prompt)
```
> 품목별 LoRA를 파이프라인에 로드한 뒤, 패딩 이미지와 SAM 마스크를 이용해 인페인팅 이미지를 생성하고 저장한다. 생성이 완료되면 다음 품목 학습을 위해 LoRA 가중치를 자동으로 언로드한다.
>
> | 매개변수 | 역할 |
> |---|---|
> | `lora_dir` | 적용할 LoRA 가중치 파일 경로 |
> | `padded_dir` | 패딩 이미지가 저장된 폴더 경로 |
> | `mask_dir` | SAM 마스크 이미지가 저장된 폴더 경로 |
> | `output_dir` | 생성된 증강 이미지를 저장할 폴더 경로 |
> | `prompt` | 인페인팅 시 사용할 텍스트 프롬프트 |
>
> | 지역 변수 | 역할 |
> |---|---|
> | `img_name` | 반복 중인 이미지 파일명 |
> | `padded_img_path` | 패딩 이미지 전체 경로 |
> | `mask_img_path` | 마스크 이미지 전체 경로 |
> | `init_image` | RGB로 변환된 패딩 PIL 이미지 |
> | `mask_image` | RGB로 변환된 마스크 PIL 이미지 |
> | `result_image` | SDXL 인페인팅이 적용된 결과 PIL 이미지 |
> | `result_image_path` | 결과 이미지 저장 전체 경로 (`gen_{img_name}`) |

---

## 🔗 Related Documents

- [ImageSquarePadder 분석](image-square-padder.md) - Pillow 기반 이미지 정사각형 패딩 처리 분석 (process.py의 `pad_image`와 동일 개념)
- [Python Study Dashboard](../../../cs/python/index.md) - 파이썬 전체 학습 대시보드
