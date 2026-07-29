# KKCC 결함 데이터 증강 시스템 (DataAugmentation)

> **산업용 이상 탐지(Anomaly Detection)** 모델 훈련을 위한  
> MVTec AD 기반 **결함 이미지 합성·증강** 파이프라인입니다.

---

## 목차

1. [프로젝트 배경](#1-프로젝트-배경)  
2. [시스템 전체 구조](#2-시스템-전체-구조)  
3. [디렉터리 구조](#3-디렉터리-구조)  
4. [사용하는 데이터셋](#4-사용하는-데이터셋)  
5. [파이프라인 A — 기본 SD 3.5 증강 (main.py)](#5-파이프라인-a--기본-sd-35-증강-mainpy)  
6. [파이프라인 B — 2단계 하이브리드 증강 (hybrid_runner.py)](#6-파이프라인-b--2단계-하이브리드-증강-hybrid_runnerpy)  
   - [Stage 1 : 범용 베이스라인](#61-stage-1--범용-베이스라인)  
   - [Stage 2 : 도메인 스페셜리스트](#62-stage-2--도메인-스페셜리스트)  
7. [모듈별 상세 설명](#7-모듈별-상세-설명)  
   - [data.py](#71-datapy)  
   - [process.py](#72-processpy)  
   - [aug_hybrid/two_stage_hybrid_pipeline/](#73-aug_hybridtwo_stage_hybrid_pipeline)  
   - [hybrid_runner.py](#74-hybrid_runnerpy)  
8. [스페셜리스트 품목별 시뮬레이션 원리](#8-스페셜리스트-품목별-시뮬레이션-원리)  
9. [출력 결과 구조](#9-출력-결과-구조)  
10. [설치 및 환경 설정](#10-설치-및-환경-설정)  
11. [실행 방법](#11-실행-방법)  
12. [코드 흐름 다이어그램](#12-코드-흐름-다이어그램)  
13. [확장 방법](#13-확장-방법)  
14. [자주 묻는 질문 (FAQ)](#14-자주-묻는-질문-faq)  

---

## 1. 프로젝트 배경

공장에서 생산된 제품의 **불량 여부를 자동으로 판별**하는 이상 탐지(Anomaly Detection) AI 모델을 개발하려면, 결함 이미지가 대량으로 필요합니다.  
그러나 현실에서 결함 샘플은 **매우 희소**합니다. 공정 자체가 대부분 정상 제품을 생산하도록 설계돼 있기 때문입니다.

이 프로젝트는 다음 두 가지 방식으로 **결함 이미지를 인공적으로 합성·증강**합니다.

```
| 방식 | 설명 | 파이프라인 |
|------|------|-----------|
| **SD 3.5 기반 이미지 변환** | 정상 이미지를 Stable Diffusion 3.5 로 결함 이미지로 변환 | main.py |
| **2단계 하이브리드 물리·AI 합성** | 물리 시뮬레이션으로 결함 형태를 먼저 만들고 AI로 사실감 추가 | hybrid_runner.py |
```
---

## 2. 시스템 전체 구조

```
[입력 : MVTec AD 1·2 원본 이미지]
         │
         ▼
┌─────────────────────────────────────────────┐
│  파이프라인 A : main.py                       │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ data.py  │──▶│process.py│──▶│생성 이미지│ │
│  │ RawDataset│  │ Padding  │   │  SD 3.5  │ │
│  │ PaddingImg│  │ Caption  │   │  SDXL    │ │
│  └──────────┘  │ Generate │   └──────────┘ │
│                └──────────┘                 │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  파이프라인 B : hybrid_runner.py              │
│                                             │
│  Stage 1 ─ 범용 베이스라인                   │
│  ┌─────────────────────────────────────┐   │
│  │ Stage1UniversalAugmentor            │   │
│  │  · 기하 프리미티브 생성(원, 타원 등) │   │
│  │  · 결함 마스크(GT Mask) 자동 생성   │   │
│  └─────────────────────────────────────┘   │
│         │                                  │
│         ▼                                  │
│  Stage 2 ─ 도메인 스페셜리스트               │
│  ┌────────┬────────┬────────┬──────────┐   │
│  │  can   │  vial  │ sheet  │wallplugs │   │
│  │ (캔)   │(약병)  │(판금)  │(앙카)    │   │
│  │ 좌굴   │균열/파단│스크래치│백화/파손 │   │
│  └────────┴────────┴────────┴──────────┘   │
│         │                                  │
│         ▼                                  │
│  [SUPIR 복원 : 초고해상도 노이즈·질감 복원]  │
└─────────────────────────────────────────────┘
         │
         ▼
[출력 : 합성 결함 이미지 + GT 마스크 쌍]
```

---

## 3. 디렉터리 구조

```
DataAugmentation/
│
├── main.py                          # [진입점 A] SD 3.5 기반 증강 실행
├── hybrid_runner.py                 # [진입점 B] 2단계 하이브리드 파이프라인 실행  ← NEW
│
├── data.py                          # 데이터셋 경로 관리 및 이미지 로더
├── process.py                       # 패딩·캡션·이미지 생성 로직
│
├── aug_hybrid/                      # 하이브리드 파이프라인 패키지  ← NEW
│   ├── __init__.py                  # aug_hybrid 패키지 선언  ← NEW
│   └── two_stage_hybrid_pipeline/   # 2단계 파이프라인 서브 패키지
│       ├── __init__.py              # 패키지 초기화 + 모듈 노출  ← NEW
│       ├── main.py                  # 파이프라인 오케스트레이터
│       ├── stage1_universal.py      # Stage 1 : 범용 증강기
│       ├── stage2_can_specialist.py        # Stage 2 : 캔 스페셜리스트
│       ├── stage2_vial_specialist.py       # Stage 2 : 약병 스페셜리스트
│       ├── stage2_sheetmetal_specialist.py # Stage 2 : 판금 스페셜리스트
│       └── stage2_wallplugs_specialist.py  # Stage 2 : 앙카 스페셜리스트
│
├── docs/                            # 프로젝트 문서
├── later/                           # 추후 구현 예정 모듈 (SAM, SDXL 인페인팅 등)
├── conversation_summary.md          # 프로젝트 대화 요약
└── README.md                        # 현재 파일
```

> ← NEW 표시가 붙은 항목은 기존 코드를 수정하지 않고 새로 추가된 파일입니다.

---

## 4. 사용하는 데이터셋

### MVTec AD 1
공장 생산 품목 15종의 정상·결함 이미지가 포함된 표준 이상 탐지 벤치마크 데이터셋입니다.
```
| 품목 목록 |
|-----------|
| bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper |

- 학습 데이터(	rain/good/): 정상 이미지만 포함
- 테스트 데이터(	est/): 정상 + 결함 이미지 혼합
- 정답 마스크(ground_truth/): 픽셀 단위 결함 위치 레이블
```

### MVTec AD 2
8종의 산업 품목이 추가된 확장 버전입니다.
```
| 품목 목록 |
|-----------|
| can, fabric, fruit_jelly, rice, sheet_metal, vial, wallplugs, walnuts |

- 	est_private 등 다양한 분할(split) 존재
- 이 프로젝트의 **2단계 하이브리드 파이프라인**은 MVTec AD 2의 주요 품목(can, ial, sheet_metal, wallplugs)을 핵심 타겟으로 설계됨
```

### 데이터 경로 설정 (data.py)

```python
root = '/NHNHOME/WORKSPACE/26moel002_ex07/AD/Data'   # 서버 경로 (환경에 맞게 수정)
PATH_TO_MVTEC_AD_1_FOLDER = root + '/OpenDataset/mvtec_ad_1'
PATH_TO_MVTEC_AD_2_FOLDER = root + '/OpenDataset/mvtec_ad_2'
```

> ⚠️ 로컬에서 실행하려면 data.py 상단의  root 경로를 실제 데이터셋 경로로 수정해야 합니다.

---

## 5. 파이프라인 A — 기본 SD 3.5 증강 (main.py)

### 개념

정상 이미지를 **Stable Diffusion 3.5**의 img2img 모드로 변환해 결함처럼 보이는 이미지를 생성합니다.

### 실행 흐름

```
main.py 실행
    │
    ├── 품목 선택 (all 또는 특정 품목명 입력)
    │
    ├── Processing 객체 생성
    │       │
    │       ├── RawDataset   : 원본 데이터셋 이미지 경로 수집
    │       └── PaddingImages: 패딩 이미지 저장 경로 관리
    │
    └── generate_image_SD35() 호출
            │
            ├── 1. 원본 이미지를 1024×1024 로 letterbox 패딩
            ├── 2. SD 3.5 Large (img2img) 에 패딩 이미지 + 프롬프트 입력
            ├── 3. strength=0.3, guidance_scale=4.0, steps=50 으로 생성
            └── 4. gen_<품목>_<번호>.png 로 저장 + batch_metadata.json 기록
```

### 핵심 파라미터
```
| 파라미터 | 값 | 설명 |
|----------|-----|------|
| strength | 0.3 | 원본 이미지 변형 강도 (0.0 = 원본 그대로, 1.0 = 완전히 다른 이미지) |
| guidance_scale | 4.0 | 프롬프트 반영 강도 (SD 3.5 권장값: 3.5~4.5) |
| 
um_inference_steps | 50 | 이미지 생성 반복 횟수 (높을수록 품질↑, 속도↓) |
| seed | 42 | 재현 가능한 결과를 위한 난수 시드 |
| 출력 해상도 | 1024×1024 | |
```

### 사용 모델

- **Stable Diffusion 3.5 Large** (stabilityai/stable-diffusion-3.5-large)  
  → HuggingFace 토큰 필요 (	okens.py 에 토큰 저장)
- **Stable Diffusion XL Refiner** (stabilityai/stable-diffusion-xl-refiner-1.0)  
  → generate_image_SDXL() 메서드로 대체 사용 가능

---

## 6. 파이프라인 B — 2단계 하이브리드 증강 (hybrid_runner.py)

### 개념

**물리 시뮬레이션 → AI 텍스처 합성 → 초고해상도 복원** 3단계 체인으로,  
재료역학 원리에 기반한 사실적인 결함 이미지와 정답 마스크(GT Mask)를 함께 생성합니다.

### 왜 2단계인가?

```
| | Stage 1 (범용) | Stage 2 (스페셜리스트) |
|--|---------------|----------------------|
| 속도 | 빠름 | 느림 (품목 특화 시뮬레이션) |
| 품질 | 보통 | 높음 (물리 기반 변형 반영) |
| 품목 커버리지 | 전 품목 | can, vial, sheet_metal, wallplugs |
| 활용 | 빠른 프로토타이핑, 전 품목 베이스라인 확보 | 고품질 학습 데이터 생성 |
```

### 6.1 Stage 1 : 범용 베이스라인

**파일**: ug_hybrid/two_stage_hybrid_pipeline/stage1_universal.py  
**클래스**: Stage1UniversalAugmentor

어떤 품목에도 적용 가능한 **범용 기하 마스크** 기반 증강입니다.

```
Stage 1 동작 순서
    │
    ├── 1024×1024 캔버스에 타원형 객체 영역(Object Mask) 생성
    │       → cv2.ellipse() 로 중앙에 타원 그림
    │
    ├── 타원 내부 임의 위치에 원형 결함 영역(GT Mask) 배치
    │       → cv2.circle() + cv2.bitwise_and() 로 영역 내 마스크 생성
    │
    ├── Depth 맵 역할의 회색조 합성 이미지 생성
    │       → 마스크 영역은 회색, 결함 부위는 붉은색 (R=200, G=50, B=50)
    │
    └── images/<sample_id>.png  + masks/<sample_id>_mask.png 쌍 저장
```

**출력 경로**:
```
<base_dir>/two_stage_hybrid_output/stage1_universal/<품목>/
    ├── images/   <품목>_s1_0001.png  ...
    └── masks/    <품목>_s1_0001_mask.png  ...
```

### 6.2 Stage 2 : 도메인 스페셜리스트

**파일**: aug_hybrid/two_stage_hybrid_pipeline/stage2_*.py

각 품목의 **물리적 특성**을 수학 공식으로 모델링해 고품질 결함을 생성합니다.  
모든 스페셜리스트는 아래 공통 4단계 구조를 따릅니다.

```
1. 물체 기하 마스크(Object Mask) 생성
        ↓
2. 물리 시뮬레이션으로 결함 위치·형태 결정 (GT Mask 생성)
        ↓
3. 재질 특성(금속 반사, 유리 굴절, 플라스틱 매트) 텍스처 합성
        ↓
4. 초고해상도 업스케일(×2) + 가우시안 노이즈 추가 (SUPIR 효과 시뮬레이션)
```

**출력 경로**:
```
<base_dir>/two_stage_hybrid_output/stage2_specialized/<품목>/
    ├── images/   <품목>_sota_0001.png  ...
    └── masks/    <품목>_sota_0001_mask.png  ...
```

---

## 7. 모듈별 상세 설명

### 7.1 data.py

데이터셋 경로 관리와 이미지 로딩을 담당하는 **데이터 레이어**입니다.

#### RawDataset 클래스
```
| 메서드/속성 | 설명 |
|------------|------|
| __init__(dataset_name, object_name, split) | 데이터셋 종류, 품목명, 분할 종류를 받아 이미지 경로 목록 생성 |
| get_image_path() | glob 패턴으로 해당 분할의 이미지 경로 반환 |
| get_image(idx) | idx 번째 이미지를 PIL Image로 반환 |
| __len__() | 이미지 총 개수 반환 |
| image_paths | 정렬된 이미지 경로 리스트 (property) |
```
MVTec AD 1 과 2의 디렉터리 구조가 다르기 때문에, get_image_path() 내부에서 데이터셋 종류와 split에 따라 glob 패턴을 분기합니다.

#### PaddingImages 클래스

RawDataset을 상속하지만, 패딩 완료된 이미지 폴더를 관리합니다.  
이미지를 직접 로드하는 대신, 패딩 저장 경로를 생성하고 저장된 이미지 목록을 반환합니다.

#### 유틸리티 함수

```python
set_metadata_path(object_name)         # BLIP 캡션 메타데이터(.jsonl) 저장 경로 반환
set_generated_image_path(model_name, object_name)  # 생성 이미지 저장 경로 생성 및 반환
```

---

### 7.2 process.py

실제 **전처리·캡션 생성·이미지 합성** 로직을 담당하는 **처리 레이어**입니다.

#### Processing 클래스

RawDataset과 PaddingImages를 내부에 포함(composition)하는 고수준 클래스입니다.
```
| 메서드 | 설명 |
|--------|------|
| pad_image(idx) | 지정 인덱스 이미지를 letterbox 패딩해 1024×1024 로 저장 |
| padding_all_images_in_object() | 품목 내 전체 이미지를 일괄 패딩 |
| generate_caption(idx) | BLIP 모델로 패딩 이미지에서 캡션 텍스트 생성 |
| create_diffusers_metadata() | 캡션을 JSONL 형식으로 저장 (파인튜닝 용) |
| generate_image_SDXL() | SDXL Refiner로 결함 이미지 생성 |
| generate_image_SD35() | **SD 3.5 Large**로 결함 이미지 생성 (주 사용) |
```
#### Letterbox 패딩이란?

원본 이미지의 **가로·세로 비율을 그대로 유지**하면서 1024×1024 정사각형으로 맞추는 기법입니다.  
빈 공간은 흰색으로 채웁니다.

```
원본 (640×480)                패딩 후 (1024×1024)
┌────────────────┐            ┌──────────────────────┐
│                │            │  (흰색 여백 192px)   │
│   실제 이미지  │  →패딩→   │ ┌──────────────────┐ │
│                │            │ │   실제 이미지    │ │
└────────────────┘            │ └──────────────────┘ │
                              │  (흰색 여백 192px)   │
                              └──────────────────────┘
```

---

### 7.3 ug_hybrid/two_stage_hybrid_pipeline/

2단계 하이브리드 파이프라인의 **핵심 패키지**입니다.

#### __init__.py — 패키지 초기화

일반적인 rom .module import Class 방식 대신 **importlib** 를 사용합니다.

이유: 내부 스크립트(main.py, stage1_universal.py 등)가 서로를  
from stage1_universal import ... 처럼 **절대 이름으로 import**합니다.  
이를 패키지 내부에서 상대 import로 바꾸면 기존 코드를 수정해야 하므로,  
__init__.py에서 해당 디렉터리를 sys.path에 등록한 뒤 importlib로 로드합니다.

```python
# __init__.py 핵심 동작
sys.path.insert(0, str(_PIPELINE_DIR))    # 절대 import가 동작하도록 경로 등록
_stage1 = _load_module("stage1_universal") # importlib로 파일 직접 로드
Stage1UniversalAugmentor = _stage1.Stage1UniversalAugmentor  # 패키지 레벨로 노출
```

#### main.py — 오케스트레이터


un_two_stage_hybrid_pipeline(targets, num_stage1, num_stage2, base_dir) 함수가 핵심입니다.

1. Stage 1: 지정된 모든 품목에 대해 Stage1UniversalAugmentor.run_baseline_generation() 호출
2. Stage 2: 품목명에 따라 해당 스페셜리스트 클래스를 선택해 
un_specialized_generation() 호출

#### 스페셜리스트 클래스 공통 인터페이스

```python
class Stage2XxxSpecialist:
    def __init__(self, output_base_dir: str):
        # 출력 폴더 생성 (images/, masks/)

    def run_specialized_generation(self, count: int = 2):
        # count 개수만큼 (이미지, 마스크) 쌍 생성
```

---

### 7.4 hybrid_runner.py

기존 main.py를 수정하지 않고 하이브리드 파이프라인을 실행할 수 있는 **독립 진입점**입니다.

```
hybrid_runner.py
    │
    ├── sys.path 에 프로젝트 루트 등록
    ├── aug_hybrid.two_stage_hybrid_pipeline 패키지 import
    │
    ├── run_stage1_only()   Stage 1만 실행
    ├── run_stage2_only()   Stage 2만 실행
    └── main()              argparse로 CLI 인자를 받아 적절한 함수 호출
```

---

## 8. 스페셜리스트 품목별 시뮬레이션 원리

### 🥫 Can (알루미늄 캔) — stage2_can_specialist.py

**물리 모델**: 얇은 원통 쉘의 **외부 압력에 의한 좌굴(Buckling)**

`python
# 원통 표면의 Z값 계산 (실린더 기하)
z_cylinder = np.sqrt(np.maximum(0, 0.70 - xx**2))

# 가우시안 함수로 덴트(Dent) 변형 시뮬레이션
dent_deform = 0.35 * np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2 * 0.3**2))

# 변형량이 임계값(0.06)을 초과한 영역 = 결함 마스크
gt_mask = (dent_deform > 0.06).astype(np.uint8) * 255
`

| 시뮬레이션 요소 | 설명 |
|----------------|------|
| 원통 형상 | xx² ≤ 0.70 조건으로 캔 측면 영역 정의 |
| 덴트 변형 | 2D 가우시안 함수로 국소 압입 표현 |
| 금속 반사 | Z값에 비례한 명도 + 파랗게 편향된 RGB |
| 결함 음영 | 결함 부위의 픽셀값 45%로 감소 + 노이즈 |

---

### 💊 Vial (유리 약병) — stage2_vial_specialist.py

**물리 모델**: 유리의 **취성 파괴(Brittle Fracture)** — 방사형 균열선

`python
# 임의 방향·길이의 균열 가지(Branch) 여러 개 생성
for _ in range(num_branches):          # 4~8개
    angle = np.random.uniform(0, 2*pi)
    length = np.random.randint(60, 160)
    cv2.line(gt_mask, crack_center, end_pt, 255, thickness=3~8)
`

| 시뮬레이션 요소 | 설명 |
|----------------|------|
| 약병 형상 | 직사각형으로 단순화한 유리병 바디 |
| 균열 패턴 | 중심점에서 뻗어나가는 방사형 선분 |
| 유리 투명감 | 밝은 회색 베이스(R=220, G=235, B=245) |
| 굴절 표현 | 균열선 픽셀을 청록색(R=40, G=210, B=245)으로 강조 |

---

### 🔩 Sheet Metal (판금) — stage2_sheetmetal_specialist.py

**물리 모델**: 평면 금속판의 **인장/전단 변형 + 압연 마크(Rolling Mark)**

`python
# 직선형 균열 시뮬레이션
cv2.line(gt_mask, pt1, pt2, 255, thickness=8~20)

# 압연 결 텍스처 (사인파)
rolling_lines = (np.sin(np.linspace(0, 50, h)) * 15).astype(np.uint8)
metal_bg = np.clip(metal_bg + rolling_lines[:, None], 0, 255)
`

| 시뮬레이션 요소 | 설명 |
|----------------|------|
| 기하 | 전체 프레임이 금속판 (마스크 = 전체 흰색) |
| 균열선 | 임의 두께(8~20px)의 직선형 균열 + 덴트 원형 영역 |
| 압연 마크 | 수직 방향 사인파로 롤링 라인 텍스처 표현 |
| 결함 음영 | 결함 픽셀값 35%로 감소 (깊은 그늘 표현) |

---

### 🔌 Wallplugs (플라스틱 앙카) — stage2_wallplugs_specialist.py

**물리 모델**: 사출 성형 플라스틱의 **팁 부러짐 + 응력 백화(Stress Whitening)**

`python
# 앙카 형상: 직사각형 바디 + 양쪽 날개(타원)
cv2.rectangle(plug_mask, ...)
cv2.ellipse(plug_mask, (w*0.35, h*0.4), (50, 120), 30, ...)  # 왼쪽 날개
cv2.ellipse(plug_mask, (w*0.65, h*0.4), (50, 120), -30, ...) # 오른쪽 날개

# 팁 하단 75% 이후 = 부러진 영역
```
gt_mask[int(h*0.75):, int(w*0.35):int(w*0.65)] = 255
```

```
| 시뮬레이션 요소 | 설명 |
|----------------|------|
| 앙카 형상 | 직사각형 바디 + 기울어진 타원 2개 (날개) |
| 파손 위치 | 전체 높이 75% 이하 = 팁 부러짐 |
| 백화 현상 | 파손 부위를 흰색(R=255, G=250, B=240)으로 표현 |
| 매트 플라스틱 | 밝은 회색 베이스, 낮은 노이즈(σ=1.8) |
```
---

## 9. 출력 결과 구조

### 파이프라인 A 출력 (SD 3.5)

```
<root>/GenerateDataset/
└── SD35/
    └── <품목>/
        ├── gen_<품목>_000.png
        ├── gen_<품목>_001.png
        ├── ...
        └── <품목>_batch_metadata.json   ← 각 이미지의 입력 경로·seed 기록
```

### 파이프라인 B 출력 (하이브리드)

```
D:\KKCC_Project\two_stage_hybrid_output\
│
├── stage1_universal/
│   ├── can/
│   │   ├── images/  can_s1_0001.png  can_s1_0002.png  ...
│   │   └── masks/   can_s1_0001_mask.png  ...
│   ├── vial/
│   └── ... (지정한 모든 품목)
│
└── stage2_specialized/
    ├── can/
    │   ├── images/  can_sota_0001.png  ...
    │   └── masks/   can_sota_0001_mask.png  ...
    ├── vial/
    ├── sheet_metal/
    └── wallplugs/
```

---

## 10. 설치 및 환경 설정

### 요구 사항

- Python 3.10+
- CUDA 지원 GPU (VRAM 16GB 이상 권장 — SD 3.5 Large fp16 기준)
- HuggingFace 계정 및 액세스 토큰 (SD 3.5 모델 다운로드용)

### 패키지 설치

```bash
# 가상환경 생성 (권장)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 의존성 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate pillow opencv-python huggingface_hub
```

### HuggingFace 토큰 설정

DataAugmentation/tokens.py 파일을 생성하고 아래 내용을 입력합니다.

```
python
# tokens.py  (절대 Git에 커밋하지 마세요!)
token = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

> ⚠️tokens.py 는 반드시 .gitignore 에 등록해야 합니다.

### 데이터 경로 수정

data.py 상단의 root 변수를 실제 데이터셋 경로로 변경합니다.

```python
# data.py
root = 'D:/KKCC_Project'           # 예: 로컬 Windows 환경
# root = '/home/user/datasets'     # 예: Linux 서버 환경
```

---

## 11. 실행 방법

### 파이프라인 A — SD 3.5 기반 증강

`
bash
python main.py
`

실행 후 프롬프트가 나타나면 품목명을 입력합니다.

`
이미지 증강할 품목을 선택하시오.(all = 모든 품목 / object_name = 해당 품목
> all          # 모든 품목 실행
> can          # can 품목만 실행
`

---

### 파이프라인 B — 2단계 하이브리드 증강

```bash
# 도움말 확인
```

```
python hybrid_runner.py --help
```

# 기본 실행 (can, vial, sheet_metal, wallplugs / 전체 2단계 / 샘플 2개)
```
python hybrid_runner.py
```

# Stage 1만 실행 (전 품목)
```
python hybrid_runner.py --stage 1 --targets all
```

# Stage 2만, 특정 품목
```
python hybrid_runner.py --stage 2 --targets can vial
```

# 샘플 수 조정
```
python hybrid_runner.py --num_stage1 10 --num_stage2 5 --targets can sheet_metal
```

# 출력 경로 변경
```
python hybrid_runner.py --base_dir "E:\MyProject" --targets can
```

# 전 단계, 전 품목, 각 10개 샘플
```
python hybrid_runner.py --stage all --targets all --num_stage1 10 --num_stage2 10
```

#### CLI 옵션 전체 목록

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| --stage | ll | 실행 단계 (1, 2, ll) |
| --targets | can vial sheet_metal wallplugs | 대상 품목 목록 (공백 구분, ll 입력 시 전 품목) |
| --num_stage1 | 2 | Stage 1 품목당 생성 샘플 수 |
| --num_stage2 | 2 | Stage 2 품목당 생성 샘플 수 |
| --base_dir | D:\KKCC_Project | 출력 루트 디렉터리 |

---

## 12. 코드 흐름 다이어그램

### 파이프라인 A (main.py)

```
main.py
  └─ Processing.__init__(dataset, object, prompt)
        ├─ RawDataset(dataset, object, split)
        │    └─ glob(get_image_path()) → self._image_paths
        └─ PaddingImages(object)
             └─ 패딩 이미지 저장 경로 준비

  └─ preprocessing.generate_image_SD35()
        ├─ HuggingFace login(token)
        ├─ StableDiffusion3Img2ImgPipeline.from_pretrained(...)
        └─ for idx in range(len(dataset)):
              ├─ PaddingImages.get_image(idx)  → PIL Image
              ├─ pipe(prompt, image, strength=0.3, ...) → generated_image
              ├─ generated_image.save(gen_<object>_<idx>.png)
              └─ batch_metadata[history].append({input, output, seed})
```

### 파이프라인 B (hybrid_runner.py)

``
hybrid_runner.py
  └─ main()
        ├─ parse_args() → args (stage, targets, num_stage1, num_stage2, base_dir)
        │
        ├─ [stage == "1"] run_stage1_only(targets, num_stage1, base_dir)
        │     └─ for cat in targets:
        │           Stage1UniversalAugmentor(cat, out_base)
        │             └─ run_baseline_generation(num_stage1)
        │                   ├─ 타원 object mask 생성
        │                   ├─ 원형 defect mask 생성 (bitwise_and로 영역 한정)
        │                   ├─ 회색조 합성 이미지 생성
        │                   └─ images/<id>.png + masks/<id>_mask.png 저장
        │
        ├─ [stage == "2"] run_stage2_only(targets, num_stage2, base_dir)
        │     └─ for cat in targets:
        │           Stage2XxxSpecialist(out_base)
        │             └─ run_specialized_generation(num_stage2)
        │                   ├─ 1. 물체 기하 마스크 생성
        │                   ├─ 2. 물리 시뮬레이션 → GT Mask 생성
        │                   ├─ 3. 재질 텍스처 합성 이미지 생성
        │                   ├─ 4. ×2 업스케일 + 가우시안 노이즈 (SUPIR 효과)
        │                   └─ images/<id>.png + masks/<id>_mask.png 저장
        │
        └─ [stage == "all"] run_two_stage_hybrid_pipeline(...)
              ├─ Stage 1 전체 실행
              └─ Stage 2 전체 실행
``

---

## 13. 확장 방법

### 새 품목 스페셜리스트 추가하기

1. aug_hybrid/two_stage_hybrid_pipeline/ 에 stage2_<품목>_specialist.py 파일 생성

`python
# stage2_new_item_specialist.py
import os, cv2, numpy as np
from PIL import Image

class Stage2NewItemSpecialist:
    def __init__(self, output_base_dir: str):
        self.output_dir = os.path.join(output_base_dir, "stage2_specialized", "new_item")
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "masks"), exist_ok=True)

    def run_specialized_generation(self, count: int = 2):
        for i in range(count):
            # 1. 기하 마스크 생성
            # 2. 결함 마스크 생성 (물리 시뮬레이션)
            # 3. 텍스처 합성
            # 4. 업스케일 + 노이즈
            pass
`

2. aug_hybrid/two_stage_hybrid_pipeline/__init__.py 에 import 추가

`python
_stage2n = _load_module("stage2_new_item_specialist")
Stage2NewItemSpecialist = _stage2n.Stage2NewItemSpecialist
`

3. aug_hybrid/two_stage_hybrid_pipeline/main.py 의 
un_two_stage_hybrid_pipeline() 에 분기 추가

`python
elif cat_lower == "new_item":
    Stage2NewItemSpecialist(out_base).run_specialized_generation(num_stage2)
`

4. hybrid_runner.py 의 
un_stage2_only() 에도 동일하게 분기 추가

---

## 14. 자주 묻는 질문 (FAQ)

**Q. GPU가 없으면 실행이 안 되나요?**  
A. 파이프라인 A (main.py, process.py)는 	orch.cuda.is_available() 가 False이면 실행하지 않습니다.  
파이프라인 B (hybrid_runner.py) 의 Stage 1·2는 NumPy/OpenCV 기반이라 **CPU만 있어도 실행 가능**합니다.

**Q. 	okens.py 파일이 없어서 오류가 나요.**  
A. DataAugmentation/tokens.py 파일을 직접 만들고 	oken = "hf_..." 를 추가하세요.  
파이프라인 B는 토큰이 필요 없습니다.

**Q. Stage 2에서 can, vial, sheet_metal, wallplugs 외 품목을 지정하면 어떻게 되나요?**  
A. Stage 1 결과만 사용하도록 경고 메시지가 출력되고 넘어갑니다. 새 스페셜리스트를 직접 추가해서 확장할 수 있습니다.

**Q. data.py의 경로를 바꿔야 하는데 다른 사람도 쓰는 코드라 수정하기 어려워요.**  
A. 환경변수를 활용하거나, data.py 상단에 os.environ.get('MVTEC_ROOT', '/기본경로') 형태로 수정하면 됩니다.

**Q. 생성된 이미지 품질이 낮아요.**  
A. 파이프라인 B의 Stage 2는 실제 SD·SUPIR 모델 없이 수학적 근사(NumPy + OpenCV)로 구현돼 있습니다.  
실제 고품질 출력을 원하면 각 스페셜리스트에서 SD 3.5 + ControlNet 연동 코드를 추가해야 합니다.

---

*이 문서는 2026-07-29 기준으로 작성되었습니다.*
