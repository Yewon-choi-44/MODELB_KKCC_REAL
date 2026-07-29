
# KKCC - model B 전용 MVTec AD 2 불량 데이터 증강 파이프라인

## 프로젝트 개요 (Project Summary)

본 프로젝트는 **KKCC 팀**이 제조업 이상 탐지(Anomaly Detection) 분야의 **'model B' 성능을 최적화하고 극대화하기 위해 개발한 전용 불량 데이터 증강 파이프라인**입니다.

산업 현장에서 AI를 활용한 불량 검출 시 가장 큰 걸림돌은 정상 제품에 비해 불량품 데이터가 극도로 부족한 '데이터 희소성 및 불균형' 문제입니다. 




**model B**는 학습 효율을 완벽하게 서포트하기 위해,

* 생성형 AI인 Stable Diffusion 3.5(SD3.5)를 기반으로 형태를 정밀 제어하는 **ControlNet**
* 모델 전체 파라미터를 직관적으로 학습시키는 **풀 파인튜닝(Full Fine-Tuning)**
* 미세 질감을 현미경 수준으로 다듬는 **SUPIR** 기술을 결합한 **조립식 모듈형** 파이프라인을 구축했습니다.

본 시스템을 통해 원본 불량의 구조적 특징을 완벽히 유지하면서도 대량의 고품질 불량 데이터를 생성하여 model B의 이상 탐지 정확도를 혁신적으로 향상시키고자 합니다.

---

## ■ Model B 파이프라인 요약

> **Stable Diffusion 3.5 Large** + **Full Fine-Tuning** + **ControlNet (Canny + Depth)** + **SUPIR**

* **Stable Diffusion 3.5 Large**: MMDiT(Multimodal Diffusion Transformer) 구조를 활용한 차세대 이미지 생성 모델
* **Full Fine-Tuning**: SD3.5 Large 모델의 전체 파라미터를 불량 데이터셋에 직접 학습시켜 표현력 극대화
* **ControlNet (Canny + Depth)**: Canny Edge로 경계선/윤곽선 위치를 고정하고, Depth Map으로 입체적 깊이와 원근감을 유지하여 형태 왜곡 방지
* **SUPIR**: 원본의 미세한 스크래치, 균열, 질감까지 초고해상도로 복원

---

## ■ 목차

1. **프로젝트 개요 (Introduction)**: SD3.5, ControlNet, Full Fine-Tuning, SUPIR를 활용한 MVTec AD 2 불량 데이터 증강 핵심 아키텍처 소개
2. **프로젝트 폴더 구조 (Project Directory Structure)**: 모듈별 분할 파일과 역할을 트리 구조로 시각화
3. **파이프라인 상세 단계 (Pipeline Stages)**: 1단계 전처리부터 6단계 SUPIR 복원까지의 모듈별 구체적 역할 설명
4. **핵심 기술 용어 사전 (Glossary for Beginners)**: 공식 정의와 비전공자용 비유 풀이로 구성한 용어 설명
5. **공식 정보 출처 및 관련 자료 링크**: 참고 데이터셋 및 공식 오픈소스 라이브러리 출처 주소

---

## ■ 1. 프로젝트 개요

* **명칭**: MVTec AD 2 불량 데이터 증강 파이프라인 (Defect Data Augmentation Pipeline)
* **목적**: 제조업 이상 탐지(Anomaly Detection) 분야의 고질적인 문제인 **'부족한 불량 데이터(Scarcity of Defect Data)'** 문제를 해결하기 위해 고안된 고급 생성형 AI 기반 데이터 증강 파이프라인입니다.
* **특징**: **Stable Diffusion 3.5(SD3.5)** 거대 화가 모델을 기반으로, 형태를 엄격하게 제어하는 **ControlNet**, **풀 파인튜닝**, 그리고 미세한 균열과 질감을 초고해상도로 복원하는 **SUPIR** 기술을 모듈화하여 유기적으로 결합했습니다.

---

## ■ 2. 프로젝트 폴더 구조 (Project Directory Structure)

본 프로젝트는 유지보수와 기능별 재사용성을 극대화하기 위해 조립식 부품처럼 코드를 쪼개어 관리하는 **모듈화(Modularization)** 설계를 채택했습니다.

```text
ModelB_KKCC_REAL/
├── README.md        # 프로젝트 소개 및 가이드라인 문서
├── preprocessor.py # [1단계] 이미지 정방형 패딩 및 규격화 모듈
├── extractor.py    # [2, 3단계] Canny 외곽선 및 Depth 입체 구조 추출 모듈
├── trainer.py      # [4단계] SD3.5 모델 풀 파인튜닝(Full Fine-Tuning) 모듈
├── augmentor.py    # [5단계] 다중 조건 가이드 기반 ControlNet 증강 생성 모듈
├── restorer.py     # [6단계] SUPIR 기반 고화질 미세 스크래치 질감 복원 모듈
└── main.py         # 모든 모듈 부품을 조립하고 총괄 제어하는 메인 관제탑

```

---

## ■ 3. 파이프라인 상세 단계 (Pipeline Stages)

#### (1) 이미지 규격화 및 전처리 (`preprocessor.py`)

* **기능**: 카테고리별 불량 이미지의 가로세로 비율을 유지하면서 $1024 \times 1024$ 정방형 규격으로 맞춥니다.
* **상세**: 이미지가 축소되거나 늘어나 왜곡되는 것을 방지하기 위해, 원본 비율을 지킨 채 크기를 조정한 후 남는 빈 공간(여백)을 깨끗한 하얀색 도화지로 채웁니다.

#### (2) Canny 외곽선 추출 (`extractor.py` - `CannyEdgeExtractor`)

* **기능**: 이미지 한 장마다 Canny 알고리즘을 적용하여 불량품의 경계선과 윤곽선의 정확한 위치 좌표를 지정합니다.
* **상세**: 인공지능 화가가 그림을 그릴 때 뼈대가 될 '투명 외곽선 밑그림 도안'을 만드는 과정입니다.

#### (3) Depth Map 입체 구조 추출 (`extractor.py` - `DepthMapExtractor`)

* **기능**: 이미지 한 장마다 원근감과 깊이(Depth) 지도를 분석하여, 생성 시 발생할 수 있는 이미지 평면화 및 형태 왜곡을 방지합니다.
* **상세**: 물체의 입체적인 굴곡과 깊이를 흑백 명암으로 기록하여, 인공지능 화가가 불량 데이터의 볼륨감을 정확히 인지하도록 돕습니다.

#### (4) SD3.5 모델 풀 파인튜닝 (`trainer.py` - `ModelTrainer`)

* **기능**: MVTec AD 2 불량 데이터셋의 고유한 질감과 결함 형태를 인공지능에 각인시키기 위해 SD3.5 Large 모델의 전체 파라미터를 학습시킵니다.
* **상세**: 일부 가중치만 고치는 경량 학습 대신 AI 화가의 뇌 전체를 산업 불량 데이터에 맞게 재정비하여, 도메인 특화된 고품질 불량 이미지를 정밀하게 그릴 수 있도록 만드는 과정입니다.

#### (5) ControlNet 기반 조건부 증강 이미지 생성 (`augmentor.py`)

* **기능**: 학습된 모델 가중치와 앞서 추출한 두 가지 가이드(Canny 외곽선 + Depth 입체 지도)를 융합하여 정밀 제어된 불량 증강 이미지를 생성합니다.
* **상세**: 텍스트 명령어로 불량을 묘사하면서 동시에 2단계의 밑그림선과 3단계의 입체 도안을 결합하여, 원본의 구조를 완벽히 유지한 채 완전히 새로운 고품질 불량 이미지를 찍어냅니다.

#### (6) SUPIR 고해상도 질감 및 스크래치 복원 (`restorer.py`)

* **기능**: 생성된 증강 이미지의 미세한 스크래치, 미세 균열, 재질의 텍스처(질감)까지 실제 현미경으로 보듯 고해상도로 정밀 복원하여 최종 증강 데이터를 완성합니다.
* **상세**: 흐릿하거나 픽셀이 깨진 부분을 실제 산업용 부품처럼 리얼하게 다듬어주는 '인공지능 복원 돋보기' 단계를 거쳐 데이터의 최종 품질을 극대화합니다.

---

## ■ 4. 핵심 기술 용어 사전 (Glossary for Beginners)

#### MVTec AD 2 (MVTec Anomaly Detection 2)

* **공식 정의**: 제조업 분야의 비지도 학습 기반 이상 탐지 모델 성능을 검증하기 위해 전 세계적으로 통용되는 산업용 제품 비전 검사 표준 벤치마크 데이터셋입니다.
* **비전공자용 풀이**: AI가 공장의 불량품을 얼마나 잘 찾아내는지 시험해 보기 위해 전 세계 과학자들이 공통으로 풀어보는 '인공지능 전용 불량품 문제집'입니다.

#### SD3.5 (Stable Diffusion 3.5)

* **공식 정의**: MMDiT(Multimodal Diffusion Transformer) 아키텍처를 기반으로 텍스트 입력값에 부합하는 고화질 이미지를 생성하는 최신 거대 생성형 AI 모델입니다.
* **비전공자용 풀이**: 사람이 글로 "금속 표면의 기름 유출 불량"이라고 입력하면, 그에 맞는 사실적인 그림을 순식간에 그려내는 '천재 AI 화가'입니다.

#### ControlNet (컨트롤넷)

* **공식 정의**: 사전 학습된 대형 확산 모델에 스케치, 깊이 정보 등의 공간적 조건(Spatial Conditioning)을 추가하여 이미지의 구조적 형태를 미세 제어하는 신경망 구조입니다.
* **비전공자용 풀이**: AI 화가에게 그림을 그리라고 할 때, "이 밑그림 스케치와 원근감을 절대로 벗어나지 말고 그려!"라고 틀을 씌워주는 '투명 도안 가이드라인'입니다.

#### Full Fine-Tuning (풀 파인튜닝)

* **공식 정의**: 사전 학습된 인공신경망의 일부층만 수정하는 매개변수 효율적 기법(PEFT)과 달리, 모델의 전체 가중치(Parameters)를 타겟 데이터셋에 맞추어 새로 업데이트하는 재학습 방식입니다.
* **비전공자용 풀이**: AI 화가의 뇌 속 메모리 전체를 완전히 새로 업그레이드하여, 특정한 불량 그림을 아주 전문적이고 사실적으로 그리도록 만드는 '전면 재학습 방식'입니다.

#### DoRA (Weight-Decomposed Low-Rank Adaptation)

* **공식 정의**: 거대 모델의 가중치를 크기(Magnitude)와 방향(Direction) 성분으로 분해한 후, 낮은 순위(Low-rank) 행렬만을 미세조정하여 높은 학습 안정성을 확보한 매개변수 효율적 미세조정(PEFT) 기술입니다.
* **비전공자용 풀이**: 거대한 AI 화가의 뇌 전체를 뜯어고치는 대신, 지식을 '크기'와 '방향'으로 쪼개어 컴퓨터 메모리를 아주 적게 쓰면서도 핵심 요점만 빠르게 학습시키는 '스마트 가중치 과외' 방식입니다.

#### SUPIR (Photo-Realistic Image Restoration)

* **공식 정의**: 확산 모델의 강력한 사전 지식과 텍스트 프롬프트를 결합하여, 훼손되거나 해상도가 낮은 이미지의 세부 텍스처와 고주파 성분을 사실적으로 복원하는 초고해상도(Super-Resolution) 알고리즘입니다.
* **비전공자용 풀이**: AI 화가가 대략적으로 그려낸 그림 위에 실제 물체 특유의 거친 느낌, 아주 미세한 실금이나 균열까지 실제 사진처럼 생생하게 코팅하고 다듬어주는 '최종 화질 복원 도구'입니다.

---

## ■ 5. 공식 정보 출처 및 관련 자료 링크

본 파이프라인 설계 및 검증, 공식 정의 작성을 위해 참고한 공식 저장소와 사이트 주소입니다.

* **MVTec AD 공식 데이터셋 다운로드**: [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)
* **Kaggle MVTec AD 데이터셋 공유소**: [https://www.kaggle.com/datasets/vipin20/mvtec-ad](https://www.kaggle.com/datasets/vipin20/mvtec-ad)
* **Hugging Face Stable Diffusion 3.5 Large**: [https://huggingface.co/stabilityai/stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
* **Hugging Face Diffusers ControlNet 개발 문서**: [https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet](https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet)
* **Hugging Face PEFT (DoRA / LoRA 기술 명세)**: [https://huggingface.co/docs/peft/index](https://huggingface.co/docs/peft/index)
* **SUPIR 오픈소스 오리지널 코드 저장소 (GitHub)**: [https://github.com/Fanghua-Yu/SUPIR](https://github.com/Fanghua-Yu/SUPIR)