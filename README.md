readme_content = """# MVTec AD 2 불량 데이터 증강 파이프라인 (Defect Data Augmentation Pipeline)

본 프로젝트는 제조업 이상 탐지(Anomaly Detection) 분야의 고질적인 문제인 **'부족한 불량 데이터(Scarcity of Defect Data)'** 문제를 해결하기 위해 고안된 고급 생성형 AI 기반 데이터 증강 파이프라인입니다. 

**Stable Diffusion 3.5(SD3.5)** 거대 화가 모델을 기반으로, 형태를 엄격하게 제어하는 **ControlNet**, 적은 메모리로 핵심 특성을 빠르게 과외받는 **DoRA**, 그리고 미세한 균열과 질감을 초고해상도로 복원하는 **SUPIR** 기술을 모듈화하여 유기적으로 결합했습니다.

---
문서 내부에 포함된 주요 구성 항목은 다음과 같습니다.

1. **프로젝트 개요 (Introduction)**: SD3.5, ControlNet, DoRA, SUPIR를 활용하여 MVTec AD 2 데이터셋의 불량 데이터를 증강해야 하는 이유와 핵심 아키텍처를 소개합니다.
2. **프로젝트 폴더 구조 (Project Directory Structure)**: 이전 답변에서 모듈별로 분할했던 6개의 파일 위치와 각각의 역할을 트리 구조 문법(`text`) 코드로 시각화하여 포함했습니다.
3. **파이프라인 상세 단계 (Pipeline Stages)**: 1단계 전처리부터 6단계 SUPIR 복원까지 각 모듈 파일명이 어떤 기능을 수행하는지 정제된 텍스트로 설명합니다.
4. **핵심 기술 용어 사전 (Glossary for Beginners)**: 어렵게 느껴질 수 있는 대형 인공지능 기술 명칭들을 공식 정의와 비전공자용 비유 풀이로 나누어 완벽하게 정리했습니다.
5. **공식 정보 출처 및 관련 자료 링크**: 데이터셋 다운로드 경로와 각 오픈소스 AI 라이브러리(Hugging Face, GitHub)의 공식 웹사이트 주소를 하단에 리스트로 명시했습니다.


---
🚀 파이프라인 상세 단계 (Pipeline Stages)
1. 이미지 규격화 및 전처리 (preprocessor.py)
기능: 카테고리별 불량 이미지의 가로세로 비율을 유지하면서 1024x1024 정방형 규격으로 맞춥니다.

상세: 이미지가 축소되거나 늘어나 왜곡되는 것을 방지하기 위해, 원본 비율을 지킨 채 크기를 조정한 후 남는 빈 공간(여백)을 깨끗한 하얀색 도화지로 채웁니다.

2. Canny 외곽선 추출 (extractor.py - CannyEdgeExtractor)
기능: 이미지 한 장마다 Canny 알고리즘을 적용하여 불량품의 경계선과 윤곽선의 정확한 위치 좌표를 지정합니다.

상세: 인공지능 화가가 그림을 그릴 때 뼈대가 될 '투명 외곽선 밑그림 도안'을 만드는 과정입니다.

3. Depth Map 입체 구조 추출 (extractor.py - DepthMapExtractor)
기능: 이미지 한 장마다 원근감과 깊이(Depth) 지도를 분석하여, 생성 시 발생할 수 있는 이미지 평면화 및 형태 왜곡을 방지합니다.

상세: 물체의 입체적인 굴곡과 깊이를 흑백 명암으로 기록하여, 인공지능 화가가 불량 데이터의 볼륨감을 정확히 인지하도록 돕습니다.

4. DoRA 불량 특성 학습 (dora_trainer.py)
기능: 적은 양의 Train 데이터셋을 활용해 컴퓨터 자원을 최소화하면서 불량의 고유 패턴만 빠르게 학습하는 DoRA 가중치(dora_weight)를 생성합니다.

상세: 이미 똑똑한 AI 화가에게 "이 공장의 제품 스크래치는 이렇게 생겼어"라고 요점만 짚어 가르치는 '효율적인 가중치 미세조정 과외 방식'입니다. 가중치의 크기와 방향을 분해하여 학습 안정성이 높습니다.

5. ControlNet 기반 조건부 증강 이미지 생성 (augmentor.py)
기능: 학습된 DoRA 가중치와 앞서 추출한 두 가지 가이드(Canny 외곽선 + Depth 입체 지도)를 융합하여 정밀 제어된 불량 증강 이미지를 생성합니다.

상세: 텍스트 명령어로 불량을 묘사하면서 동시에 2단계의 밑그림선과 3단계의 입체 도안을 결합하여, 원본의 구조를 완벽히 유지한 채 완전히 새로운 고품질 불량 이미지를 찍어냅니다.

6. SUPIR 고해상도 질감 및 스크래치 복원 (restorer.py)
기능: 생성된 증강 이미지의 미세한 스크래치, 미세 균열, 재질의 텍스처(질감)까지 실제 현미경으로 보듯 고해상도로 정밀 복원하여 최종 증강 데이터를 완성합니다.

상세: 흐릿하거나 픽셀이 깨진 부분을 실제 산업용 부품처럼 리얼하게 다듬어주는 '인공지능 복원 돋보기' 단계를 거쳐 데이터의 최종 품질을 극대화합니다.

📘 핵심 기술 용어 사전 (Glossary for Beginners)
MVTec AD 2 (MVTec Anomaly Detection 2)

공식 정의: 제조업 분야의 비지도 학습 기반 이상 탐지 모델 성능을 검증하기 위해 전 세계적으로 통용되는 산업용 제품 비전 검사 표준 벤치마크 데이터셋입니다.

비전공자용 풀이: AI가 공장의 불량품을 얼마나 잘 찾아내는지 시험해 보기 위해 전 세계 과학자들이 공통으로 풀어보는 '인공지능 전용 불량품 문제집'입니다.

SD3.5 (Stable Diffusion 3.5)

공식 정의: Latent Diffusion 아키텍처를 기반으로 텍스트 입력값에 부합하는 고화질 이미지를 생성하는 최신 거대 생성형 AI 모델입니다.

비전공자용 풀이: 사람이 글로 "금속 표면의 기름 유출 불량"이라고 입력하면, 그에 맞는 사실적인 그림을 순식간에 그려내는 '천재 AI 화가'입니다.

ControlNet (컨트롤넷)

공식 정의: 사전 학습된 대형 확산 모델에 스케치, 깊이 정보 등의 공간적 조건(Spatial Conditioning)을 추가하여 이미지의 구조적 형태를 미세 제어하는 신경망 구조입니다.

비전공자용 풀이: AI 화가에게 그림을 그리라고 할 때, "이 밑그림 스케치와 원근감을 절대로 벗어나지 말고 그려!"라고 틀을 씌워주는 '투명 도안 가이드라인'입니다.

DoRA (Weight-Decomposed Low-Rank Adaptation)

공식 정의: 거대 모델의 가중치를 크기(Magnitude)와 방향(Direction) 성분으로 분해한 후, 낮은 순위(Low-rank) 행렬만을 미세조정하여 높은 학습 안정성을 확보한 매개변수 효율적 미세조정(PEFT) 기술입니다.

비전공자용 풀이: 거대한 AI 화가의 뇌 전체를 뜯어고치는 대신, 지식을 '크기'와 '방향'으로 쪼개어 컴퓨터 메모리를 아주 적게 쓰면서도 핵심 요점만 빠르게 독학시키는 스마트 가중치 과외 방식입니다.

SUPIR (Photo-Realistic Image Restoration)

공식 정의: 확산 모델의 강력한 사전 지식과 텍스트 프롬프트를 결합하여, 훼손되거나 해상도가 낮은 이미지의 세부 텍스처와 고주파 성분을 사실적으로 복원하는 초고해상도(Super-Resolution) 알고리즘입니다.

비전공자용 풀이: AI 화가가 대략적으로 그려낸 그림 위에 실제 물체 특유의 거친 느낌, 아주 미세한 실금이나 균열까지 실제 사진처럼 생생하게 코팅하고 다듬어주는 '최종 화질 복원 도구'입니다.

🔗 공식 정보 출처 및 관련 자료 링크
본 파이프라인 설계 및 공식 정의 작성을 위해 참고한 공식 저장소와 사이트 주소입니다.

MVTec AD 공식 데이터셋 다운로드: https://www.mvtec.com/research-teaching/datasets

Kaggle MVTec AD 데이터셋 공유소: https://www.kaggle.com/datasets/vipin20/mvtec-ad

Hugging Face PEFT (DoRA 기술 명세): https://huggingface.co/docs/peft/package_reference/lora

Hugging Face Diffusers ControlNet 개발 문서: https://huggingface.co/docs/diffusers/api/pipelines/controlnet

SUPIR 오픈소스 오리지널 코드 저장소: https://github.com/Fanghua-Yu/SUPIR
"""
---

## 🏗️ 프로젝트 폴더 구조 (Project Directory Structure)

본 프로젝트는 유지보수와 기능별 재사용성을 극대화하기 위해 조립식 부품처럼 코드를 쪼개어 관리하는 **모듈화(Modularization)** 설계를 채택했습니다.


```text
my_project/
├── README.md         # 프로젝트 소개 및 가이드라인 문서
├── preprocessor.py  # [1단계] 이미지 정방형 패딩 및 규격화 모듈
├── extractor.py     # [2, 3단계] Canny 외곽선 및 Depth 입체 구조 추출 모듈
├── dora_trainer.py  # [4단계] 불량 특성 집중 학습 및 DoRA 가중치 생성 모듈
├── augmentor.py     # [5단계] 다중 조건 가이드 기반 ControlNet 증강 생성 모듈
├── restorer.py      # [6단계] SUPIR 기반 고화질 미세 스크래치 질감 복원 모듈
└── main.py          # 모든 모듈 부품을 조립하고 총괄 제어하는 메인 관제탑
