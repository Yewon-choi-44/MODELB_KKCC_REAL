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
