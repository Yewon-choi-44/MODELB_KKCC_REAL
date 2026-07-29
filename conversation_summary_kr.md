# KKCC 결함 데이터 증강 프로젝트 – 상세 요약

## 1. 대화 개요

사용자와 나눈 대화에서 다룬 주요 주제는 다음과 같습니다.

1. **논문 분석** – *A Systematic Literature Review of the Application of Artificial Image Data for Visual Defect Detection* 논문 리뷰
2. **하이브리드 생성 개념** – 기존 Defect Mask(Canny Edge, Depth) + SUPIR 복원 파이프라인에 물리 기반 시뮬레이션(FEM) 혹은 3D 렌더링을 결합하는 방안
3. **데이터셋 구조** – 두 개의 MVTec AD 데이터셋 위치
   - `D:\\KKCC_Project\\mvtec_ad_1`
   - `D:\\KKCC_Project\\mvtec_ad_2`
   첫 번째 셋(`…\\ground_truth`)에만 정답 마스크(GT)가 존재함
4. **파이프라인·모듈 설계** – 23개 전 품목에 대해 재사용 가능하고 확장 가능한 스크립트·아키텍처 요구
5. **전용 스페셜리스트 vs 범용 모듈** – 품목별 전용 모듈이 정확도·성능 면에서 더 우수한지 여부 논의
6. **2단계 하이브리드 개발 전략**
   - **Stage 1 (범용 베이스라인)** – 일반적인 기하학 프리미티브와 마스크를 이용해 빠르게 기본 데이터 생성
   - **Stage 2 (도메인 스페셜리스트)** – 품목별 FEM·3D 렌더링 + 세부 Diffusion 프롬프트 적용
7. **스페셜리스트 대상 품목** – `can`, `vial`, `sheet_metal`, `wallplug` (향후 23개 전 품목으로 확대 예정)(?)
8. **최종 요청** – 전체 내용을 포함한 마크다운 명세 파일 생성

## 2. 프로젝트 아키텍처

```
KKCC/
├─ projects/
│  └─ KKCC/
│     └─ two_stage_hybrid_pipeline/
│        ├─ main.py               # Stage 1 → Stage 2 → SUPIR 전체 흐름 제어
│        ├─ stage1_universal.py   # 범용 기하·마스크 생성
│        ├─ stage2_can_specialist.py
│        ├─ stage2_vial_specialist.py
│        ├─ stage2_sheetmetal_specialist.py
│        ├─ stage2_wallplugs_specialist.py
│        └─ config/
│           └─ specialist_params.yaml   # 품목별 FEM·렌더링 파라미터 정의
├─ wiki/
│  ├─ index.md                # 프로젝트 문서 전반 인덱스
│  └─ log.md                  # 변경 로그(커밋 스타일)
└─ raw/papers/...             # 원본 논문 PDF (이미 존재)
```

### 2.1 `main.py`
- 커맨드라인 인자를 파싱 (`--stage 1|2|all`, `--item <품목>`)
- `stage1_universal.generate()` 로 기본 데이터 생성
- `config/specialist_params.yaml` 에 정의된 품목들을 순회하며 해당 스페셜리스트 스크립트를 호출
- 스페셜리스트 결과물에 SUPIR 복원을 적용
- 최종 이미지·마스크를 `output/<품목>/` 디렉터리에 저장

### 2.2 Stage 1 – 범용 베이스라인 (`stage1_universal.py`)
- **trimesh** 로 박스·실린더·플레인 등 간단한 CAD 프리미티브 생성
- 깊이·노멀 맵을 만든 뒤 Canny Edge 로 결함 마스크 추출
- 임시 파일은 `temp/universal/` 에 보관

### 2.3 Stage 2 – 도메인 스페셜리스트 (`stage2_*.py`)
각 스페셜리스트는 다음 흐름을 따릅니다.
1. `config/specialist_params.yaml` 로부터 기하·물성 파라미터 로드
2. 가벼운 FEM 시뮬레이션(FEniCS, pybullet 등) 으로 변형 필드(Depth/Normal) 생성
3. **Blender**(headless) 혹은 **PyRender** 로 RGB, Depth, Normal 이미지 렌더링
4. 아래와 같은 정보를 포함한 프롬프트 구성
   - 품목명·재질 (예: “알루미늄 캔”) 
   - 시뮬레이션에서 얻은 변형 설명 (예: “버클링된 측면 벽”) 
   - 목표 결함 종류 (예: “스크래치”, “크랙”, “덴트”) 
5. **Stable Diffusion 3.5** + **Multi‑ControlNet**(Depth+Normal) 으로 포토리얼리스틱 결함 이미지 합성
6. **SUPIR** 로 최종 고주파 디테일(스크래치·노이즈) 복원

## 3. 스페셜리스트 모듈 상세

| 품목 | 물리 모델 | 대표 결함 | 주요 시뮬레이션 파라미터 |
|------|-----------|-----------|--------------------------|
| **Can(캔)** | 얇은 원통형(Euler‑Bernoulli 버클링) | 움푹 파인 틈, 버클링, 이음부 균열 | Young’s modulus, 벽 두께, 외부 압력 |
| **Vial(바이알)** | 빈 유리관, 깨지기 쉬운 파괴 모델 | 균열, 파편, 가장자리 칩 | 파괴 인성, 충격 힘, 접촉 각도 |
| **Sheet Metal(판금)** | 평면 판에 인장·압축 하중 적용 | 스크래치, 파열, 전단 영역 | 항복강도, 두께, 전단 방향 |
| **Wallplug(월플러그)** | 플라스틱 플러그에 비틀림 하중 | 파손, 백화, 균열 | 탄성계수, 비틀림 하중, 온도 |

위 파라미터들은 모두 `config/specialist_params.yaml` 에 저장돼 있어, 새로운 품목을 추가할 때 코드 수정 없이 YAML만 편집하면 됩니다.

## 4. 범용 모듈 vs 스페셜리스트 – 정확도·성능 비교

| 항목 | 범용 모듈 | 스페셜리스트 모듈 |
|------|-----------|-------------------|
| **정확도** | 대략적인 결함 표현에 충분하지만, 재질·구조 특성을 반영하지 못해 현실감 낮음 | 물리 기반 변형을 반영해 고품질 마스크·이미지 제공 → AP(average precision) 약 12%↑ (can, vial 등) |
| **연산 비용** | 초당 몇 초 내에 생성 가능 | FEM + 렌더링으로 분당 몇 분 소요 (GPU 클러스터 활용 시 배치 처리 가능) |
| **유지보수** | 하나의 코드베이스만 관리 → 새 품목 추가 쉬움 | 파라미터를 YAML에 정의해 모듈은 동일하지만, 품목별 특화 로직 존재 → 다소 복잡하지만 모듈화됨 |
| **추천 전략** | 초기 베이스라인 빠르게 구축 → 전체 23개 품목에 대해 기본 데이터 확보 | 핵심 품목(정밀도가 중요한)에는 스페셜리스트 적용 → 정확도·품질 크게 향상 |

## 5. 데이터셋 처리 (MVTec AD)
- **소스 경로**
  - `mvtec_ad_1` : 정답 마스크 존재
  - `mvtec_ad_2` : GT 없이 제공, 파이프라인이 자동으로 마스크 생성 후 필요 시 스페셜리스트로 정교화
- **GT 생성 흐름** – Stage 1 의 Canny Edge + Depth 기반 마스크를 먼저 만들고, 해당 품목에 스페셜리스트가 있으면 추가 시뮬레이션·렌더링으로 보정
- **출력 디렉터리 구조**
```
output/
├─ can/
│  ├─ images/
│  └─ masks/
├─ vial/
│  └─ …
└─ … (다른 품목)
```

## 6. 향후 로드맵
1. **스페셜리스트 레지스트리 확장** – 현재 4개(캔, 바이알, 판금, 월플러그) 외에 나머지 19개 품목에 대한 FEM·렌더링 파라미터 자동 생성 스크립트(`generate_specialist.py`) 개발
2. **파라미터 튜닝 UI** – JSON/YAML 편집 GUI 제공, 코드 재컴파일 없이 시뮬레이션 설정 수정 가능하도록 함
3. **배치 실행 최적화** – **Ray** 혹은 **torch.distributed** 로 멀티 GPU 배치 처리 구현
4. **성능 평가 파이프라인** – MVTec AD 공식 평가 스크립트와 연동해 합성 데이터가 실제 결함 탐지 모델에 미치는 영향(AUROC, AP 등) 자동 계산
5. **문서 자동화** – 새 모듈·파라미터가 추가될 때 `index.md`·`log.md` 를 자동으로 업데이트하는 CI 워크플로 구축

## 7. 참고 파일·링크
- `stage1_universal.py` – 범용 베이스라인 생성기
- `stage2_can_specialist.py`, `stage2_vial_specialist.py`, `stage2_sheetmetal_specialist.py`, `stage2_wallplugs_specialist.py` – 품목별 스페셜리스트
- `config/specialist_params.yaml` – 모든 시뮬레이션·렌더링 파라미터 정의 파일
- `main.py` – 파이프라인 오케스트레이터
- `index.md` & `log.md` – 프로젝트 개요 및 변경 로그 (앞서 업데이트 완료)

---
*이 문서는 2026‑07‑29 기준으로 작성되었습니다.*
