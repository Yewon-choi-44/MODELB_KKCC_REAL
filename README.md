# KKCC Model B 파이프라인 구현 가이드

본 문서는 [pipeline.md](file:///D:/wiki/wiki/projects/KKCC/pipeline.md)에 기술된 **model B 전용 MVTec AD 2 불량 데이터 증강 파이프라인**을 실제 파이썬 프로그램으로 구현하기 위한 아키텍처 설계도 및 스켈레톤 코드 명세서입니다. 유지보수와 모듈화를 극대화하기 위해 각 구성 요소를 독립적인 파이썬 파일로 구성하며, `main.py`에서 총괄 제어하도록 설계되었습니다.

---

## 📦 전체 시스템 아키텍처

```mermaid
flowchart TD
    Raw[MVTec AD 2 원본 이미지] --> Prep[preprocessor.py: 정방형 패딩]
    Prep --> PadImg[패딩 이미지 1024x1024]
    
    PadImg --> ExtCanny[extractor.py: Canny Edge 추출]
    PadImg --> ExtDepth[extractor.py: Depth Map 추출]
    
    PadImg --> Train[trainer.py: SD3.5 Full Fine-Tuning]
    Train --> FineTunedSD[SD3.5 Tuning Weights]
    
    ExtCanny --> Aug[augmentor.py: Multi-ControlNet 생성]
    ExtDepth --> Aug
    FineTunedSD --> Aug
    
    Aug --> GenImg[생성된 증강 이미지]
    GenImg --> Rest[restorer.py: SUPIR 초고해상도 복원]
    Rest --> Final[최종 불량 증강 데이터셋]
```
