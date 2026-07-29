# =============================================================================
# ADDED: hybrid_runner.py
#
# 역할:
#   - DataAugmentation/main.py 의 기존 코드를 전혀 수정하지 않고,
#     aug_hybrid/two_stage_hybrid_pipeline 의 모듈들을 import 하여
#     2단계 하이브리드 파이프라인을 실행하는 독립 진입점.
#
# 사용법:
#   python hybrid_runner.py                            # 대화형 모드
#   python hybrid_runner.py --targets can vial         # 지정 품목
#   python hybrid_runner.py --targets all              # 전 품목
#   python hybrid_runner.py --stage 1                  # Stage 1 만
#   python hybrid_runner.py --stage 2 --targets can    # Stage 2 만, can 품목
#   python hybrid_runner.py --num_stage1 5 --num_stage2 10
#
# =============================================================================

import sys                          # ADDED
import argparse                     # ADDED
from pathlib import Path            # ADDED

# ---------------------------------------------------------------------------
# ADDED: aug_hybrid 패키지(two_stage_hybrid_pipeline)를 import 가능하도록
#        DataAugmentation 루트를 sys.path 에 등록.
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent          # ADDED: DataAugmentation/
sys.path.insert(0, str(ROOT_DIR))                   # ADDED

# ---------------------------------------------------------------------------
# ADDED: two_stage_hybrid_pipeline 패키지에서 핵심 함수·클래스 import.
#        __init__.py 를 통해 패키지 레벨로 노출된 심볼을 사용.
# ---------------------------------------------------------------------------
from aug_hybrid.two_stage_hybrid_pipeline import (  # ADDED
    Stage1UniversalAugmentor,                        # ADDED
    Stage2CanSpecialist,                             # ADDED
    Stage2VialSpecialist,                            # ADDED
    Stage2SheetMetalSpecialist,                      # ADDED
    Stage2WallplugSpecialist,                        # ADDED
    run_two_stage_hybrid_pipeline,                   # ADDED: 오케스트레이션 함수
)

# ---------------------------------------------------------------------------
# ADDED: DataAugmentation/main.py 에서 사용하는 품목 목록과 동일하게 유지.
# ---------------------------------------------------------------------------
MVTEC_AD_1_OBJECTS = [                              # ADDED (main.py와 동기화)
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill',
    'screw', 'tile', 'toothbrush', 'transistor',
    'wood', 'zipper',
]
MVTEC_AD_2_OBJECTS = [                              # ADDED (main.py와 동기화)
    'can', 'fabric', 'fruit_jelly', 'rice',
    'sheet_metal', 'vial', 'wallplugs', 'walnuts',
]
ALL_OBJECTS = MVTEC_AD_1_OBJECTS + MVTEC_AD_2_OBJECTS  # ADDED

# ---------------------------------------------------------------------------
# ADDED: Stage 1 전용 실행 함수
# ---------------------------------------------------------------------------
def run_stage1_only(targets: list, num_samples: int, base_dir: str):
    """Stage 1 범용 베이스라인만 실행합니다."""
    out_base = str(Path(base_dir) / "two_stage_hybrid_output")
    for cat in targets:
        aug = Stage1UniversalAugmentor(cat, out_base)   # ADDED
        aug.run_baseline_generation(num_samples)         # ADDED
    print(f"\n[Stage 1 완료] 출력 경로: {out_base}")


# ---------------------------------------------------------------------------
# ADDED: Stage 2 전용 실행 함수
# ---------------------------------------------------------------------------
def run_stage2_only(targets: list, num_samples: int, base_dir: str):
    """Stage 2 스페셜리스트만 실행합니다."""
    out_base = str(Path(base_dir) / "two_stage_hybrid_output")
    for cat in targets:
        cat_lower = cat.lower()
        if cat_lower == "can":
            Stage2CanSpecialist(out_base).run_specialized_generation(num_samples)       # ADDED
        elif cat_lower == "vial":
            Stage2VialSpecialist(out_base).run_specialized_generation(num_samples)      # ADDED
        elif cat_lower in ["sheet_metal", "sheetmetal"]:
            Stage2SheetMetalSpecialist(out_base).run_specialized_generation(num_samples)  # ADDED
        elif cat_lower in ["wallplugs", "wallplug"]:
            Stage2WallplugSpecialist(out_base).run_specialized_generation(num_samples)  # ADDED
        else:
            print(f"[Stage 2 Warning] '{cat}' 는 Stage 2 스페셜리스트가 없습니다. Stage 1 결과를 사용하세요.")
    print(f"\n[Stage 2 완료] 출력 경로: {out_base}")


# ---------------------------------------------------------------------------
# ADDED: CLI 진입점
# ---------------------------------------------------------------------------
def parse_args():                                                       # ADDED
    parser = argparse.ArgumentParser(
        description="Two-Stage Hybrid Defect Augmentation Runner"
    )
    parser.add_argument(
        "--stage",
        choices=["1", "2", "all"],
        default="all",
        help="실행할 단계 (1: 범용, 2: 스페셜리스트, all: 전체). 기본값 all",
    )                                                                   # ADDED
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["can", "vial", "sheet_metal", "wallplugs"],
        help="대상 품목 목록 (all 입력 시 전 품목). 예: --targets can vial",
    )                                                                   # ADDED
    parser.add_argument(
        "--num_stage1",
        type=int,
        default=2,
        help="Stage 1 에서 품목당 생성할 샘플 수. 기본값 2",
    )                                                                   # ADDED
    parser.add_argument(
        "--num_stage2",
        type=int,
        default=2,
        help="Stage 2 에서 품목당 생성할 샘플 수. 기본값 2",
    )                                                                   # ADDED
    parser.add_argument(
        "--base_dir",
        type=str,
        default=r"D:\KKCC_Project",
        help=r"프로젝트 루트 디렉터리. 기본값 D:\KKCC_Project",
    )                                                                   # ADDED
    return parser.parse_args()


def main():                                                             # ADDED
    args = parse_args()

    # "all" 입력 시 전 품목으로 확장
    targets = ALL_OBJECTS if args.targets == ["all"] else args.targets  # ADDED

    print("=" * 60)
    print(" KKCC 2단계 하이브리드 파이프라인 실행")
    print(f" 단계    : {args.stage}")
    print(f" 대상    : {targets}")
    print(f" Stage1 샘플 수 : {args.num_stage1}")
    print(f" Stage2 샘플 수 : {args.num_stage2}")
    print(f" 출력 경로 : {args.base_dir}")
    print("=" * 60)

    if args.stage == "1":                                               # ADDED
        run_stage1_only(targets, args.num_stage1, args.base_dir)
    elif args.stage == "2":                                             # ADDED
        run_stage2_only(targets, args.num_stage2, args.base_dir)
    else:  # "all"                                                      # ADDED
        run_two_stage_hybrid_pipeline(                                  # ADDED
            targets=targets,
            num_stage1=args.num_stage1,
            num_stage2=args.num_stage2,
            base_dir=args.base_dir,
        )


if __name__ == "__main__":                                              # ADDED
    main()
