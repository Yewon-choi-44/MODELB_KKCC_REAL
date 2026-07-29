# ADDED: __init__.py
# two_stage_hybrid_pipeline 디렉터리를 Python 패키지로 인식시킵니다.
#
# NOTE: 내부 스크립트(main.py 등)가 "from stage1_universal import ..." 처럼
#       절대 이름으로 import하기 때문에, 패키지 초기화 시 해당 디렉터리를
#       sys.path 에 추가한 뒤 importlib 로 각 모듈을 직접 로드합니다.
#       기존 파일은 한 글자도 수정하지 않습니다.

import sys                                          # ADDED
import importlib.util                               # ADDED
from pathlib import Path                            # ADDED

_PIPELINE_DIR = Path(__file__).resolve().parent     # ADDED: this package's directory

# 패키지 내부 스크립트들이 서로를 절대 이름으로 참조할 수 있도록 sys.path 등록
if str(_PIPELINE_DIR) not in sys.path:              # ADDED
    sys.path.insert(0, str(_PIPELINE_DIR))          # ADDED


def _load_module(name: str):
    """패키지 디렉터리에서 파일을 직접 로드해 모듈 객체를 반환합니다."""  # ADDED
    spec = importlib.util.spec_from_file_location(  # ADDED
        name, _PIPELINE_DIR / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)     # ADDED
    sys.modules[name] = mod                         # ADDED: 이후 import 캐시에 등록
    spec.loader.exec_module(mod)                    # ADDED
    return mod                                      # ADDED


# ---------------------------------------------------------------------------
# ADDED: 각 모듈을 로드하고 주요 심볼을 패키지 네임스페이스로 노출
# ---------------------------------------------------------------------------
_stage1   = _load_module("stage1_universal")        # ADDED
_stage2c  = _load_module("stage2_can_specialist")   # ADDED
_stage2v  = _load_module("stage2_vial_specialist")  # ADDED
_stage2s  = _load_module("stage2_sheetmetal_specialist")  # ADDED
_stage2w  = _load_module("stage2_wallplugs_specialist")   # ADDED
_main_mod = _load_module("main")                    # ADDED  (main.py)

Stage1UniversalAugmentor     = _stage1.Stage1UniversalAugmentor     # ADDED
Stage2CanSpecialist          = _stage2c.Stage2CanSpecialist          # ADDED
Stage2VialSpecialist         = _stage2v.Stage2VialSpecialist         # ADDED
Stage2SheetMetalSpecialist   = _stage2s.Stage2SheetMetalSpecialist   # ADDED
Stage2WallplugSpecialist     = _stage2w.Stage2WallplugSpecialist     # ADDED
run_two_stage_hybrid_pipeline = _main_mod.run_two_stage_hybrid_pipeline  # ADDED

__all__ = [                                         # ADDED
    "Stage1UniversalAugmentor",
    "Stage2CanSpecialist",
    "Stage2VialSpecialist",
    "Stage2SheetMetalSpecialist",
    "Stage2WallplugSpecialist",
    "run_two_stage_hybrid_pipeline",
]
