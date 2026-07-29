from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class CategorySpec:
    name: str
    dataset_version: str  # "mvtec_ad_1" or "mvtec_ad_2"
    geom_type: str        # "rigid_cylinder", "rigid_box", "planar_surface", "organic_particle"
    material: str         # "metal", "glass", "plastic", "fabric", "wood", "ceramic", "food"
    default_prompt: str
    supported_defects: List[str]

CATEGORY_REGISTRY: Dict[str, CategorySpec] = {
    # --- MVTec AD 2 (8 Categories) ---
    "can": CategorySpec(
        name="can", dataset_version="mvtec_ad_2", geom_type="rigid_cylinder", material="metal",
        default_prompt="industrial metallic beverage can with severe dented and scratched surface, realistic specular reflection, studio lighting, 8k",
        supported_defects=["dent", "scratch", "puncture"]
    ),
    "sheet_metal": CategorySpec(
        name="sheet_metal", dataset_version="mvtec_ad_2", geom_type="planar_surface", material="metal",
        default_prompt="industrial sheet metal surface with deep dent, crack and scratch defect, metallic texture, 8k",
        supported_defects=["dent", "crack", "scratch", "hole"]
    ),
    "vial": CategorySpec(
        name="vial", dataset_version="mvtec_ad_2", geom_type="rigid_cylinder", material="glass",
        default_prompt="transparent glass pharmaceutical vial with broken crack and dent defect, studio lighting, 8k",
        supported_defects=["crack", "dent", "chip"]
    ),
    "wallplugs": CategorySpec(
        name="wallplugs", dataset_version="mvtec_ad_2", geom_type="rigid_box", material="plastic",
        default_prompt="plastic wallplug anchor with deformed, broken and twisted defect, industrial macro photo, 8k",
        supported_defects=["deformation", "breakage", "crack"]
    ),
    "fabric": CategorySpec(
        name="fabric", dataset_version="mvtec_ad_2", geom_type="planar_surface", material="fabric",
        default_prompt="textile fabric surface with torn hole, stain and thread defect, macro texture, 8k",
        supported_defects=["tear", "stain", "hole"]
    ),
    "rice": CategorySpec(
        name="rice", dataset_version="mvtec_ad_2", geom_type="organic_particle", material="food",
        default_prompt="rice grains with discolored, broken and cracked defective grain, high resolution macro",
        supported_defects=["chip", "discoloration", "crack"]
    ),
    "fruit_jelly": CategorySpec(
        name="fruit_jelly", dataset_version="mvtec_ad_2", geom_type="organic_particle", material="food",
        default_prompt="translucent fruit jelly with air bubble and foreign particle defect, macro photography, 8k",
        supported_defects=["bubble", "foreign_body", "deformation"]
    ),
    "walnuts": CategorySpec(
        name="walnuts", dataset_version="mvtec_ad_2", geom_type="organic_particle", material="food",
        default_prompt="natural walnut shell with severe crack, mold and broken piece defect, detailed macro",
        supported_defects=["crack", "mold", "breakage"]
    ),

    # --- MVTec AD 1 (15 Categories) ---
    "bottle": CategorySpec(
        name="bottle", dataset_version="mvtec_ad_1", geom_type="rigid_cylinder", material="glass",
        default_prompt="glass bottle with cracked glass surface and broken rim defect, 8k photo",
        supported_defects=["crack", "broken_rim", "contamination"]
    ),
    "cable": CategorySpec(
        name="cable", dataset_version="mvtec_ad_1", geom_type="rigid_cylinder", material="plastic",
        default_prompt="electric cable with cut insulation wire defect, industrial macro photo",
        supported_defects=["cut", "bent_wire", "missing_cable"]
    ),
    "capsule": CategorySpec(
        name="capsule", dataset_version="mvtec_ad_1", geom_type="rigid_cylinder", material="plastic",
        default_prompt="pharmaceutical capsule with cracked shell and squeeze defect, macro",
        supported_defects=["crack", "squeeze", "faulty_imprint"]
    ),
    "carpet": CategorySpec(
        name="carpet", dataset_version="mvtec_ad_1", geom_type="planar_surface", material="fabric",
        default_prompt="carpet surface with hole, stain and cut defect, macro texture",
        supported_defects=["hole", "stain", "cut"]
    ),
    "grid": CategorySpec(
        name="grid", dataset_version="mvtec_ad_1", geom_type="planar_surface", material="metal",
        default_prompt="metal wire grid with broken mesh and bent wire defect, industrial",
        supported_defects=["broken_wire", "bent_wire", "glue"]
    ),
    "hazelnut": CategorySpec(
        name="hazelnut", dataset_version="mvtec_ad_1", geom_type="organic_particle", material="food",
        default_prompt="hazelnut with cracked shell and hole defect, detailed macro",
        supported_defects=["crack", "hole", "print"]
    ),
    "leather": CategorySpec(
        name="leather", dataset_version="mvtec_ad_1", geom_type="planar_surface", material="leather",
        default_prompt="leather material surface with cut, fold and color stain defect, 8k",
        supported_defects=["cut", "fold", "stain"]
    ),
    "metal_nut": CategorySpec(
        name="metal_nut", dataset_version="mvtec_ad_1", geom_type="rigid_box", material="metal",
        default_prompt="metal hex nut with scratched thread and dent defect, industrial macro",
        supported_defects=["scratch", "bent", "color"]
    ),
    "pill": CategorySpec(
        name="pill", dataset_version="mvtec_ad_1", geom_type="organic_particle", material="food",
        default_prompt="medical pill tablet with cracked edge and color stain defect, macro",
        supported_defects=["crack", "combined", "color"]
    ),
    "screw": CategorySpec(
        name="screw", dataset_version="mvtec_ad_1", geom_type="rigid_cylinder", material="metal",
        default_prompt="metal screw with deformed thread and scratched head defect, macro",
        supported_defects=["deformed_thread", "scratch", "head_damage"]
    ),
    "tile": CategorySpec(
        name="tile", dataset_version="mvtec_ad_1", geom_type="planar_surface", material="ceramic",
        default_prompt="ceramic floor tile surface with deep crack, oil stain and broken corner defect, 8k",
        supported_defects=["crack", "oil_stain", "gray_stroke"]
    ),
    "toothbrush": CategorySpec(
        name="toothbrush", dataset_version="mvtec_ad_1", geom_type="rigid_box", material="plastic",
        default_prompt="toothbrush head with bent bristles and missing hair defect, macro",
        supported_defects=["defective_bristle", "missing_hair"]
    ),
    "transistor": CategorySpec(
        name="transistor", dataset_version="mvtec_ad_1", geom_type="rigid_box", material="plastic",
        default_prompt="electronic transistor with bent pin and damaged casing defect",
        supported_defects=["bent_lead", "damaged_case", "cut_lead"]
    ),
    "wood": CategorySpec(
        name="wood", dataset_version="mvtec_ad_1", geom_type="planar_surface", material="wood",
        default_prompt="wooden board surface with knot, scratch and hole defect, natural texture 8k",
        supported_defects=["hole", "scratch", "liquid"]
    ),
    "zipper": CategorySpec(
        name="zipper", dataset_version="mvtec_ad_1", geom_type="planar_surface", material="metal",
        default_prompt="metal zipper with broken tooth and split track defect, macro",
        supported_defects=["broken_teeth", "split_teeth", "fabric_tear"]
    )
}

def get_category_spec(category_name: str) -> CategorySpec:
    if category_name not in CATEGORY_REGISTRY:
        raise ValueError(f"Category '{category_name}' not found in registry. Available: {list(CATEGORY_REGISTRY.keys())}")
    return CATEGORY_REGISTRY[category_name]
