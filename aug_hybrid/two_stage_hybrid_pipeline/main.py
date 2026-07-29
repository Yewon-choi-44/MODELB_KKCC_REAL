import os
import argparse

from stage1_universal import Stage1UniversalAugmentor
from stage2_can_specialist import Stage2CanSpecialist
from stage2_vial_specialist import Stage2VialSpecialist
from stage2_sheetmetal_specialist import Stage2SheetMetalSpecialist
from stage2_wallplugs_specialist import Stage2WallplugSpecialist

def run_two_stage_hybrid_pipeline(targets, num_stage1, num_stage2, base_dir):
    print("==========================================================")
    print("[START] Two-Stage Hybrid Defect Augmentation Pipeline")
    print(f"        Target Categories : {targets}")
    print(f"        Stage 1 Samples   : {num_stage1} per category (Universal)")
    print(f"        Stage 2 Samples   : {num_stage2} per category (Specialized SOTA)")
    print("==========================================================")
    
    out_base = os.path.join(base_dir, "two_stage_hybrid_output")
    
    # ---------------------------------------------------------
    # STAGE 1: Universal Baseline Generation
    # ---------------------------------------------------------
    print("\n----------------------------------------------------------")
    print(">>> Phase 1: Rapid Baseline Generation via Universal Engine")
    print("----------------------------------------------------------")
    for cat in targets:
        stage1_augmentor = Stage1UniversalAugmentor(cat, out_base)
        stage1_augmentor.run_baseline_generation(num_stage1)
        
    # ---------------------------------------------------------
    # STAGE 2: Domain-Specific Specialized SOTA Generation
    # ---------------------------------------------------------
    print("\n----------------------------------------------------------")
    print(">>> Phase 2: High-Precision Domain Specialist Augmentation")
    print("----------------------------------------------------------")
    for cat in targets:
        cat_lower = cat.lower()
        if cat_lower == "can":
            can_specialist = Stage2CanSpecialist(out_base)
            can_specialist.run_specialized_generation(num_stage2)
        elif cat_lower == "vial":
            vial_specialist = Stage2VialSpecialist(out_base)
            vial_specialist.run_specialized_generation(num_stage2)
        elif cat_lower in ["sheet_metal", "sheetmetal"]:
            sheet_specialist = Stage2SheetMetalSpecialist(out_base)
            sheet_specialist.run_specialized_generation(num_stage2)
        elif cat_lower in ["wallplugs", "wallplug"]:
            wallplug_specialist = Stage2WallplugSpecialist(out_base)
            wallplug_specialist.run_specialized_generation(num_stage2)
        else:
            print(f"[Stage 2 Warning] Category '{cat}' relies on Stage 1 Universal dataset.")

    print("\n==========================================================")
    print("[SUCCESS] Two-Stage Hybrid Pipeline Execution Completed!")
    print(f"          Output Location: {out_base}")
    print("==========================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-Stage Hybrid Defect Augmentation Strategy")
    parser.add_argument("--targets", nargs="+", default=["can", "vial", "sheet_metal", "wallplugs"], help="Target categories for Stage 2 specialization")
    parser.add_argument("--num_stage1", type=int, default=2, help="Number of Stage 1 Universal Baseline samples")
    parser.add_argument("--num_stage2", type=int, default=2, help="Number of Stage 2 SOTA Specialized samples")
    parser.add_argument("--base_dir", type=str, default=r"D:\KKCC_Project", help="KKCC Project Base Directory")
    
    args = parser.parse_args()
    run_two_stage_hybrid_pipeline(args.targets, args.num_stage1, args.num_stage2, args.base_dir)
