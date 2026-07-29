import os
import cv2
import argparse
import numpy as np

from category_registry import CATEGORY_REGISTRY, get_category_spec
from geometry_factory import UniversalGeometryFactory
from universal_augmentor import UniversalAugmentor

def run_universal_augmentation(category: str, num_samples: int, dataset_base_dir: str):
    """
    범용 결함 데이터 증강 메인 관제 함수
    """
    if category.lower() == "all":
        target_categories = list(CATEGORY_REGISTRY.keys())
    else:
        target_categories = [category]

    print("==========================================================")
    print(f"[START] Starting Universal Hybrid Defect Augmentation Framework")
    print(f"        Target Categories Count: {len(target_categories)}")
    print(f"        Samples per Category   : {num_samples}")
    print("==========================================================")

    for cat_name in target_categories:
        spec = get_category_spec(cat_name)
        
        # 품목별 출력 폴더 설정
        out_dir = os.path.join(dataset_base_dir, spec.dataset_version, spec.name, "augmented_output")
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "depths"), exist_ok=True)

        print(f"\n>>> Processing Category: [{spec.name}] ({spec.dataset_version})")
        print(f"    Geometry Type : {spec.geom_type} | Material: {spec.material}")
        print(f"    Output Path   : {out_dir}")

        geom_factory = UniversalGeometryFactory(spec)
        augmentor = UniversalAugmentor(spec)

        for i in range(num_samples):
            sample_id = f"{spec.name}_aug_{i+1:04d}"
            
            # 1. 3D & 물리 시뮬레이션 기하 맵 생성
            geom_data = geom_factory.generate()
            
            # 2. Generative AI 실사 합성 & 복원
            synth_img = augmentor.synthesize_and_restore(geom_data)
            
            # 3. 데이터 저장
            img_path = os.path.join(out_dir, "images", f"{sample_id}.png")
            mask_path = os.path.join(out_dir, "masks", f"{sample_id}_mask.png")
            depth_path = os.path.join(out_dir, "depths", f"{sample_id}_depth.png")
            
            synth_img.save(img_path)
            cv2.imwrite(mask_path, geom_data["gt_mask"])
            cv2.imwrite(depth_path, geom_data["depth_map"])
            
            print(f" -> [{i+1}/{num_samples}] Saved {sample_id}.png + GT Mask")

    print("\n==========================================================")
    print(f"[SUCCESS] All target category augmentations finished cleanly.")
    print("==========================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Hybrid Defect Augmentation Framework")
    parser.add_argument("--category", type=str, default="can", help="Category name or 'all'")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of synthetic samples to generate per category")
    parser.add_argument("--base_dir", type=str, default=r"D:\KKCC_Project", help="Base directory for KKCC_Project")
    
    args = parser.parse_args()
    run_universal_augmentation(args.category, args.num_samples, args.base_dir)
