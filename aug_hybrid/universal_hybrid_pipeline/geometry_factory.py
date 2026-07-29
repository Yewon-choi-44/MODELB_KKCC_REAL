import numpy as np
import cv2
from typing import Dict, Tuple
from category_registry import CategorySpec

class UniversalGeometryFactory:
    """
    범용 3D/시뮬레이션 기하 팩토리 (Universal 3D Geometry Factory)
    geom_type별로 알맞은 3D 변형(Dent, Crack, Scratch, Breakage 등) 및 
    Depth/Normal/GT Mask를 산출함.
    """
    def __init__(self, spec: CategorySpec, image_size: Tuple[int, int] = (1024, 1024)):
        self.spec = spec
        self.width, self.height = image_size

    def generate(self, defect_type: str = None) -> Dict[str, np.ndarray]:
        if defect_type is None:
            defect_type = self.spec.supported_defects[0]
            
        geom_type = self.spec.geom_type
        
        if geom_type == "rigid_cylinder":
            return self._gen_cylinder_geometry(defect_type)
        elif geom_type == "planar_surface":
            return self._gen_planar_geometry(defect_type)
        elif geom_type == "rigid_box":
            return self._gen_box_geometry(defect_type)
        elif geom_type == "organic_particle":
            return self._gen_organic_geometry(defect_type)
        else:
            return self._gen_planar_geometry(defect_type)

    def _gen_cylinder_geometry(self, defect_type: str) -> Dict[str, np.ndarray]:
        x = np.linspace(-1.0, 1.0, self.width)
        y = np.linspace(-1.0, 1.0, self.height)
        xx, yy = np.meshgrid(x, y)
        
        mask = (xx**2 <= 0.75).astype(np.uint8) * 255
        z_base = np.sqrt(np.maximum(0, 0.75 - xx**2))
        
        cx, cy = np.random.uniform(-0.3, 0.3), np.random.uniform(-0.4, 0.4)
        rad, intensity = np.random.uniform(0.2, 0.4), np.random.uniform(0.2, 0.45)
        
        dist_sq = (xx - cx)**2 + (yy - cy)**2
        deform = intensity * np.exp(-dist_sq / (2 * (rad**2)))
        deform[mask == 0] = 0.0
        
        gt_mask = (deform > (intensity * 0.15)).astype(np.uint8) * 255
        gt_mask = cv2.bitwise_and(gt_mask, mask)
        
        z_dented = np.maximum(0, z_base - deform)
        depth_map = (z_dented / np.max(z_base) * 255).astype(np.uint8)
        depth_map = cv2.bitwise_and(depth_map, mask)
        
        dz_dx, dz_dy = np.gradient(z_dented)
        normal_map = self._compute_normal_map(-dz_dx, -dz_dy, mask)
        
        return {"depth_map": depth_map, "normal_map": normal_map, "gt_mask": gt_mask, "mask": mask}

    def _gen_planar_geometry(self, defect_type: str) -> Dict[str, np.ndarray]:
        mask = np.full((self.height, self.width), 255, dtype=np.uint8)
        depth_map = np.full((self.height, self.width), 200, dtype=np.uint8)
        
        gt_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # 무작위 크랙/스크래치/구멍 생성
        num_defects = np.random.randint(1, 4)
        for _ in range(num_defects):
            pt1 = (np.random.randint(100, self.width-100), np.random.randint(100, self.height-100))
            pt2 = (pt1[0] + np.random.randint(-200, 200), pt1[1] + np.random.randint(-200, 200))
            thickness = np.random.randint(5, 25)
            cv2.line(gt_mask, pt1, pt2, 255, thickness)
            
        depth_map[gt_mask > 0] = np.clip(depth_map[gt_mask > 0] - 80, 0, 255)
        
        dz_dx, dz_dy = np.gradient(depth_map.astype(np.float32))
        normal_map = self._compute_normal_map(-dz_dx, -dz_dy, mask)
        
        return {"depth_map": depth_map, "normal_map": normal_map, "gt_mask": gt_mask, "mask": mask}

    def _gen_box_geometry(self, defect_type: str) -> Dict[str, np.ndarray]:
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.rectangle(mask, (200, 200), (self.width-200, self.height-200), 255, -1)
        
        depth_map = mask.copy()
        gt_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # 모서리 파손(Breakage/Chip)
        corner = (np.random.randint(self.width-300, self.width-150), np.random.randint(200, 350))
        cv2.circle(gt_mask, corner, np.random.randint(40, 90), 255, -1)
        gt_mask = cv2.bitwise_and(gt_mask, mask)
        
        depth_map[gt_mask > 0] = 0
        dz_dx, dz_dy = np.gradient(depth_map.astype(np.float32))
        normal_map = self._compute_normal_map(-dz_dx, -dz_dy, mask)
        
        return {"depth_map": depth_map, "normal_map": normal_map, "gt_mask": gt_mask, "mask": mask}

    def _gen_organic_geometry(self, defect_type: str) -> Dict[str, np.ndarray]:
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        center = (self.width // 2, self.height // 2)
        axes = (int(self.width * 0.3), int(self.height * 0.25))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        depth_map = mask.copy()
        gt_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # 불규칙 곰팡이/크랙 맵
        cx = center[0] + np.random.randint(-100, 100)
        cy = center[1] + np.random.randint(-100, 100)
        cv2.circle(gt_mask, (cx, cy), np.random.randint(30, 70), 255, -1)
        gt_mask = cv2.bitwise_and(gt_mask, mask)
        
        depth_map[gt_mask > 0] = np.clip(depth_map[gt_mask > 0] - 50, 0, 255)
        dz_dx, dz_dy = np.gradient(depth_map.astype(np.float32))
        normal_map = self._compute_normal_map(-dz_dx, -dz_dy, mask)
        
        return {"depth_map": depth_map, "normal_map": normal_map, "gt_mask": gt_mask, "mask": mask}

    def _compute_normal_map(self, dz_dx, dz_dy, mask) -> np.ndarray:
        normal_z = np.ones_like(dz_dx)
        norm = np.sqrt(dz_dx**2 + dz_dy**2 + normal_z**2) + 1e-6
        normal_x = dz_dx / norm
        normal_y = dz_dy / norm
        normal_z = normal_z / norm
        
        normal_map = np.stack([
            (normal_x + 1) * 0.5 * 255,
            (normal_y + 1) * 0.5 * 255,
            (normal_z + 1) * 0.5 * 255
        ], axis=-1).astype(np.uint8)
        
        return cv2.bitwise_and(normal_map, normal_map, mask=mask)
