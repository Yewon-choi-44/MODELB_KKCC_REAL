import glob
import os

import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import to_tensor

root = '/NHNHOME/WORKSPACE/26moel002_ex07/AD/Data' #B200 데이터셋 저장 폴더(공유폴더임)
# root = '/home/ai-engr/KKCC/' # ai-engr 컴퓨터 사용할 경우 폴더 경로임

PATH_TO_MVTEC_AD_1_FOLDER = root + '/OpenDataset/mvtec_ad_1'
MVTEC_AD_1_OBJECTS = [
    'bottle',
    'cable',
    'capsule',
    'carpet', 
    'grid', 
    'hazelnut',
    'leather', 
    'metal nut',
    'pill',
    'screw', 
    'tile', 
    'toothbrush', 
    'transistor', 
    'wood', 
    'zipper',
]

PATH_TO_MVTEC_AD_2_FOLDER = root + '/OpenDataset/mvtec_ad_2'
MVTEC_AD_2_OBJECTS = [
    'can',
    'fabric',
    'fruit_jelly',
    'rice',
    'sheet_metal',
    'vial',
    'wallplugs',
    'walnuts',
]

"""
Dataset에서 이미지 가져오는 등 컨트롤하기 위한 클래스
        <MVTec AD 1>
            * brachet은 원본 MVTec AD 1 데이터셋에는 없는 품목
            object (str): bottle, cable, capsule, carpet, grid, hazelnut, leather, metal nut,
                                pill, screw, tile, toothbrush, transistor, wood, zipper
            split (str): 'ground_truth', 'test', 'train'

        <MVTec AD 2>
            object_name (str): can, fabric, fruit_jelly, rice, sheet_metal, vial, wallplugs, walnuts
            split (str): train, validation, test_public, test_private, test_private_mixed
"""
class RawDataset:
    def __init__(self,
                 dataset_name = 'MVTEC_AD_2',
                 object_name = "can",
                 split = "train" ):
        self.dataset = dataset_name
        self.object = object_name
        self.split = split

        if self.dataset == 'MVTEC_AD_1':
            self._image_base_dir = PATH_TO_MVTEC_AD_1_FOLDER
        elif self.dataset == "MVTEC_AD_2" :
            self._image_base_dir = PATH_TO_MVTEC_AD_2_FOLDER

        self._object_dir = os.path.join(self._image_base_dir, object_name)

        # get all images from the split
        self._image_paths = sorted(glob.glob(self.get_image_path()))

    def get_image_path(self) -> str:
        if self.dataset == 'MVTEC_AD_1':
            if self.split == 'train':
                return os.path.join(
                    self._object_dir,
                    self.split, 'good',
                    '[0-9][0-9][0-9].png'
                )
            elif self.split == 'test': # 정상품과 불량품이 섞여 있음 나중에 각각 따로 가져와야할 경우 코드 수정필요!
                return os.path.join(
                    self._object_dir,
                    self.split, '*',
                    '[0-9][0-9][0-9].png'
                )
            else :
                return os.path.join(
                    self._object_dir,
                    self.split,
                    '*',
                    '[0-9][0-9]*.png'
                )
        else :
            if 'private' in self.split:
                return os.path.join(
                    self._object_dir, self.split, '[0-9][0-9][0-9]*.png'
                )
            return os.path.join(
                self._object_dir,
                self.split,
                '[gb][oa][od]*',
                '[0-9][0-9][0-9]*.png',
            )

    def get_image(self, idx: int):
        """Get dataset item for the index ``idx``.

        Args:
            idx (int): Index to get the image
        Returns:
            the sample image
        """
        image_path = self._image_paths[idx]
        sample = default_loader(image_path) # PIL open과 같음
        # sample.show()
        return sample

    def __len__(self):
        return len(self._image_paths)

    @property
    def image_paths(self):
        return self._image_paths

"""패딩한 이미지 폴더와 생성한 이미지 폴더(일단 메타데이터 폴더는 아직 사용안했음. 무시 바람)"""
PADDING_IMAGE_FOLDER = root + '/OpenDataset/padded_image'
METADATA_FOLDER = root + '/GenerateDataset/metadata'
GENERATED_IMAGE_FOLDER = root + '/GenerateDataset/'

class PaddingImages(RawDataset):
    def __init__(self, object_name, padding_image_folder=PADDING_IMAGE_FOLDER):
        self.object = object_name
        self._image_base_dir = padding_image_folder
        self._object_dir = os.path.join(self._image_base_dir, object_name)
        self._image_paths = None

    def set_image_path(self):
        os.makedirs(self._object_dir, exist_ok=True)
        return self._object_dir

    def get_image_path(self):
        return os.path.join(
            self._object_dir,
            '*.png'
        )

    @property
    def image_paths(self):
        if self._image_paths is None:
            self._image_paths = sorted(glob.glob(self.get_image_path()))
        return self._image_paths

def set_metadata_path(object_name):
    metadata_name = object_name + '_metadata.jsonl'
    metadata_dir = os.path.join(METADATA_FOLDER, metadata_name)
    return metadata_dir

def set_generated_image_path(model_name, object_name):
    generated_image_dir = os.path.join(GENERATED_IMAGE_FOLDER, model_name, object_name)
    os.makedirs(generated_image_dir, exist_ok=True)
    return generated_image_dir

