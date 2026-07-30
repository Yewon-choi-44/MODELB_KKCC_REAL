import os
from PIL import Image
import matplotlib.pyplot as plt

"""
GenerateDataset
|-- SD35
|   `-- can
|-- SDXL
|   |-- can_numInference_50
|   `-- vial
`-- SDXL_only_0720
    |-- bottle
    |-- cable
    |-- can
    |-- capsule
    |-- ...
    `-- zipper
"""

padding_path = "/NHNHOME/WORKSPACE/26moel002_ex07/AD/Data/OpenDataset/padded_image/can"
SD35_path = "/NHNHOME/WORKSPACE/26moel002_ex07/AD/Data/GenerateDataset/SD35/can"
SDXL_path = "/NHNHOME/WORKSPACE/26moel002_ex07/AD/Data/GenerateDataset/SDXL/can"

fig, axes = plt.subplots(1, 3, figsize=(30, 11))

image_path = ["/NHNHOME/WORKSPACE/26moel002_ex07/AD/Data/OpenDataset/padded_image/can/can_001.png",
              "/NHNHOME/WORKSPACE/26moel002_ex07/AD/Data/GenerateDataset/SD35/can/gen_can_001.png",
              "/NHNHOME/WORKSPACE/26moel002_ex07/AD/Data/GenerateDataset/SDXL/can/gen_can_001.png",]
titles = ["Raw Image", "SD35", "SDXL"]

for i in range(3):
    img = Image.open(image_path[i])
    axes[i].imshow(img)
    axes[i].set_title(titles[i], fontsize=12, fontweight='bold')

plt.show()
plt.savefig("can_001")