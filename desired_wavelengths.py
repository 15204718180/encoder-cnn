import spectral
import os
import numpy as np
import spectral.io.envi as envi

def save_combined_hdr_image(combined_hdr_image, output_path):
    """
    保存汇总后的 HDR 图像
    """
    # 创建一个 ENVI 图像对象
    header = {
        'lines': combined_hdr_image.shape[0],
        'samples': combined_hdr_image.shape[1],
        'bands': combined_hdr_image.shape[2],
        'interleave': 'bil',
        'data type': 4,
        'byte order': 0,
        'wavelength': np.arange(combined_hdr_image.shape[2])
    }
    envi.save_image(output_path, combined_hdr_image, dtype=np.float32, metadata=header)

# 读取高光谱图像
image = spectral.open_image(r'D:\Python_Projects\8TRANS\data\025.hdr')  # 使用相应的文件格式
# 将数据转换为 numpy 数组
data = image.load()
data = np.array(data)
print(data.shape)#(160, 160, 480)
# 获取图像的波段数量
num_bands = data.shape[2]
print("Number of bands:", num_bands)  # 打印实际波段数量
# 定义所需波段索引（从0开始）
desired_band_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,
                        35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,
                        67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,85,86,87,88,89,90,91,92,93,94,95,97,98,99,100,
                        101,102,104,105,106,107,110,111,112,114,116,124,125,126,127,129,130,131,132,133,134,135,136,137,
                        138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,
                        162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,
                        186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,
                        210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,
                        234,235,236,237,238,240,241,242,243,244,245,246,247,248,252,255,256,257,258,260,264,272,276,277,
                        278,279,280,281,282,283,284,285,286,288,290,328,329]
# 确保索引在有效范围内
valid_indices = [i for i in desired_band_indices if 0 <= i < num_bands]
#print("Valid indices:", valid_indices)  # 打印有效的索引
# 提取指定波段
if valid_indices:
    extracted_image = data[:, :, valid_indices]  # 提取有效波段
    print("Extracted image shape:", extracted_image.shape)  # 应该是 (160, 160, len(valid_indices))
else:
    print("No valid band indices found.")
save_combined_hdr_image(extracted_image, r'D:\Python_Projects\8TRANS\data256\025.hdr')