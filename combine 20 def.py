import os
import numpy as np
import pandas as pd
import torch
import spectral
import spectral.io.envi as envi
import skimage.io as io
import spectral as spy

##########################################################################################
#dat和hdr的高光谱文件
def read_hdr_img_image(file_path):
    """
    读取单个 HDR 和 IMG 图像文件
    """
    image = spy.open_image(file_path)
    data = image.load()
    print(data.shape)
    return data


def load_hyperspectral_images(folder_path):
    """
    从指定文件夹读取所有高光谱图像（HDR和IMG格式）。
    参数：
    folder_path (str): 包含高光谱图像的文件夹路径。
    返回：
    images (list): 包含图像名称和数据的元组列表。
    """
    images = []
    # 遍历文件夹中的文件
    for file_name in os.listdir(folder_path):
        # 检查是否为HDR文件
        if file_name.endswith('.hdr'):
            hdr_path = os.path.join(folder_path, file_name)
            img_path = hdr_path.replace('.hdr', '.img')  # 生成对应的IMG文件路径

            # 确保IMG文件存在
            if os.path.isfile(img_path):
                # 读取高光谱图像
                image = spy.open_image(hdr_path)

                # 加载图像数据
                data = image.load()

                # 确保数据为 NumPy 数组
                if not isinstance(data, np.ndarray):
                    data = np.array(data)

                # 将图像和对应的文件名添加到列表中
                images.append((file_name, data))

                print(f"Loaded image: {file_name}, shape: {data.shape}")

    # 提取图像数据为数组
    images_data = [img[1] for img in images]

    return images_data  # 返回图像数据列表

def read_hdr_dat_image(file_path):
    """
    读取单个 HDR 和 DAT 图像文件
    """
    hdr_file = file_path.replace('.dat', '.hdr') if file_path.endswith('.dat') else file_path
    img = envi.open(hdr_file)
    print(img.shape)
    return img.load()

def read_hdr_images_from_folder(folder_path):
    """
    读取文件夹内所有 HDR 和 DAT 图像文件
    """
    hdr_images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.hdr') or file.endswith('.dat'):
                if file.endswith('.dat'):  # Only process .dat files to avoid duplication
                    file_path = os.path.join(root, file)
                    hdr_image = read_hdr_dat_image(file_path)
                    hdr_images.append(hdr_image)
                    print(f"Processed HDR/DAT file: {file_path}")
    return hdr_images

def combine_hdr_images(hdr_images):
    """
    将所有 HDR 图像按行拼接成一个 HDR 图像
    """
    combined_hdr_image = np.concatenate(hdr_images, axis=1)
    return combined_hdr_image

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


##########################################################################################
#tiff
def read_tiff_image(file_path):
    """
    读取单个 TIFF 图像文件
    """
    return io.imread(file_path)

def read_tiff_images_from_folder(folder_path):
    """
    读取文件夹内所有 TIFF 图像文件
    """
    tiff_images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.tif') or file.endswith('.tiff'):
                file_path = os.path.join(root, file)
                tiff_image = read_tiff_image(file_path)
                tiff_images.append(tiff_image)
                print(f"Processed TIFF file: {file_path}")
    return tiff_images

def combine_tiff_images(tiff_images):
    """
    将所有 TIFF 图像按行拼接成一个 TIFF 图像
    """
    combined_tiff_image = np.concatenate(tiff_images, axis=1)
    return combined_tiff_image

def save_combined_tiff_image(combined_tiff_image, output_path):
    """
    保存汇总后的 TIFF 图像
    """
    io.imsave(output_path, combined_tiff_image)

##########################################################################################
# 植被指数特征
#npy
def read_npy_image(folder_path):
    """
    读取单个 npy 图像文件
    """
    # 获取文件夹中所有的 .npy 文件
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    target_shape = (160, 3200, 128)  # 目标数组的形状3200,14个的大小是2240
    # 初始化目标数组
    combined_array = np.zeros(target_shape)
    # 读取每个 .npy 文件并合并到目标数组中
    for i, file_name in enumerate(file_list):
        file_path = os.path.join(folder_path, file_name)
        data = np.load(file_path)

        # 检查每个文件的形状是否正确
        if data.shape != (160, 160, 128):
            raise ValueError(f"文件 {file_name} 的形状不正确：{data.shape}")

        # 计算合并位置
        start_col = i * 160
        end_col = start_col + 160

        # 将数据放入目标数组中
        combined_array[:, start_col:end_col, :] = data

    return combined_array

##########################################################################################
##########################################################################################
##########################################################################################
#这里是单个特征的20个样方数据进行按行拼接，选择各自程序运行
# 使用，将20个样方数据按行拼接成（160，3200，128）,标签是(160, 3200)，纹理是(160, 3200, 160)，指数是(160, 3200, 128)
#换地址，且使用那部分就将其他部分注释掉
folder_path = 'data'#vegetation_indices或者data128或者data20_t160#D:/Python_Projects/9data-enhance/enhance_samples+14samples/t-feature
output_path = 'D:/Python_Projects/8TRANS/combined 20 def/480/480samples_combined_20.hdr'#combined_20.tiff或者combined_20.npy或者combined_20.hdr#D:/Python_Projects/9data-enhance/enhance_samples+14samples/samples_combined_14_t.tif
##########################################################################################
#读取高光谱hdr和dat格式的文件
hdr_images = read_hdr_images_from_folder(folder_path)#是个列表，里含20个 ,纹理每个（160,160,160），空谱是每个（160,160,1128）#读取HDR和DAT文件的图像数据时使用
#hdr_images = load_hyperspectral_images(folder_path)#读取HDR和IMG格式的高光谱图像数据时使用
print(len(hdr_images))
combined_hdr_image = np.concatenate(hdr_images, axis=1)
print(combined_hdr_image.shape)#纹理(160, 3200, 160)空谱(160, 3200, 128)
save_combined_hdr_image(combined_hdr_image, output_path)

print(f"Combined HDR image saved to: {output_path}")
##########################################################################################
#读取标签tiff格式的文件
#tiff_images = read_tiff_images_from_folder(folder_path)
#combined_tiff_image = combine_tiff_images(tiff_images)
#print(combined_tiff_image.shape)#(160, 3200)
#save_combined_tiff_image(combined_tiff_image, output_path)

#print(f"Combined TIFF image saved to: {output_path}")
##########################################################################################
#读取标签npy格式的文件
#combined_array = read_npy_image(folder_path)
#print(combined_array.shape)#(160, 3200, 128)
#np.save(output_path, combined_array)
#print(f"合并后的数组已保存到 {output_path}")

##########################################################################################
##########################################################################################
##########################################################################################
#这里是两个特征的20个样方数据进行按行拼接，选择各自程序运行
#换地址，且使用那部分就将其他部分注释掉
#folder1_path = r"D:/Python_Projects/9data-enhance/enhance_samples-2+14samples/enhance_labels_combined-2+14.tif"#vegetation_indices或者data128或者data20_t160
#folder2_path = r"D:/Python_Projects/9data-enhance/enhance_samples-2+14samples/enhance_labels_combined-2+14_t.tif"#vegetation_indices或者data128或者data20_t160
#folder3_path = r"D:/Python_Projects/9data-enhance/enhance_samples-2+14samples/enhance_labels_combined-2+14_e.tif"#vegetation_indices或者data128或者data20_t160
#output_path = r"D:/Python_Projects/9data-enhance/enhance_samples-2+14samples/enhance_labels_combined-2+14_s+t+e.tif"#combined_20.tiff或者combined_20.npy或者combined_20.hdr
##########################################################################################
#读取高光谱hdr和dat格式的文件++++纹理dat的文件
#hdr_images1 = read_hdr_images_from_folder(folder1_path)#是个列表，里含20个 ,空谱每个（160,160,128）
#combined_s_image = combine_hdr_images(hdr_images1)
#print(combined_s_image.shape)#空谱(160, 3200, 128)

#hdr_images2 = read_hdr_images_from_folder(folder2_path)#是个列表，里含20个 ,纹理每个（160,160,160）
#combined_t_image = combine_hdr_images(hdr_images2)
#print(combined_t_image.shape)#纹理(160, 3200, 160)

# 在最后一个维度上合并
#combined_st_image = np.concatenate((combined_s_image, combined_t_image), axis=-1)
# 检查结果的形状
#print(combined_st_image.shape)  # 输出应该是 (160, 3200, 288)
#save_combined_hdr_image(combined_st_image, output_path)
#print(f"合并后的数组已保存到 {output_path}")

##########################################################################################
#读取高光谱hdr和dat格式的文件++++指数特征npy格式的文件
#hdr_images1 = read_hdr_images_from_folder(folder1_path)#是个列表，里含20个 ,空谱每个（160,160,128）
#combined_s_image = combine_hdr_images(hdr_images1)
#print(combined_s_image.shape)#空谱(160, 3200, 128)

#读取指数npy格式的文件
#combined_s_image = read_tiff_image(folder1_path)#指数(160, 3200, 128)
#print(combined_s_image.shape)
#combined_t_image = read_tiff_image(folder2_path)#指数(160, 3200, 128)
#print(combined_t_image.shape)
#combined_e_image = read_tiff_image(folder3_path)#指数(160, 3200, 128)
#print(combined_e_image.shape)
#combined_ste_image = np.concatenate((combined_s_image, combined_t_image, combined_e_image), axis=1)
# 在最后一个维度上合并
#combined_se_image = np.concatenate((combined_s_image, combined_e_image), axis=-1)
# 检查结果的形状
#print(combined_ste_image.shape)  # 输出应该是 (160, 3200, 256)
#io.imsave(output_path, combined_ste_image)#保存成npy格式的数据
#print(f"合并后的数组已保存到 {output_path}")

##########################################################################################
#读取纹理文件++++指数特征npy格式的文件
#hdr_images1 = read_hdr_images_from_folder(folder1_path)#是个列表，里含20个 ,纹理每个（160,160,160）
#combined_t_image = combine_hdr_images(hdr_images1)
#print(combined_t_image.shape)#空谱(160, 3200, 160)

#读取指数npy格式的文件
#combined_e_image = read_npy_image(folder2_path)#指数(160, 3200, 128)
#print(combined_e_image.shape)#空谱(160, 3200, 136)

# 在最后一个维度上合并
#combined_te_image = np.concatenate((combined_t_image, combined_e_image), axis=-1)
# 检查结果的形状
#print(combined_te_image.shape)  # 输出应该是 (160, 3200, 288)
#np.save(output_path, combined_te_image)#保存成npy格式的数据
#print(f"合并后的数组已保存到 {output_path}")


##########################################################################################
##########################################################################################
##########################################################################################
#这里是三个特征的20个样方数据进行按行拼接，选择各自程序运行
# 使用，将20个样方数据按行拼接成（160，3200，128）,标签是(160, 3200)，纹理是(160, 3200, 160)
#换地址，且使用那部分就将其他部分注释掉
#folder1_path = 'data128'#vegetation_indices或者data128或者data20_t160
#folder2_path = 'data20_t160'#vegetation_indices或者data128或者data20_t160
#folder3_path = 'vegetation_indices'#vegetation_indices或者data128或者data20_t160
#output_path = 'combined 20 def/2_3feature/combined_20_s+t+e.npy'#combined_20.tiff或者combined_20.npy或者combined_20.hdr
##########################################################################################
#读取高光谱hdr和dat格式的文件++++纹理dat++++指数格式npy的文件
#hdr_images1 = read_hdr_images_from_folder(folder1_path)#是个列表，里含20个 ,空谱每个（160,160,128）
#combined_s_image = combine_hdr_images(hdr_images1)
#print(combined_s_image.shape)#空谱(160, 3200, 128)

#hdr_images2 = read_hdr_images_from_folder(folder2_path)#是个列表，里含20个 ,纹理每个（160,160,160）
#combined_t_image = combine_hdr_images(hdr_images2)
#print(combined_t_image.shape)#纹理(160, 3200, 160)

#读取指数npy格式的文件
#combined_e_image = read_npy_image(folder2_path)#指数(160, 3200, 128)
#print(combined_e_image.shape)#空谱(160, 3200, 128)

# 在最后一个维度上合并
#combined_ste_image = np.concatenate((combined_s_image, combined_t_image, combined_e_image), axis=-1)
# 检查结果的形状
#print(combined_ste_image.shape)  # 输出应该是 (160, 3200, 416)
##print(f"合并后的数组已保存到 {output_path}")
