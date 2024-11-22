import numpy as np
import spectral
from osgeo import gdal

#舍弃这个指数###############################计算 MSRre：确保平方根内的数值都是非负值，负值都被设置为0
def calculate_msrre(nir_band2, red_edge_band):
    safe_subtract = np.clip(np.subtract(nir_band2, red_edge_band), 0, None)
    return np.divide(np.subtract(nir_band2, red_edge_band) - 1, np.sqrt(safe_subtract) + 1)

def replace_zeros_with_epsilon(data, band_index, epsilon=1e-10):#只有样本增强中的8-1中计算指数特征使用，其他时候就注释掉
    """
    将指定波段中的0值替换为一个非常小的数值 epsilon。
    参数：
    - data: 三维数据数组。
    - band_index: 要处理的波段索引（从0开始计数）。
    - epsilon: 替换0值的非常小的数值，默认值为 1e-10。
    返回：
    - 修改后的三维数据数组。
    """
    # 将指定波段中的所有数值替换为 epsilon
    data[:, :, band_index] = epsilon

    # 提取并返回修改后的波段数据
    return data[:, :, band_index]
#这里是读取dat格式的高光谱图像文件
#*******************************************************************************************
# 读取多光谱图像数据 dat，以及各个波段的数据
# 读取高光谱数据
#data_file = r"D:\Python_Projects\8TRANS\data128\025.dat"
#data = gdal.Open(data_file)
# 读取数据
#data_array = data.ReadAsArray()
#print(data_array.shape)
# 将数组的维度从（128，160，160）变换成（160，160，128）
#data = np.transpose(data_array, (1, 2, 0))
#print(data.shape)

#这里是读取hdr格式的高光谱图像文件
#*******************************************************************************************
input_file = r"D:\Python_Projects\9data-enhance\enhance_samples+14samples\s-feature\025_cl.hdr"#r"D:\Python_Projects\9data-enhance\outdata128\8-1_up2cocl_2cl.hdr"
# 读取 ENVI 数据
img = spectral.open_image(input_file)
# 将数据转换为 numpy 数组
data = img.load()
data = np.array(data)
print(data.shape)#(160, 160, 128)

#这里是读取虚拟样本Npy格式的文件
#data_file = r"D:\Python_Projects\9data-enhance\outdata_virtual_samples\virtual_samples.npy"
#data = np.load(data_file)
#print(data.shape)#(160, 320, 128)

#开始获取各个波段信息，准备计算植被指数
#*******************************************************************************************

#要求数据时浮点型数据，避免出现零除错误
blue_band = data[:,:,10]# 获取蓝光波段数据,蓝光波段在第11个波段431
#print(blue_band)
#print(blue_band.shape)#(160, 160)
green_band = data[:,:,37]# 获取绿光波段数据,绿光波段在第38个波段515
#print(green_band)
red_band1 = data[:,:,75]# 获取红光波段数据,红光1波段在第76个波段649
red_band2 = data[:,:,82]# 获取红光波段数据,红光2波段在第83个波段658
red_band3 = data[:,:,68]# 获取红光波段数据,红光3波段在第69个波段640
red_band4 = data[:,:,93]# 获取红光波段数据,红光4波段在第94个波段673
red_edge_band = data[:,:,107]# 获取红边波段数据,红边波段在第108个波段691
nir_band1 = data[:,:,120]# 获取近红外波段数据,近红外波段1在第121个波段763
nir_band2 = data[:,:,122]# 获取近红外波段数据,近红外波段2在第123个波段取766,总有个别样方计算MSRre时会出现开平方是负值的报错

#********************************************************************************************
#只有样本增强中的8-1中计算指数特征使用，其他时候就注释掉，增强样本8-1的绿、红、红边、近红外波段的额数据均为0，无法计算指数，所以都置为一个很小的数
# 调用函数替换第38个波段中的0值
#modified_data_g = replace_zeros_with_epsilon(data, 37)
#print(modified_data_g.shape)#(160, 160)
# 调用函数替换第75个波段中的0值
#modified_data_r1 = replace_zeros_with_epsilon(data, 75)
# 调用函数替换第82个波段中的0值
#modified_data_r2 = replace_zeros_with_epsilon(data, 82)
# 调用函数替换第68个波段中的0值
#modified_data_r3 = replace_zeros_with_epsilon(data, 68)
# 调用函数替换第93个波段中的0值
#modified_data_r4 = replace_zeros_with_epsilon(data, 93)
# 调用函数替换第107个波段中的0值
#modified_data_re = replace_zeros_with_epsilon(data, 107)
# 调用函数替换第120个波段中的0值
#modified_data_n1 = replace_zeros_with_epsilon(data, 120)
# 调用函数替换第122个波段中的0值
#modified_data_n2 = replace_zeros_with_epsilon(data, 122)

#green_band = modified_data_g#
#red_band1 = modified_data_r1#
#red_band2 = modified_data_r2#
#red_band3 = modified_data_r3#
#red_band4 = modified_data_r4#
#red_edge_band = modified_data_re#
#nir_band1 = modified_data_n1#
#nir_band2 = modified_data_n2#




#*************************************************************************************************************
###第一组指数特征值：红光波段取1，近红外波段取1
# 计算 IRECI---参照哨兵二号
#ireci1 = np.divide(np.subtract(nir_band, red_band), (np.divide(data_705, data_750)))
# 计算 CLre
clre1 = np.subtract((np.divide(nir_band1, red_edge_band)), 1)
# 计算 NDRE
ndre1 = np.divide(np.subtract(nir_band1, red_edge_band), np.add(nir_band1, red_edge_band))
# 计算 RNDVI
rndvi1 = np.divide(np.subtract(red_edge_band, red_band1), np.add(red_edge_band, red_band1))
#计算 MSRre
#msrre1 = calculate_msrre(nir_band1, red_edge_band)
# 计算 MTCI
#如果分母为0时置为非常小的一个数：在计算去噪加噪的扩展样本8-1中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
# 使用 np.where 替换分母为0或接近0的情况
denominator1 = np.subtract(red_edge_band, red_band1)
denominator_safe1 = np.where(np.abs(denominator1) < 1e-10, 1e-10, denominator1)
# 计算 mtci1
mtci1 = np.divide(np.subtract(nir_band1, red_edge_band), denominator_safe1)
#原始的计算公式
#mtci1 = np.divide(np.subtract(nir_band1, red_edge_band), np.subtract(red_edge_band, red_band1))
#计算 REP---参照哨兵二号
#rep1 = np.add(data_705, 35 * (np.subtract(np.divide((np.add(data_665, data_783)), 2), data_705)) / (np.subtract(data_740, data_705)))
# 计算 NDVI
ndvi1 = np.divide((np.subtract(nir_band1, red_band1)), (np.add(nir_band1, red_band1)))
# 计算 GNDVI
gndvi1 = np.divide((np.subtract(nir_band1, green_band)), (np.add(nir_band1, green_band)))
# 计算 OSAVI
osavi1 = np.divide(np.subtract(nir_band1, red_band1), np.add(nir_band1, red_band1 + 0.16))
# 计算 LCI
lci1 = np.divide(np.subtract(nir_band1, red_edge_band), np.add(nir_band1, red_band1))
# 计算 EVI
evi1 = 2.5 * np.divide(np.subtract(nir_band1, red_band1), np.add(nir_band1, 6 * red_band1 - 7.5 * blue_band + 1))
# 计算 DVI
dvi1 = np.subtract(nir_band1, red_band1)
# 计算 RVI
rvi1 = np.divide(nir_band1, red_band1)
# 计算 SAVI,这里L暂取0.5
savi1 = np.multiply((np.divide((np.subtract(nir_band1, red_band1)), (np.add(nir_band1, red_band1 + 0.5)))), (np.add(1 , 0.5)))
# 计算 MSAVI
#如果根号下有负值则置为0：在计算去噪加噪的扩展样本中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
express3 = np.power(2 * nir_band1 + 1, 2) - 8 * (nir_band1 - red_band1)
express3[express3 < 0] = 0
msavi1 = np.divide(2 * nir_band1 + 1 - np.sqrt(express3), 2)
#原始的计算公式
#msavi1 = np.divide(2 * nir_band1 + 1 - np.sqrt(np.power(2 * nir_band2 + 1, 2) - 8 * (nir_band2 - red_band1)), 2)
# 计算 GCI
gci1 = np.divide(nir_band1, green_band) - 1
# 计算 TVI
tvi1 = np.subtract(60 * np.subtract(nir_band1, green_band), 100 * np.subtract(red_band1, green_band))
# 计算 MTVI2
#如果根号下有负值则置为0：在计算去噪加噪的扩展样本中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
expression1 = np.power(2 * nir_band1 + 1, 2) - (6 * nir_band1 - 5 * np.sqrt(np.abs(red_band1))) - 0.5
expression1[expression1 < 0] = 0
# 防止分母为0的情况，将分母中接近0的值设置为一个很小的数值，以避免除零错误
epsilon1 = 1e-10  # 非常小的值
expression_with_epsilon1 = np.where(expression1 < epsilon1, epsilon1, expression1)
mtvi21 = 1.5 * (1.2 * (nir_band1 - green_band) - 2.5 * (red_band1 - green_band)) / np.sqrt(expression1)
#原始的计算公式
mtvi21 = 1.5 * (1.2 * (nir_band1 - green_band) - 2.5 * (red_band1 - green_band)) / np.sqrt(np.power(2 * nir_band2+ 1, 2)  - (6 * nir_band2 - 5 * np.sqrt(red_band1)) - 0.5)


###第二组指数特征值：红光波段取2，近红外波段取1
# 计算 IRECI---参照哨兵二号
#ireci2 = np.divide(np.subtract(nir_band1, red_band), (np.divide(data_705, data_750)))
# 计算 CLre
clre2 = np.subtract((np.divide(nir_band1, red_edge_band)), 1)
# 计算 NDRE
ndre2 = np.divide(np.subtract(nir_band1, red_edge_band), np.add(nir_band1, red_edge_band))
# 计算 RNDVI
rndvi2 = np.divide(np.subtract(red_edge_band, red_band2), np.add(red_edge_band, red_band2))
#计算 MSRre
#msrre2 = calculate_msrre(nir_band1, red_edge_band)
# 计算 MTCI
#如果分母为0时置为非常小的一个数：在计算去噪加噪的扩展样本8-1中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
# 使用 np.where 替换分母为0或接近0的情况
denominator2 = np.subtract(red_edge_band, red_band2)
denominator_safe2 = np.where(np.abs(denominator2) < 1e-10, 1e-10, denominator2)
# 计算 mtci1
mtci2 = np.divide(np.subtract(nir_band1, red_edge_band), denominator_safe2)
#原始的计算公式
#mtci2 = np.divide(np.subtract(nir_band1, red_edge_band), np.subtract(red_edge_band, red_band2))
#计算 REP---参照哨兵二号
#rep2 = np.add(data_705, 35 * (np.subtract(np.divide((np.add(data_665, data_783)), 2), data_705)) / (np.subtract(data_740, data_705)))
# 计算 NDVI
ndvi2 = np.divide((np.subtract(nir_band1, red_band2)), (np.add(nir_band1, red_band2)))
# 计算 GNDVI
gndvi2 = np.divide((np.subtract(nir_band1, green_band)), (np.add(nir_band1, green_band)))
# 计算 OSAVI
osavi2 = np.divide(np.subtract(nir_band1, red_band2), np.add(nir_band1, red_band2 + 0.16))
# 计算 LCI
lci2 = np.divide(np.subtract(nir_band1, red_edge_band), np.add(nir_band1, red_band2))
# 计算 EVI
evi2 = 2.5 * np.divide(np.subtract(nir_band1, red_band2), np.add(nir_band1, 6 * red_band2 - 7.5 * blue_band + 1))
# 计算 DVI
dvi2 = np.subtract(nir_band1, red_band2)
# 计算 RVI
rvi2 = np.divide(nir_band1, red_band2)
# 计算 SAVI,这里L暂取0.5
savi2 = np.multiply((np.divide((np.subtract(nir_band1, red_band2)), (np.add(nir_band1, red_band2 + 0.5)))), (np.add(1 , 0.5)))
# 计算 MSAVI
#如果根号下有负值则置为0：在计算去噪加噪的扩展样本中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
express1 = np.power(2 * nir_band1 + 1, 2) - 8 * (nir_band1 - red_band2)
express1[express1 < 0] = 0
msavi2 = np.divide(2 * nir_band1 + 1 - np.sqrt(express1), 2)
#原始的计算公式
#msavi2 = np.divide(2 * nir_band1 + 1 - np.sqrt(np.power(2 * nir_band1 + 1, 2) - 8 * (nir_band1 - red_band2)), 2)
# 计算 GCI
gci2 = np.divide(nir_band1, green_band) - 1
# 计算 TVI
tvi2 = np.subtract(60 * np.subtract(nir_band1, green_band), 100 * np.subtract(red_band2, green_band))
# 计算 MTVI2
#如果根号下有负值则置为0：在计算去噪加噪的扩展样本中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
expression2 = np.power(2 * nir_band1 + 1, 2) - (6 * nir_band1 - 5 * np.sqrt(np.abs(red_band2))) - 0.5
expression2[expression2 < 0] = 0
# 防止分母为0的情况，将分母中接近0的值设置为一个很小的数值，以避免除零错误
epsilon2 = 1e-10  # 非常小的值
expression_with_epsilon2 = np.where(expression2 < epsilon2, epsilon2, expression2)
mtvi22 = 1.5 * (1.2 * (nir_band1 - green_band) - 2.5 * (red_band2 - green_band)) / np.sqrt(expression2)
#原始的计算公式
#mtvi22 = 1.5 * (1.2 * (nir_band1 - green_band) - 2.5 * (red_band2 - green_band)) / np.sqrt(np.power(2 * nir_band2+ 1, 2)  - (6 * nir_band2 - 5 * np.sqrt(red_band2)) - 0.5)


###第三组指数特征值：红光波段取3，近红外波段取1
# 计算 IRECI---参照哨兵二号
#ireci3 = np.divide(np.subtract(nir_band1, red_band), (np.divide(data_705, data_750)))
# 计算 CLre
clre3 = np.subtract((np.divide(nir_band1, red_edge_band)), 1)
# 计算 NDRE
ndre3 = np.divide(np.subtract(nir_band1, red_edge_band), np.add(nir_band1, red_edge_band))
# 计算 RNDVI
rndvi3 = np.divide(np.subtract(red_edge_band, red_band3), np.add(red_edge_band, red_band3))
#计算 MSRre
#msrre3 = np.divide(np.subtract(nir_band1, red_edge_band) - 1, np.sqrt(np.subtract(nir_band1, red_edge_band)) + 1)
# 计算 MTCI
#如果分母为0时置为非常小的一个数：在计算去噪加噪的扩展样本8-1中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
# 使用 np.where 替换分母为0或接近0的情况
denominator3 = np.subtract(red_edge_band, red_band1)
denominator_safe3 = np.where(np.abs(denominator3) < 1e-10, 1e-10, denominator3)
# 计算 mtci1
mtci3 = np.divide(np.subtract(nir_band1, red_edge_band), denominator_safe3)
#原始的计算公式
#mtci3 = np.divide(np.subtract(nir_band1, red_edge_band), np.subtract(red_edge_band, red_band3))
#计算 REP---参照哨兵二号
#rep3 = np.add(data_705, 35 * (np.subtract(np.divide((np.add(data_665, data_783)), 2), data_705)) / (np.subtract(data_740, data_705)))
# 计算 NDVI
ndvi3 = np.divide((np.subtract(nir_band1, red_band3)), (np.add(nir_band1, red_band3)))
# 计算 GNDVI
gndvi3 = np.divide((np.subtract(nir_band1, green_band)), (np.add(nir_band1, green_band)))
# 计算 OSAVI
osavi3 = np.divide(np.subtract(nir_band1, red_band3), np.add(nir_band1, red_band3 + 0.16))
# 计算 LCI
lci3 = np.divide(np.subtract(nir_band1, red_edge_band), np.add(nir_band1, red_band3))
# 计算 EVI
evi3 = 2.5 * np.divide(np.subtract(nir_band1, red_band3), np.add(nir_band1, 6 * red_band3 - 7.5 * blue_band + 1))
# 计算 DVI
dvi3 = np.subtract(nir_band1, red_band3)
# 计算 RVI
rvi3 = np.divide(nir_band1, red_band3)
# 计算 SAVI,这里L暂取0.5
savi3 = np.multiply((np.divide((np.subtract(nir_band1, red_band3)), (np.add(nir_band1, red_band3 + 0.5)))), (np.add(1 , 0.5)))
# 计算 MSAVI
#如果根号下有负值则置为0：在计算去噪加噪的扩展样本中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
express7 = np.power(2 * nir_band1 + 1, 2) - 8 * (nir_band1 - red_band3)
express7[express7 < 0] = 0
msavi3 = np.divide(2 * nir_band1 + 1 - np.sqrt(express7), 2)
#原始的计算公式
#msavi3 = np.divide(2 * nir_band1 + 1 - np.sqrt(np.power(2 * nir_band1 + 1, 2) - 8 * (nir_band1 - red_band3)), 2)
# 计算 GCI
gci3 = np.divide(nir_band1, green_band) - 1
# 计算 TVI
tvi3 = np.subtract(60 * np.subtract(nir_band1, green_band), 100 * np.subtract(red_band3, green_band))
# 计算 MTVI2
#如果根号下有负值则置为0：在计算去噪加噪的扩展样本中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
expression3 = np.power(2 * nir_band1 + 1, 2) - (6 * nir_band1 - 5 * np.sqrt(np.abs(red_band3))) - 0.5
expression3[expression3 < 0] = 0
# 防止分母为0的情况，将分母中接近0的值设置为一个很小的数值，以避免除零错误
epsilon3 = 1e-10  # 非常小的值
expression_with_epsilon3 = np.where(expression3 < epsilon3, epsilon3, expression3)
mtvi23 = 1.5 * (1.2 * (nir_band1 - green_band) - 2.5 * (red_band3 - green_band)) / np.sqrt(expression3)
#原始的计算公式
#mtvi23 = 1.5 * (1.2 * (nir_band1 - green_band) - 2.5 * (red_band3 - green_band)) / np.sqrt(np.power(2 * nir_band1+ 1, 2)  - (6 * nir_band1 - 5 * np.sqrt(red_band3)) - 0.5)


###第四组指数特征值：红光波段取4，近红外波段取1
# 计算 IRECI---参照哨兵二号
#ireci4 = np.divide(np.subtract(nir_band1, red_band), (np.divide(data_705, data_750)))
# 计算 CLre
clre4 = np.subtract((np.divide(nir_band1, red_edge_band)), 1)
# 计算 NDRE
ndre4 = np.divide(np.subtract(nir_band1, red_edge_band), np.add(nir_band1, red_edge_band))
# 计算 RNDVI
rndvi4 = np.divide(np.subtract(red_edge_band, red_band4), np.add(red_edge_band, red_band4))
#计算 MSRre
#msrre4 = np.divide(np.subtract(nir_band1, red_edge_band) - 1, np.sqrt(np.subtract(nir_band1, red_edge_band)) + 1)
# 计算 MTCI
#如果分母为0时置为非常小的一个数：在计算去噪加噪的扩展样本8-1中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
# 使用 np.where 替换分母为0或接近0的情况
denominator4 = np.subtract(red_edge_band, red_band4)
denominator_safe4 = np.where(np.abs(denominator4) < 1e-10, 1e-10, denominator4)
# 计算 mtci1
mtci4 = np.divide(np.subtract(nir_band1, red_edge_band), denominator_safe4)
#原始的计算公式
#mtci4 = np.divide(np.subtract(nir_band1, red_edge_band), np.subtract(red_edge_band, red_band4))
#计算 REP---参照哨兵二号
#rep4 = np.add(data_705, 35 * (np.subtract(np.divide((np.add(data_665, data_783)), 2), data_705)) / (np.subtract(data_740, data_705)))
# 计算 NDVI
ndvi4 = np.divide((np.subtract(nir_band1, red_band4)), (np.add(nir_band1, red_band4)))
# 计算 GNDVI
gndvi4 = np.divide((np.subtract(nir_band1, green_band)), (np.add(nir_band1, green_band)))
# 计算 OSAVI
osavi4 = np.divide(np.subtract(nir_band1, red_band4), np.add(nir_band1, red_band4 + 0.16))
# 计算 LCI
lci4 = np.divide(np.subtract(nir_band1, red_edge_band), np.add(nir_band1, red_band4))
# 计算 EVI
evi4 = 2.5 * np.divide(np.subtract(nir_band1, red_band4), np.add(nir_band1, 6 * red_band4 - 7.5 * blue_band + 1))
# 计算 DVI
dvi4 = np.subtract(nir_band1, red_band4)
# 计算 RVI
rvi4 = np.divide(nir_band1, red_band4)
# 计算 SAVI,这里L暂取0.5
savi4 = np.multiply((np.divide((np.subtract(nir_band1, red_band4)), (np.add(nir_band1, red_band4 + 0.5)))), (np.add(1 , 0.5)))
# 计算 MSAVI
#如果根号下有负值则置为0：在计算去噪加噪的扩展样本中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
express5 = np.power(2 * nir_band1 + 1, 2) - 8 * (nir_band1 - red_band4)
express5[express5 < 0] = 0
msavi4 = np.divide(2 * nir_band1 + 1 - np.sqrt(express5), 2)
#原始的计算公式
#msavi4 = np.divide(2 * nir_band1 + 1 - np.sqrt(np.power(2 * nir_band1 + 1, 2) - 8 * (nir_band1 - red_band4)), 2)
# 计算 GCI
gci4 = np.divide(nir_band1, green_band) - 1
# 计算 TVI
tvi4 = np.subtract(60 * np.subtract(nir_band1, green_band), 100 * np.subtract(red_band4, green_band))
# 计算 MTVI2
#如果根号下有负值则置为0：在计算去噪加噪的扩展样本中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
expression4 = np.power(2 * nir_band1 + 1, 2) - (6 * nir_band1 - 5 * np.sqrt(np.abs(red_band4))) - 0.5
expression4[expression4 < 0] = 0
# 防止分母为0的情况，将分母中接近0的值设置为一个很小的数值，以避免除零错误
epsilon4 = 1e-10  # 非常小的值
expression_with_epsilon4 = np.where(expression4 < epsilon4, epsilon4, expression4)
mtvi24 = 1.5 * (1.2 * (nir_band1 - green_band) - 2.5 * (red_band4 - green_band)) / np.sqrt(expression4)
#原始的计算公式
#mtvi24 = 1.5 * (1.2 * (nir_band1 - green_band) - 2.5 * (red_band4 - green_band)) / np.sqrt(np.power(2 * nir_band1+ 1, 2)  - (6 * nir_band1 - 5 * np.sqrt(red_band4)) - 0.5)

###第5组指数特征值：红光波段取1，近红外波段取2
# 计算 IRECI---参照哨兵二号
#ireci1 = np.divide(np.subtract(nir_band, red_band), (np.divide(data_705, data_750)))
# 计算 CLre
clre5 = np.subtract((np.divide(nir_band2, red_edge_band)), 1)
# 计算 NDRE
ndre5 = np.divide(np.subtract(nir_band2, red_edge_band), np.add(nir_band2, red_edge_band))
# 计算 RNDVI
rndvi5 = np.divide(np.subtract(red_edge_band, red_band1), np.add(red_edge_band, red_band1))
#计算 MSRre
#msrre5 = np.divide(np.subtract(nir_band2, red_edge_band) - 1, np.sqrt(np.subtract(nir_band2, red_edge_band)) + 1)
# 计算 MTCI
#如果分母为0时置为非常小的一个数：在计算去噪加噪的扩展样本8-1中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
# 使用 np.where 替换分母为0或接近0的情况
denominator5 = np.subtract(red_edge_band, red_band1)
denominator_safe5 = np.where(np.abs(denominator5) < 1e-10, 1e-10, denominator5)
# 计算 mtci1
mtci5 = np.divide(np.subtract(nir_band2, red_edge_band), denominator_safe5)
#原始的计算公式
#mtci5 = np.divide(np.subtract(nir_band2, red_edge_band), np.subtract(red_edge_band, red_band1))
#计算 REP---参照哨兵二号
#rep5 = np.add(data_705, 35 * (np.subtract(np.divide((np.add(data_665, data_783)), 2), data_705)) / (np.subtract(data_740, data_705)))
# 计算 NDVI
ndvi5 = np.divide((np.subtract(nir_band2, red_band1)), (np.add(nir_band2, red_band1)))
# 计算 GNDVI
gndvi5 = np.divide((np.subtract(nir_band2, green_band)), (np.add(nir_band2, green_band)))
# 计算 OSAVI
osavi5 = np.divide(np.subtract(nir_band2, red_band1), np.add(nir_band2, red_band1 + 0.16))
# 计算 LCI
lci5 = np.divide(np.subtract(nir_band2, red_edge_band), np.add(nir_band2, red_band1))
# 计算 EVI
evi5 = 2.5 * np.divide(np.subtract(nir_band2, red_band1), np.add(nir_band2, 6 * red_band1 - 7.5 * blue_band + 1))
# 计算 DVI
dvi5 = np.subtract(nir_band2, red_band1)
# 计算 RVI
rvi5 = np.divide(nir_band2, red_band1)
# 计算 SAVI,这里L暂取0.5
savi5 = np.multiply((np.divide((np.subtract(nir_band2, red_band1)), (np.add(nir_band2, red_band1 + 0.5)))), (np.add(1 , 0.5)))
# 计算 MSAVI
#如果根号下有负值则置为0：在计算去噪加噪的扩展样本中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
express4 = np.power(2 * nir_band2 + 1, 2) - 8 * (nir_band2 - red_band1)
express4[express4 < 0] = 0
msavi5 = np.divide(2 * nir_band2 + 1 - np.sqrt(express4), 2)
#原始的计算公式
#msavi5 = np.divide(2 * nir_band2 + 1 - np.sqrt(np.power(2 * nir_band2 + 1, 2) - 8 * (nir_band2 - red_band1)), 2)
# 计算 GCI
gci5 = np.divide(nir_band2, green_band) - 1
# 计算 TVI
tvi5 = np.subtract(60 * np.subtract(nir_band2, green_band), 100 * np.subtract(red_band1, green_band))
# 计算 MTVI2
#如果根号下有负值则置为0：在计算去噪加噪的扩展样本中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
expression5 = np.power(2 * nir_band2 + 1, 2) - (6 * nir_band2 - 5 * np.sqrt(np.abs(red_band1))) - 0.5
expression5[expression5 < 0] = 0
# 防止分母为0的情况，将分母中接近0的值设置为一个很小的数值，以避免除零错误
epsilon5 = 1e-10  # 非常小的值
expression_with_epsilon5 = np.where(expression5 < epsilon5, epsilon5, expression5)
mtvi25 = 1.5 * (1.2 * (nir_band2 - green_band) - 2.5 * (red_band1 - green_band)) / np.sqrt(expression5)
#原始的计算公式
#mtvi25 = 1.5 * (1.2 * (nir_band2 - green_band) - 2.5 * (red_band1 - green_band)) / np.sqrt(np.power(2 * nir_band2+ 1, 2)  - (6 * nir_band2 - 5 * np.sqrt(red_band1)) - 0.5)


###第6组指数特征值：红光波段取2，近红外波段取1
# 计算 IRECI---参照哨兵二号
#ireci6 = np.divide(np.subtract(nir_band2, red_band), (np.divide(data_705, data_750)))
# 计算 CLre
clre6 = np.subtract((np.divide(nir_band2, red_edge_band)), 1)
# 计算 NDRE
ndre6 = np.divide(np.subtract(nir_band2, red_edge_band), np.add(nir_band2, red_edge_band))
# 计算 RNDVI
rndvi6 = np.divide(np.subtract(red_edge_band, red_band2), np.add(red_edge_band, red_band2))
#计算 MSRre
#msrre6 = np.divide(np.subtract(nir_band2, red_edge_band) - 1, np.sqrt(np.subtract(nir_band2, red_edge_band)) + 1)
# 计算 MTCI
#如果分母为0时置为非常小的一个数：在计算去噪加噪的扩展样本8-1中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
# 使用 np.where 替换分母为0或接近0的情况
denominator6 = np.subtract(red_edge_band, red_band2)
denominator_safe6 = np.where(np.abs(denominator6) < 1e-10, 1e-10, denominator6)
# 计算 mtci1
mtci6 = np.divide(np.subtract(nir_band2, red_edge_band), denominator_safe6)
#原始的计算公式
#mtci6 = np.divide(np.subtract(nir_band2, red_edge_band), np.subtract(red_edge_band, red_band2))
#计算 REP---参照哨兵二号
#rep2 = np.add(data_705, 35 * (np.subtract(np.divide((np.add(data_665, data_783)), 2), data_705)) / (np.subtract(data_740, data_705)))
# 计算 NDVI
ndvi6 = np.divide((np.subtract(nir_band2, red_band2)), (np.add(nir_band2, red_band2)))
# 计算 GNDVI
gndvi6 = np.divide((np.subtract(nir_band2, green_band)), (np.add(nir_band2, green_band)))
# 计算 OSAVI
osavi6 = np.divide(np.subtract(nir_band2, red_band2), np.add(nir_band2, red_band2 + 0.16))
# 计算 LCI
lci6 = np.divide(np.subtract(nir_band2, red_edge_band), np.add(nir_band2, red_band2))
# 计算 EVI
evi6 = 2.5 * np.divide(np.subtract(nir_band2, red_band2), np.add(nir_band2, 6 * red_band2 - 7.5 * blue_band + 1))
# 计算 DVI
dvi6 = np.subtract(nir_band2, red_band2)
# 计算 RVI
rvi6 = np.divide(nir_band2, red_band2)
# 计算 SAVI,这里L暂取0.5
savi6 = np.multiply((np.divide((np.subtract(nir_band2, red_band2)), (np.add(nir_band2, red_band2 + 0.5)))), (np.add(1 , 0.5)))
# 计算 MSAVI
#如果根号下有负值则置为0：在计算去噪加噪的扩展样本中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
express2 = np.power(2 * nir_band2 + 1, 2) - 8 * (nir_band2 - red_band2)
express2[express2 < 0] = 0
msavi6 = np.divide(2 * nir_band2 + 1 - np.sqrt(express2), 2)
#原始的计算公式
#msavi6 = np.divide(2 * nir_band2 + 1 - np.sqrt(np.power(2 * nir_band2 + 1, 2) - 8 * (nir_band2 - red_band2)), 2)
# 计算 GCI
gci6 = np.divide(nir_band2, green_band) - 1
# 计算 TVI
tvi6 = np.subtract(60 * np.subtract(nir_band2, green_band), 100 * np.subtract(red_band2, green_band))
# 计算 MTVI2
#如果根号下有负值则置为0：在计算去噪加噪的扩展样本中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
expression6 = np.power(2 * nir_band2 + 1, 2) - (6 * nir_band2 - 5 * np.sqrt(np.abs(red_band2))) - 0.5
expression6[expression6 < 0] = 0
# 防止分母为0的情况，将分母中接近0的值设置为一个很小的数值，以避免除零错误
epsilon6 = 1e-10  # 非常小的值
expression_with_epsilon6 = np.where(expression6 < epsilon6, epsilon6, expression6)
mtvi26 = 1.5 * (1.2 * (nir_band2 - green_band) - 2.5 * (red_band2 - green_band)) / np.sqrt(expression6)
#原始的计算公式
#mtvi26 = 1.5 * (1.2 * (nir_band2 - green_band) - 2.5 * (red_band2 - green_band)) / np.sqrt(np.power(2 * nir_band2+ 1, 2)  - (6 * nir_band2 - 5 * np.sqrt(red_band2)) - 0.5)


###第7组指数特征值：红光波段取3，近红外波段取2
# 计算 IRECI---参照哨兵二号
#ireci7 = np.divide(np.subtract(nir_band2, red_band), (np.divide(data_705, data_750)))
# 计算 CLre
clre7 = np.subtract((np.divide(nir_band2, red_edge_band)), 1)
# 计算 NDRE
ndre7 = np.divide(np.subtract(nir_band2, red_edge_band), np.add(nir_band2, red_edge_band))
# 计算 RNDVI
rndvi7 = np.divide(np.subtract(red_edge_band, red_band3), np.add(red_edge_band, red_band3))
#计算 MSRre
#msrre7 = np.divide(np.subtract(nir_band2, red_edge_band) - 1, np.sqrt(np.subtract(nir_band2, red_edge_band)) + 1)
# 计算 MTCI
#如果分母为0时置为非常小的一个数：在计算去噪加噪的扩展样本8-1中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
# 使用 np.where 替换分母为0或接近0的情况
denominator7 = np.subtract(red_edge_band, red_band3)
denominator_safe7 = np.where(np.abs(denominator7) < 1e-10, 1e-10, denominator7)
# 计算 mtci1
mtci7 = np.divide(np.subtract(nir_band2, red_edge_band), denominator_safe7)
#原始的计算公式
#mtci7 = np.divide(np.subtract(nir_band2, red_edge_band), np.subtract(red_edge_band, red_band3))
#计算 REP---参照哨兵二号
#rep7 = np.add(data_705, 35 * (np.subtract(np.divide((np.add(data_665, data_783)), 2), data_705)) / (np.subtract(data_740, data_705)))
# 计算 NDVI
ndvi7 = np.divide((np.subtract(nir_band2, red_band3)), (np.add(nir_band2, red_band3)))
# 计算 GNDVI
gndvi7 = np.divide((np.subtract(nir_band2, green_band)), (np.add(nir_band2, green_band)))
# 计算 OSAVI
osavi7 = np.divide(np.subtract(nir_band2, red_band3), np.add(nir_band2, red_band3 + 0.16))
# 计算 LCI
lci7 = np.divide(np.subtract(nir_band2, red_edge_band), np.add(nir_band2, red_band3))
# 计算 EVI
evi7 = 2.5 * np.divide(np.subtract(nir_band2, red_band3), np.add(nir_band2, 6 * red_band3 - 7.5 * blue_band + 1))
# 计算 DVI
dvi7 = np.subtract(nir_band2, red_band3)
# 计算 RVI
rvi7 = np.divide(nir_band2, red_band3)
# 计算 SAVI,这里L暂取0.5
savi7 = np.multiply((np.divide((np.subtract(nir_band2, red_band3)), (np.add(nir_band2, red_band3 + 0.5)))), (np.add(1 , 0.5)))
# 计算 MSAVI
#如果根号下有负值则置为0：在计算去噪加噪的扩展样本中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
express8 = np.power(2 * nir_band2 + 1, 2) - 8 * (nir_band2 - red_band3)
express8[express8 < 0] = 0
msavi7 = np.divide(2 * nir_band2 + 1 - np.sqrt(express8), 2)
#原始的计算公式
#msavi7 = np.divide(2 * nir_band2 + 1 - np.sqrt(np.power(2 * nir_band2 + 1, 2) - 8 * (nir_band2 - red_band3)), 2)
# 计算 GCI
gci7 = np.divide(nir_band2, green_band) - 1
# 计算 TVI
tvi7 = np.subtract(60 * np.subtract(nir_band2, green_band), 100 * np.subtract(red_band3, green_band))
# 计算 MTVI2
#如果根号下有负值则置为0：在计算去噪加噪的扩展样本中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
expression7 = np.power(2 * nir_band2 + 1, 2) - (6 * nir_band2 - 5 * np.sqrt(np.abs(red_band3))) - 0.5
expression7[expression7 < 0] = 0
# 防止分母为0的情况，将分母中接近0的值设置为一个很小的数值，以避免除零错误
epsilon7 = 1e-10  # 非常小的值
expression_with_epsilon7 = np.where(expression7 < epsilon7, epsilon7, expression7)
mtvi27 = 1.5 * (1.2 * (nir_band2 - green_band) - 2.5 * (red_band3 - green_band)) / np.sqrt(expression7)
#原始的计算公式
#mtvi27 = 1.5 * (1.2 * (nir_band2 - green_band) - 2.5 * (red_band3 - green_band)) / np.sqrt(np.power(2 * nir_band2+ 1, 2)  - (6 * nir_band2 - 5 * np.sqrt(red_band3)) - 0.5)


###第8组指数特征值：红光波段取4，近红外波段取2
# 计算 IRECI---参照哨兵二号
#ireci8 = np.divide(np.subtract(nir_band2, red_band), (np.divide(data_705, data_750)))
# 计算 CLre
clre8 = np.subtract((np.divide(nir_band2, red_edge_band)), 1)
# 计算 NDRE
ndre8 = np.divide(np.subtract(nir_band2, red_edge_band), np.add(nir_band2, red_edge_band))
# 计算 RNDVI
rndvi8 = np.divide(np.subtract(red_edge_band, red_band4), np.add(red_edge_band, red_band4))
#计算 MSRre
#msrre8 = np.divide(np.subtract(nir_band2, red_edge_band) - 1, np.sqrt(np.subtract(nir_band2, red_edge_band)) + 1)
# 计算 MTCI
#如果分母为0时置为非常小的一个数：在计算去噪加噪的扩展样本8-1中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
# 使用 np.where 替换分母为0或接近0的情况
denominator8 = np.subtract(red_edge_band, red_band4)
denominator_safe8 = np.where(np.abs(denominator8) < 1e-10, 1e-10, denominator8)
# 计算 mtci1
mtci8 = np.divide(np.subtract(nir_band2, red_edge_band), denominator_safe8)
#原始的计算公式
#mtci8 = np.divide(np.subtract(nir_band2, red_edge_band), np.subtract(red_edge_band, red_band4))
#计算 REP---参照哨兵二号
#rep8 = np.add(data_705, 35 * (np.subtract(np.divide((np.add(data_665, data_783)), 2), data_705)) / (np.subtract(data_740, data_705)))
# 计算 NDVI
ndvi8 = np.divide((np.subtract(nir_band2, red_band4)), (np.add(nir_band2, red_band4)))
# 计算 GNDVI
gndvi8 = np.divide((np.subtract(nir_band2, green_band)), (np.add(nir_band2, green_band)))
# 计算 OSAVI
osavi8 = np.divide(np.subtract(nir_band2, red_band4), np.add(nir_band2, red_band4 + 0.16))
# 计算 LCI
lci8 = np.divide(np.subtract(nir_band2, red_edge_band), np.add(nir_band2, red_band4))
# 计算 EVI
evi8 = 2.5 * np.divide(np.subtract(nir_band2, red_band4), np.add(nir_band2, 6 * red_band4 - 7.5 * blue_band + 1))
# 计算 DVI
dvi8 = np.subtract(nir_band2, red_band4)
# 计算 RVI
rvi8 = np.divide(nir_band2, red_band4)
# 计算 SAVI,这里L暂取0.5
savi8 = np.multiply((np.divide((np.subtract(nir_band2, red_band4)), (np.add(nir_band2, red_band4 + 0.5)))), (np.add(1 , 0.5)))
# 计算 MSAVI
#如果根号下有负值则置为0：在计算去噪加噪的扩展样本中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
express6 = np.power(2 * nir_band2 + 1, 2) - 8 * (nir_band2 - red_band4)
express6[express6 < 0] = 0
msavi8 = np.divide(2 * nir_band2 + 1 - np.sqrt(express6), 2)
#原始的计算公式
#msavi8 = np.divide(2 * nir_band2 + 1 - np.sqrt(np.power(2 * nir_band2 + 1, 2) - 8 * (nir_band2 - red_band4)), 2)
# 计算 GCI
gci8 = np.divide(nir_band2, green_band) - 1
# 计算 TVI
tvi8 = np.subtract(60 * np.subtract(nir_band2, green_band), 100 * np.subtract(red_band4, green_band))
# 计算 MTVI2
#如果根号下有负值则置为0：在计算去噪加噪的扩展样本中需要用到，其他时候注释掉
# 重新计算表达式和处理负值
expression8 = np.power(2 * nir_band2 + 1, 2) - (6 * nir_band2 - 5 * np.sqrt(np.abs(red_band4))) - 0.5
expression8[expression8 < 0] = 0
# 防止分母为0的情况，将分母中接近0的值设置为一个很小的数值，以避免除零错误
epsilon8 = 1e-10  # 非常小的值
expression_with_epsilon8 = np.where(expression8 < epsilon8, epsilon8, expression8)
mtvi28 = 1.5 * (1.2 * (nir_band2 - green_band) - 2.5 * (red_band4 - green_band)) / np.sqrt(expression8)
#原始的计算公式
#mtvi28 = 1.5 * (1.2 * (nir_band2 - green_band) - 2.5 * (red_band4 - green_band)) / np.sqrt(np.power(2 * nir_band2+ 1, 2)  - (6 * nir_band2 - 5 * np.sqrt(red_band4)) - 0.5)


# 获取图像的高度和宽度
height = data.shape[0]
width = data.shape[1]

# 创建一个三维数组来存储空间位置和植被指数
vegetation_indices = np.zeros((height, width, 128))

# 将计算得到的植被指数存储到数组中
vegetation_indices[:,:,0] = clre1
#vegetation_indices[:,:,1] = ireci
vegetation_indices[:,:,1] = ndre1
vegetation_indices[:,:,2] = rndvi1
#vegetation_indices[:,:,3] = msrre1
vegetation_indices[:,:,3] = mtci1
vegetation_indices[:,:,4] = ndvi1
vegetation_indices[:,:,5] = gndvi1
vegetation_indices[:,:,6] = osavi1
#vegetation_indices[:,:,9] = rep
vegetation_indices[:,:,7] = lci1
vegetation_indices[:,:,8] = evi1
vegetation_indices[:,:,9] = dvi1
vegetation_indices[:,:,10] = rvi1
vegetation_indices[:,:,11] = savi1
vegetation_indices[:,:,12] = msavi1
vegetation_indices[:,:,13] = gci1
vegetation_indices[:,:,14] = tvi1
vegetation_indices[:,:,15] = mtvi21

vegetation_indices[:,:,16] = clre2
vegetation_indices[:,:,17] = ndre2
vegetation_indices[:,:,18] = rndvi2
#vegetation_indices[:,:,20] = msrre2
vegetation_indices[:,:,19] = mtci2
vegetation_indices[:,:,20] = ndvi2
vegetation_indices[:,:,21] = gndvi2
vegetation_indices[:,:,22] = osavi2
vegetation_indices[:,:,23] = lci2
vegetation_indices[:,:,24] = evi2
vegetation_indices[:,:,25] = dvi2
vegetation_indices[:,:,26] = rvi2
vegetation_indices[:,:,27] = savi2
vegetation_indices[:,:,28] = msavi2
vegetation_indices[:,:,29] = gci2
vegetation_indices[:,:,30] = tvi2
vegetation_indices[:,:,31] = mtvi22

vegetation_indices[:,:,32] = clre3
vegetation_indices[:,:,33] = ndre3
vegetation_indices[:,:,34] = rndvi3
#vegetation_indices[:,:,37] = msrre3
vegetation_indices[:,:,35] = mtci3
vegetation_indices[:,:,36] = ndvi3
vegetation_indices[:,:,37] = gndvi3
vegetation_indices[:,:,38] = osavi3
vegetation_indices[:,:,39] = lci3
vegetation_indices[:,:,40] = evi3
vegetation_indices[:,:,41] = dvi3
vegetation_indices[:,:,42] = rvi3
vegetation_indices[:,:,43] = savi3
vegetation_indices[:,:,44] = msavi3
vegetation_indices[:,:,45] = gci3
vegetation_indices[:,:,46] = tvi3
vegetation_indices[:,:,47] = mtvi23

vegetation_indices[:,:,48] = clre4
vegetation_indices[:,:,49] = ndre4
vegetation_indices[:,:,50] = rndvi4
#vegetation_indices[:,:,54] = msrre4
vegetation_indices[:,:,51] = mtci4
vegetation_indices[:,:,52] = ndvi4
vegetation_indices[:,:,53] = gndvi4
vegetation_indices[:,:,54] = osavi4
vegetation_indices[:,:,55] = lci4
vegetation_indices[:,:,56] = evi4
vegetation_indices[:,:,57] = dvi4
vegetation_indices[:,:,58] = rvi4
vegetation_indices[:,:,59] = savi4
vegetation_indices[:,:,60] = msavi4
vegetation_indices[:,:,61] = gci4
vegetation_indices[:,:,62] = tvi4
vegetation_indices[:,:,63] = mtvi24

vegetation_indices[:,:,64] = clre5
#vegetation_indices[:,:,1] = ireci
vegetation_indices[:,:,65] = ndre5
vegetation_indices[:,:,66] = rndvi5
#vegetation_indices[:,:,71] = msrre5
vegetation_indices[:,:,67] = mtci5
vegetation_indices[:,:,68] = ndvi5
vegetation_indices[:,:,69] = gndvi5
vegetation_indices[:,:,70] = osavi5
#vegetation_indices[:,:,9] = rep
vegetation_indices[:,:,71] = lci5
vegetation_indices[:,:,72] = evi5
vegetation_indices[:,:,73] = dvi5
vegetation_indices[:,:,74] = rvi5
vegetation_indices[:,:,75] = savi5
vegetation_indices[:,:,76] = msavi5
vegetation_indices[:,:,77] = gci5
vegetation_indices[:,:,78] = tvi5
vegetation_indices[:,:,79] = mtvi25

vegetation_indices[:,:,80] = clre6
vegetation_indices[:,:,81] = ndre6
vegetation_indices[:,:,82] = rndvi6
#vegetation_indices[:,:,88] = msrre6
vegetation_indices[:,:,83] = mtci6
vegetation_indices[:,:,84] = ndvi6
vegetation_indices[:,:,85] = gndvi6
vegetation_indices[:,:,86] = osavi6
vegetation_indices[:,:,87] = lci6
vegetation_indices[:,:,88] = evi6
vegetation_indices[:,:,89] = dvi6
vegetation_indices[:,:,90] = rvi6
vegetation_indices[:,:,91] = savi6
vegetation_indices[:,:,92] = msavi6
vegetation_indices[:,:,93] = gci6
vegetation_indices[:,:,94] = tvi6
vegetation_indices[:,:,95] = mtvi26

vegetation_indices[:,:,96] = clre7
vegetation_indices[:,:,97] = ndre7
vegetation_indices[:,:,98] = rndvi7
#vegetation_indices[:,:,105] = msrre7
vegetation_indices[:,:,99] = mtci7
vegetation_indices[:,:,100] = ndvi7
vegetation_indices[:,:,101] = gndvi7
vegetation_indices[:,:,102] = osavi7
vegetation_indices[:,:,103] = lci7
vegetation_indices[:,:,104] = evi7
vegetation_indices[:,:,105] = dvi7
vegetation_indices[:,:,106] = rvi7
vegetation_indices[:,:,107] = savi7
vegetation_indices[:,:,108] = msavi7
vegetation_indices[:,:,109] = gci7
vegetation_indices[:,:,110] = tvi7
vegetation_indices[:,:,111] = mtvi27

vegetation_indices[:,:,112] = clre8
vegetation_indices[:,:,113] = ndre8
vegetation_indices[:,:,114] = rndvi8
#vegetation_indices[:,:,122] = msrre8
vegetation_indices[:,:,115] = mtci8
vegetation_indices[:,:,116] = ndvi8
vegetation_indices[:,:,117] = gndvi8
vegetation_indices[:,:,118] = osavi8
vegetation_indices[:,:,119] = lci8
vegetation_indices[:,:,120] = evi8
vegetation_indices[:,:,121] = dvi8
vegetation_indices[:,:,122] = rvi8
vegetation_indices[:,:,123] = savi8
vegetation_indices[:,:,124] = msavi8
vegetation_indices[:,:,125] = gci8
vegetation_indices[:,:,126] = tvi8
vegetation_indices[:,:,127] = mtvi28
# 可以通过索引来访问特定的植被指数值，例如，要访问NDVI在位置(100, 50)处的值：
#ndvi_value = vegetation_indices[100, 50, 0]
#print("NDVI value at position (100, 50):", ndvi_value)

# 将植被指数保存到文件中，可以使用类似以下代码：
file_path = 'D:/Python_Projects/9data-enhance/enhance_samples+14samples/e-feature/025_cl_e.npy'
print(vegetation_indices.shape)#(160, 160, 128)
np.save(file_path, vegetation_indices)

print("Vegetation indices calculation completed and saved!")