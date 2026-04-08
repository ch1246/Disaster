import cv2
import os
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops


main_folder = r"C:\Users\26093\Desktop\Comprehensive Disaster Dataset(CDD)"

# 存储所有特征的列表（直接使用列表收集，避免循环内DataFrame拼接）
feature_list = []


for root, dirs, files in os.walk(main_folder):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):  # 支持更多图像格式
            image_path = os.path.join(root, file)
            
  
            image = cv2.imread(image_path)
            if image is None:
                print(f"警告：无法读取图像文件 {image_path}，可能文件损坏或非图像格式，跳过该文件")
                continue  # 跳过当前损坏或无法读取的文件
            
     
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 计算H/S/V通道直方图（8个bins）
            hist_h = cv2.calcHist([hsv_image], [0], None, [8], [0, 180]).flatten()
            hist_s = cv2.calcHist([hsv_image], [1], None, [8], [0, 256]).flatten()
            hist_v = cv2.calcHist([hsv_image], [2], None, [8], [0, 256]).flatten()
            
    
            total_pixels = image.shape[0] * image.shape[1]
            hist_h_normalized = hist_h / total_pixels if total_pixels != 0 else hist_h
            hist_s_normalized = hist_s / total_pixels if total_pixels != 0 else hist_s
            hist_v_normalized = hist_v / total_pixels if total_pixels != 0 else hist_v
            
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
            
           
            glcm = graycomatrix(
                gray_image,
                distances=[1],  # 距离为1
                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],  # 四个主要方向
                levels=256,
                symmetric=True,
                normed=True
            )
            
            # 计算各纹理特征并取平均值（多方向融合）
            contrast = graycoprops(glcm, 'contrast').mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            
         
            moments = cv2.moments(gray_image)
            hu_moments = cv2.HuMoments(moments).flatten()  # 直接获取7个Hu矩
            
          
            feature_dict = {
                'image_path': image_path,
                # H通道直方图（归一化后）
                'hue_hist_0': hist_h_normalized[0], 'hue_hist_1': hist_h_normalized[1],
                'hue_hist_2': hist_h_normalized[2], 'hue_hist_3': hist_h_normalized[3],
                'hue_hist_4': hist_h_normalized[4], 'hue_hist_5': hist_h_normalized[5],
                'hue_hist_6': hist_h_normalized[6], 'hue_hist_7': hist_h_normalized[7],
                # S通道直方图（归一化后）
               'saturation_hist_0': hist_s_normalized[0],'saturation_hist_1': hist_s_normalized[1],
               'saturation_hist_2': hist_s_normalized[2],'saturation_hist_3': hist_s_normalized[3],
               'saturation_hist_4': hist_s_normalized[4],'saturation_hist_5': hist_s_normalized[5],
               'saturation_hist_6': hist_s_normalized[6],'saturation_hist_7': hist_s_normalized[7],
                # V通道直方图（归一化后）
                'value_hist_0': hist_v_normalized[0], 'value_hist_1': hist_v_normalized[1],
                'value_hist_2': hist_v_normalized[2], 'value_hist_3': hist_v_normalized[3],
                'value_hist_4': hist_v_normalized[4], 'value_hist_5': hist_v_normalized[5],
                'value_hist_6': hist_v_normalized[6], 'value_hist_7': hist_v_normalized[7],
                # 纹理特征
                'contrast': contrast, 'dissimilarity': dissimilarity,
                'homogeneity': homogeneity, 'energy': energy, 'correlation': correlation,
                # Hu矩特征
                'hu_moment_0': hu_moments[0], 'hu_moment_1': hu_moments[1],
                'hu_moment_2': hu_moments[2], 'hu_moment_3': hu_moments[3],
                'hu_moment_4': hu_moments[4], 'hu_moment_5': hu_moments[5],
                'hu_moment_6': hu_moments[6]
            }
            
            feature_list.append(feature_dict)  # 添加到特征列表
            print(f"已处理文件：{image_path}")  # 输出进度，避免假死
            
# ---------------------- 生成最终特征数据框并保存 ----------------------
feature_df = pd.DataFrame(feature_list)
feature_df.to_csv('disaster_image_features.csv', index=False)
print(f"特征提取完成，共处理 {len(feature_list)} 张图像，结果已保存到 disaster_image_features.csv")