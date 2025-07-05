import os
import cv2
import numpy as np

def convert_hdrs_to_npy(input_folder, output_folder):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有 .hdr 文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".hdr"):
            hdr_path = os.path.join(input_folder, filename)
            npy_filename = os.path.splitext(filename)[0] + ".npy"
            npy_path = os.path.join(output_folder, npy_filename)

            print(f"Reading: {hdr_path}")
            # 读取 HDR 图像
            hdr_image = cv2.imread(hdr_path,  cv2.IMREAD_UNCHANGED)

            if hdr_image is not None:
                # 保存为 .npy 文件
                np.save(npy_path, hdr_image)
                print(f"Saved: {npy_path}")
            else:
                print(f"Failed to read HDR file: {hdr_path}")

if __name__ == "__main__":
    # 设置你的输入和输出文件夹路径
    input_folder = "/data2/wangzihao/singlehdr-pytorch/outputtest"
    output_folder = "/data2/wangzihao/singlehdr-pytorch/testnpy"

    convert_hdrs_to_npy(input_folder, output_folder)