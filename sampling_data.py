import pandas as pd
import shutil
import os

# 读取 CSV 文件
df = pd.read_csv('driver_imgs_list.csv')

# 创建目标文件夹，如果不存在
output_folder = 'data_part_80'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历 c0 到 c9 的类别
for class_label in [f'c{i}' for i in range(10)]:
    # 过滤出当前类别的数据
    if class_label == "c1":
        class_data = df[(df['classname'] == "c1") | (df['classname'] == "c3")]
    elif class_label == "c2":
        class_data = df[(df['classname'] == "c2") | (df['classname'] == "c4")]
    elif class_label in ["c3", "c4"]:
        continue
    else:
        class_data = df[df['classname'] == class_label]
    
    # 随机采样 n 个
    sampled_data = class_data.sample(n=10, replace=False)
    
    # 遍历采样的数据并复制文件，如果是c5-c9则变为c3-c7，对应的class_label要拆开然后取数字部分-2
    for index, row in enumerate(sampled_data.itertuples(), start=1):
        src_file = os.path.join('imgs/train', row.classname, row.img)
        print(src_file)
        new_class_label = f'c{int(class_label[1]) if int(class_label[1]) < 5 else int(class_label[1]) - 2}'
        dst_file = os.path.join(output_folder, f'{new_class_label}_{index:04d}.jpg')
        shutil.copy(src_file, dst_file)

print("文件复制和重命名完成")
