# 保存为preprocess_calibration.py，在宿主机执行
from PIL import Image
import os

input_dir = r"G:\项目\基于Jetson nano的智能车\yolo\detect_datasets"
output_dir = r"G:\项目\基于Jetson nano的智能车\yolo\detect_datasets_processed"
os.makedirs(output_dir, exist_ok=True)

target_size = (640, 640)

for img_name in os.listdir(input_dir):
    if img_name.endswith(('.jpg', '.jpeg', '.png')):
        try:
            # 打开图像并转换为RGB
            img = Image.open(os.path.join(input_dir, img_name)).convert('RGB')

            # 调整尺寸（保持纵横比，用黑色填充边缘）
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            new_img = Image.new('RGB', target_size, (0, 0, 0))
            new_img.paste(img, ((target_size[0] - img.size[0]) // 2,
                                (target_size[1] - img.size[1]) // 2))

            # 保存处理后的图像（强制为JPEG格式）
            output_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + '.jpg')
            new_img.save(output_path, 'JPEG', quality=95)
            print(f"已处理: {img_name} -> {output_path}")
        except Exception as e:
            print(f"处理失败 {img_name}: {e}")