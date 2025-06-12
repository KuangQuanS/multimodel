import torch
from resgsca import CTModel, generate_attention_visualization
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

def visualize_attention_demo(image_path, model_path):
    """
    简单的注意力可视化演示
    
    参数:
        image_path: CT图像路径(.npz文件)
        model_path: 模型权重文件路径
    """
    # 1. 加载模型
    model = CTModel()  # 使用默认参数
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 2. 加载和预处理图像
    # 加载.npz文件中的CT图像
    ct_data = np.load(image_path)['CT'][0,:,:]
    
    # 归一化到0-1范围
    ct_image = (ct_data - ct_data.min()) / (ct_data.max() - ct_data.min())
    
    # 转换为3通道图像（因为模型期望RGB输入）
    ct_image_3ch = np.stack([ct_image] * 3, axis=0)
    
    # 转换为tensor
    image_tensor = torch.FloatTensor(ct_image_3ch)
    
    # 调整大小到模型输入尺寸
    resize = transforms.Resize((224, 224))
    image_tensor = resize(image_tensor.unsqueeze(0)).squeeze(0)
    
    # 3. 生成注意力可视化
    # 保存原始CT图像用于显示
    original_image = np.stack([ct_image] * 3, axis=-1)  # 转换为3通道便于显示
    
    overlaid_images, _ = generate_attention_visualization(
        model=model,
        image_tensor=image_tensor,
        original_image=original_image,
        save_path='attention_result.png',  # 结果将保存到这个文件
        alpha=0.5  # 注意力图的透明度
    )
    
    print("注意力可视化已保存到 'attention_result.png'")
    
    # 4. 显示结果
    plt.figure(figsize=(15, 5))
    
    # 显示原始CT图像
    plt.subplot(1, 3, 1)
    plt.imshow(ct_image, cmap='gray')
    plt.title('Original CT Image')
    plt.axis('off')
    
    # 显示3通道版本
    plt.subplot(1, 3, 2)
    plt.imshow(original_image)
    plt.title('3-Channel CT Image')
    plt.axis('off')
    
    # 显示第一个注意力图（通常是最重要的一个）
    plt.subplot(1, 3, 3)
    plt.imshow(overlaid_images[0])
    plt.title('Attention Visualization')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 替换这些路径为您的实际路径
    IMAGE_PATH = "path/to/your/ct_image.npz"  # 您的CT图像路径（.npz文件）
    MODEL_PATH = "path/to/your/model.pth"     # 您的模型权重文件路径
    
    visualize_attention_demo(IMAGE_PATH, MODEL_PATH)