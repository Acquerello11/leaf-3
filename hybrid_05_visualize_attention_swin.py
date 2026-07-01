import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib.util

spec = importlib.util.spec_from_file_location("hybrid3_swin", "hybrid_03_finetune_swin.py")
hybrid3_swin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hybrid3_swin)
SwinVisionFinetuner = hybrid3_swin.SwinVisionFinetuner

# ====================================================
IMAGE_PATH_TO_TEST = "" 
# ====================================================

def get_prediction_knn(model, test_image_tensor, train_dir, device):
    print("\n🧠 กำลังโหลดคลังสมอง (Train Set) เพื่อนำมาทายผล...")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = datasets.ImageFolder(train_dir, transform=transform)
    classes = dataset.classes
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    train_features = []
    train_labels = []
    
    with torch.inference_mode():
        for inputs, labels in tqdm(dataloader, desc="สร้างคลังสมองชั่วคราว"):
            inputs = inputs.to(device)
            with torch.amp.autocast('cuda'):
                features = model.get_features(inputs)
            features = F.normalize(features, dim=1)
            train_features.append(features.cpu())
            train_labels.append(labels)
            
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    
    with torch.inference_mode():
        with torch.amp.autocast('cuda'):
            test_feature = model.get_features(test_image_tensor)
        test_feature = F.normalize(test_feature, dim=1).cpu()
        
        sim_matrix = torch.matmul(test_feature, train_features.T)
        best_match_idx = torch.argmax(sim_matrix, dim=1).item()
        confidence = torch.max(sim_matrix).item() * 100
        
        pred_label_idx = train_labels[best_match_idx].item()
        pred_class_name = classes[pred_label_idx]
        
    return pred_class_name, confidence

def generate_attention_map(model, image_path, device):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.inference_mode():
        with torch.amp.autocast('cuda'):
            features = model.encoder.forward_features(input_tensor)
            
    spatial_feat = features[0].cpu().numpy()
    attention_grid = np.linalg.norm(spatial_feat, axis=-1)
    attention_grid = (attention_grid - attention_grid.min()) / (attention_grid.max() - attention_grid.min() + 1e-8)
    attention_grid = cv2.resize(attention_grid, (256, 256))
    
    img_resized = np.array(img.resize((256, 256)))
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_grid), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    alpha = 0.5
    superimposed = cv2.addWeighted(img_resized, alpha, heatmap, 1 - alpha, 0)
    
    return input_tensor, img_resized, heatmap, superimposed

def main():
    print("🌟 โหมดตรวจสอบรูปภาพรายบุคคล (Single Image Tester & Attention) - SwinV2 🌟")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = "hybrid_swin_vision_finetuned.pth"
    train_dir = "Data200_Segmented_Split/train"
    
    if not os.path.exists(weights_path) or not os.path.exists(train_dir):
        print(f"❌ ไม่พบไฟล์น้ำหนัก {weights_path} หรือโฟลเดอร์ {train_dir}")
        return
        
    print("📥 กำลังโหลดน้ำหนักโมเดล (SwinV2 Engine)...")
    model = SwinVisionFinetuner().to(device)
    model.encoder.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    target_path = IMAGE_PATH_TO_TEST
    
    if target_path == "":
        print("ℹ️ ไม่ได้ระบุชื่อไฟล์ จะทำการสุ่มจากโฟลเดอร์ Test แทน...")
        test_dir = "Data200_Segmented_Split/test"
        classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
        if len(classes) == 0: return
        target_class = random.choice(classes)
        target_dir = os.path.join(test_dir, target_class)
        files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(files) == 0: return
        target_path = os.path.join(target_dir, random.choice(files))
        
    if not os.path.exists(target_path):
        print(f"❌ ไม่พบไฟล์รูปภาพ: {target_path}")
        return
        
    print(f"\n🔍 กำลังสแกนรูป: {target_path}")
    
    input_tensor, img_original, heatmap, superimposed = generate_attention_map(model, target_path, device)
    pred_class, confidence = get_prediction_knn(model, input_tensor, train_dir, device)
    
    print(f"\n🎯 คำทำนายของโมเดล: {pred_class} (ความคล้ายคลึง: {confidence:.2f}%)")
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_original)
    plt.title(f'Input Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title('Attention Heatmap (Red = Focus)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed)
    plt.title(f'Predicted: {pred_class}\nConf: {confidence:.2f}%')
    plt.axis('off')
    
    out_path = 'hybrid_attention_result_swin.png'
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"✅ บันทึกภาพผลลัพธ์ไว้ที่: '{out_path}' เรียบร้อยแล้ว!")

if __name__ == "__main__":
    main()
