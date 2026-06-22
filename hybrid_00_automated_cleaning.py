import os
import shutil
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

try:
    from transformers import AutoModel
except ImportError:
    print("❌ ไม่พบไลบรารี transformers")
    exit(1)

class SimpleImageDataset(Dataset):
    def __init__(self, file_paths, transform):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        path = self.file_paths[index]
        try:
            img = Image.open(path).convert('RGB')
            tensor = self.transform(img)
            return tensor, path, True
        except Exception as e:
            return torch.zeros(3, 224, 224), path, False

def get_green_ratio(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0.0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_pixels = np.sum(mask > 0)
        total_pixels = img.shape[0] * img.shape[1]
        return float(green_pixels) / total_pixels
    except:
        return 0.0

def main():
    print("🌟 เริ่มกระบวนการ Hybrid 00: Automated Data Cleaning (HF Foundation Model) 🌟")
    
    source_dir = "Data200"
    clean_dir = "Data200_Cleaned"
    reject_dir = "Data200_Rejected"
    
    if not os.path.exists(source_dir):
        print(f"❌ ไม่พบโฟลเดอร์ {source_dir}")
        return

    os.makedirs(clean_dir, exist_ok=True)
    outliers_reject_dir = os.path.join(reject_dir, "outliers")
    zoomed_reject_dir = os.path.join(reject_dir, "zoomed_out")
    os.makedirs(outliers_reject_dir, exist_ok=True)
    os.makedirs(zoomed_reject_dir, exist_ok=True)

    classes = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])
    print(f"พบทั้งสิ้น {len(classes)} คลาส")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"รันบนอุปกรณ์: {device}")
    
    print("[HuggingFace] กำลังโหลดโมเดล SOTA (facebook/dinov2-large) สำหรับใช้เป็นเรดาร์ตรวจจับภาพขยะ...")
    # ไม่ต้องโหลดน้ำหนัก Fine-tune เพราะเป้าหมายแค่แยก Outlier โมเดล Base ระดับ Large ฉลาดพอที่จะแยกขยะได้ทันที
    encoder = AutoModel.from_pretrained('facebook/dinov2-large')
    encoder.to(device)
    encoder.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    total_cleaned = 0
    total_outliers = 0
    total_zoomed_out = 0

    for class_name in classes:
        dest_class_dir = os.path.join(clean_dir, class_name)
        processed_marker = os.path.join(dest_class_dir, ".processed")
        
        if os.path.exists(processed_marker):
            print(f"\n✅ ข้ามคลาส: {class_name} (ประมวลผลเสร็จไปแล้ว)")
            total_cleaned += len([f for f in os.listdir(dest_class_dir) if f != '.processed'])
            continue
            
        print(f"\n📂 กำลังตรวจสอบคลาส: {class_name}")
        
        if os.path.exists(dest_class_dir):
            shutil.rmtree(dest_class_dir)
            
        class_path = os.path.join(source_dir, class_name)
        
        file_paths = []
        for root, _, files in os.walk(class_path):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_paths.append(os.path.join(root, f))
                    
        if len(file_paths) == 0:
            continue
            
        dataset = SimpleImageDataset(file_paths, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        features_list = []
        valid_paths = []
        
        with torch.inference_mode():
            for inputs, paths, valids in tqdm(dataloader, desc=f"Extracting Features", leave=False):
                inputs = inputs.to(device)
                outputs = encoder(pixel_values=inputs)
                features = outputs.pooler_output
                features = torch.nn.functional.normalize(features, dim=1).cpu().numpy()
                
                for i in range(len(paths)):
                    if valids[i]:
                        features_list.append(features[i])
                        valid_paths.append(paths[i])
                    else:
                        filename = os.path.basename(paths[i])
                        shutil.copy2(paths[i], os.path.join(outliers_reject_dir, f"corrupt_{class_name}_{filename}"))
                        total_outliers += 1
                        
        if len(features_list) == 0:
            continue
            
        features_arr = np.vstack(features_list)
        centroid = np.mean(features_arr, axis=0)
        centroid = centroid / np.linalg.norm(centroid) 
        
        similarities = np.dot(features_arr, centroid)
        
        os.makedirs(dest_class_dir, exist_ok=True)
        
        for idx, filepath in enumerate(valid_paths):
            sim = similarities[idx]
            filename = os.path.basename(filepath)
            
            green_ratio = get_green_ratio(filepath)
            
            # เกณฑ์จำแนก
            if sim < 0.68:
                target_path = os.path.join(outliers_reject_dir, f"outlier_sim{sim:.2f}_{class_name}_{filename}")
                shutil.copy2(filepath, target_path)
                total_outliers += 1
            elif sim < 0.78 and green_ratio < 0.18:
                target_path = os.path.join(zoomed_reject_dir, f"zoom_sim{sim:.2f}_gr{green_ratio:.2f}_{class_name}_{filename}")
                shutil.copy2(filepath, target_path)
                total_zoomed_out += 1
            else:
                target_path = os.path.join(dest_class_dir, filename)
                shutil.copy2(filepath, target_path)
                total_cleaned += 1

        with open(processed_marker, 'w') as f:
            f.write('done')

    print("\n================================================")
    print(f"🎉 กระบวนการ Automated Data Cleaning เสร็จสิ้น!")
    print(f"✅ ย้ายข้อมูลสะอาดไปที่ {clean_dir}: {total_cleaned} รูป")
    print(f"🗑️ คัดแยก Outliers (รูปขยะ): {total_outliers} รูป")
    print(f"🔍 คัดแยกภาพซูมไกล (Zoomed-out): {total_zoomed_out} รูป")
    print("================================================")

if __name__ == "__main__":
    main()
