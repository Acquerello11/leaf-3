import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import importlib.util

# โหลดคลาส SwinVisionFinetuner แบบไดนามิก
spec = importlib.util.spec_from_file_location("hybrid3_swin", "hybrid_03_finetune_swin.py")
hybrid3_swin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hybrid3_swin)
SwinVisionFinetuner = hybrid3_swin.SwinVisionFinetuner

class HybridEvaluatorSwin:
    def __init__(self, weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Hybrid Evaluator] ทำงานบนอุปกรณ์: {self.device}")
        
        print("📥 กำลังโหลดน้ำหนักโมเดล Swin Transformer (V2) ที่ผ่านการเทรนมาแล้ว...")
        self.model = SwinVisionFinetuner().to(self.device)
        if os.path.exists(weights_path):
            self.model.encoder.load_state_dict(torch.load(weights_path, map_location=self.device))
            print("✅ โหลดน้ำหนักสำเร็จ")
        else:
            print("⚠️ ไม่พบน้ำหนักที่เทรน (กำลังใช้ Pre-trained ดั้งเดิม)")
        self.model.eval()
        
        # SwinV2 ใช้ภาพ 256x256
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.kb_features = None
        self.kb_labels = None
        self.classes = []

    def build_knowledge_base(self, train_dir):
        print("\n🧠 กำลังสกัดความรู้ (Feature Extraction) จากข้อมูล Train 70% ...")
        dataset = datasets.ImageFolder(train_dir, transform=self.transform)
        self.classes = dataset.classes
        
        # เพิ่ม batch_size และ num_workers ให้เร็วขึ้น
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)
        
        all_features = []
        all_labels = []
        
        with torch.inference_mode():
            for inputs, labels in tqdm(dataloader, desc="สร้างคลังสมอง (Knowledge Base)"):
                inputs = inputs.to(self.device)
                with torch.amp.autocast('cuda'):
                    features = self.model.get_features(inputs)
                features = F.normalize(features, dim=1) 
                
                all_features.append(features.cpu())
                all_labels.append(labels)
                
        self.kb_features = torch.cat(all_features, dim=0).to(self.device)
        self.kb_labels = torch.cat(all_labels, dim=0).to(self.device)
        print("✅ สร้างคลังสมองสำเร็จ พร้อมสำหรับการทำนาย!")

    def evaluate(self, test_dir):
        print("\n🔍 กำลังรันโมเดลทำนายผล (Evaluation) บนข้อมูล Test 15%...")
        dataset = datasets.ImageFolder(test_dir, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)
        
        true_labels = []
        predicted_labels = []
        
        with torch.inference_mode():
            for inputs, labels in tqdm(dataloader, desc="ประเมินผล Test Set"):
                inputs = inputs.to(self.device)
                
                with torch.amp.autocast('cuda'):
                    test_features = self.model.get_features(inputs)
                test_features = F.normalize(test_features, dim=1)
                
                # คิดเลขบน GPU เร็วมาก
                similarity_matrix = torch.matmul(test_features, self.kb_features.T)
                best_match_indices = torch.argmax(similarity_matrix, dim=1)
                pred_labels = self.kb_labels[best_match_indices]
                
                true_labels.extend(labels.numpy())
                predicted_labels.extend(pred_labels.cpu().numpy())
                
        true_names = [self.classes[i] for i in true_labels]
        pred_names = [self.classes[i] for i in predicted_labels]
        
        return true_names, pred_names

def main():
    print("🌟 เข้าสู่โหมดประเมินประสิทธิภาพระบบ (System Evaluation - SwinV2 Engine) 🌟")
    
    train_dir = "Data200_Segmented_Split/train"
    test_dir = "Data200_Segmented_Split/test"
    weights_path = "hybrid_swin_vision_finetuned.pth"
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"❌ ไม่พบไฟล์ที่จำเป็น กรุณาตรวจสอบว่ามีโฟลเดอร์ {train_dir} และ {test_dir} หรือไม่")
        return
        
    evaluator = HybridEvaluatorSwin(weights_path)
    evaluator.build_knowledge_base(train_dir)
    true_labels, predicted_labels = evaluator.evaluate(test_dir)
    
    correct = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)
    overall_accuracy = (correct / len(true_labels)) * 100
    print(f"\nประมวลผลเสร็จสิ้น {len(true_labels)} ภาพ")
    print(f"*** ความแม่นยำสุทธิของระบบ (Overall Accuracy): {overall_accuracy:.2f}% ***")
    
    print("\n--- รายงานเชิงลึก (Classification Report) ---")
    report_text = classification_report(true_labels, predicted_labels, target_names=evaluator.classes)
    print(report_text)
    
    report_dict = classification_report(true_labels, predicted_labels, target_names=evaluator.classes, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    csv_path = 'hybrid_swin_evaluation_report.csv'
    df_report.to_csv(csv_path)
    print(f"✅ บันทึกรายงานสถิติลงไฟล์ {csv_path} เรียบร้อยแล้ว")
    
    print("\nกำลังสร้างกราฟ Confusion Matrix...")
    # Map back to integer indices for confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=evaluator.classes)
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=False, cmap='Oranges', xticklabels=evaluator.classes, yticklabels=evaluator.classes)
    plt.title('Confusion Matrix - SwinV2 Transformer Engine')
    plt.xlabel('Predicted Species')
    plt.ylabel('True Species')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    img_path = 'hybrid_swin_confusion_matrix_result.png'
    plt.savefig(img_path)
    print(f"✅ บันทึกกราฟลงไฟล์ '{img_path}' เรียบร้อยแล้ว!")

if __name__ == "__main__":
    main()
