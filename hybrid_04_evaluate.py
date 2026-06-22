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

# โหลดคลาส HFVisionFinetuner แบบไดนามิก
spec = importlib.util.spec_from_file_location("hybrid3", "hybrid_03_finetune_dino.py")
hybrid3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hybrid3)
HFVisionFinetuner = hybrid3.HFVisionFinetuner

class HybridEvaluator:
    def __init__(self, weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Hybrid Evaluator] ทำงานบนอุปกรณ์: {self.device}")
        
        print("📥 กำลังโหลดน้ำหนักโมเดล SOTA (HuggingFace) ที่ผ่านการเทรนมาแล้ว...")
        self.model = HFVisionFinetuner().to(self.device)
        self.model.encoder.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
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
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        all_features = []
        all_labels = []
        
        with torch.inference_mode():
            for inputs, labels in tqdm(dataloader, desc="สร้างคลังสมอง (Knowledge Base)"):
                inputs = inputs.to(self.device)
                features = self.model.get_features(inputs)
                features = F.normalize(features, dim=1) 
                
                all_features.append(features.cpu())
                all_labels.append(labels)
                
        self.kb_features = torch.cat(all_features, dim=0)
        self.kb_labels = torch.cat(all_labels, dim=0)
        print("✅ สร้างคลังสมองสำเร็จ พร้อมสำหรับการทำนาย!")

    def evaluate(self, test_dir):
        print("\n🔍 กำลังรันโมเดลทำนายผล (Evaluation) บนข้อมูล Test 15%...")
        dataset = datasets.ImageFolder(test_dir, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        true_labels = []
        predicted_labels = []
        
        with torch.inference_mode():
            for inputs, labels in tqdm(dataloader, desc="ประเมินผล Test Set"):
                inputs = inputs.to(self.device)
                
                test_features = self.model.get_features(inputs)
                test_features = F.normalize(test_features, dim=1)
                
                similarity_matrix = torch.matmul(test_features, self.kb_features.T)
                best_match_indices = torch.argmax(similarity_matrix, dim=1)
                pred_labels = self.kb_labels[best_match_indices.cpu()]
                
                true_labels.extend(labels.numpy())
                predicted_labels.extend(pred_labels.numpy())
                
        true_names = [self.classes[i] for i in true_labels]
        pred_names = [self.classes[i] for i in predicted_labels]
        
        return true_names, pred_names

def main():
    print("🌟 เข้าสู่โหมดประเมินประสิทธิภาพระบบ (System Evaluation - HF Engine) 🌟")
    
    train_dir = "Data200_Hybrid_Split/train"
    test_dir = "Data200_Hybrid_Split/test"
    weights_path = "hybrid_hf_vision_finetuned.pth"
    
    if not os.path.exists(weights_path) or not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"❌ ไม่พบไฟล์ที่จำเป็น กรุณาตรวจสอบให้แน่ใจว่ารันโค้ด hybrid ไฟล์ 2 และ 3 ครบถ้วนแล้ว")
        return
        
    evaluator = HybridEvaluator(weights_path)
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
    csv_path = 'hybrid_hf_evaluation_report.csv'
    df_report.to_csv(csv_path)
    print(f"✅ บันทึกรายงานสถิติลงไฟล์ {csv_path} เรียบร้อยแล้ว")
    
    print("\nกำลังสร้างกราฟ Confusion Matrix...")
    cm = confusion_matrix(true_labels, predicted_labels, labels=evaluator.classes)
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=evaluator.classes, yticklabels=evaluator.classes)
    plt.title('Confusion Matrix - SOTA Foundation Model (HF Engine)')
    plt.xlabel('Predicted Species')
    plt.ylabel('True Species')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    img_path = 'hybrid_hf_confusion_matrix_result.png'
    plt.savefig(img_path)
    print(f"✅ บันทึกกราฟลงไฟล์ '{img_path}' เรียบร้อยแล้ว!")

if __name__ == "__main__":
    main()
