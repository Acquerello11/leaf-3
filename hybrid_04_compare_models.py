import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def build_knowledge_base(model, loader, device):
    """สร้างคลังสมอง k-NN"""
    features_list = []
    labels_list = []
    with torch.inference_mode():
        for inputs, labels in tqdm(loader, desc="Building KB"):
            inputs = inputs.to(device)
            features = model.get_features(inputs)
            features = F.normalize(features, dim=1)
            features_list.append(features.cpu())
            labels_list.append(labels)
            
    return torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0)

def evaluate_model(model, val_loader, train_features, train_labels, device):
    """ทำข้อสอบและวัดผล"""
    correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            val_features = model.get_features(inputs)
            val_features = F.normalize(val_features, dim=1).cpu()
            
            sim_matrix = torch.matmul(val_features, train_features.T)
            best_match_indices = torch.argmax(sim_matrix, dim=1)
            preds = train_labels[best_match_indices]
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return (correct / total) * 100

def generate_dino_heatmap(model_dino, img_tensor, device):
    with torch.inference_mode():
        outputs = model_dino.encoder(pixel_values=img_tensor.to(device), output_attentions=True)
    attentions = outputs.attentions[-1]
    cls_attention = attentions[0, :, 0, 1:]
    mean_attention = torch.mean(cls_attention, dim=0).cpu().numpy()
    grid_size = int(np.sqrt(mean_attention.shape[0]))
    attention_grid = mean_attention.reshape(grid_size, grid_size)
    attention_grid = (attention_grid - attention_grid.min()) / (attention_grid.max() - attention_grid.min() + 1e-8)
    return cv2.resize(attention_grid, (224, 224))

def generate_swin_heatmap(model_swin, img_tensor, device):
    with torch.inference_mode():
        features = model_swin.encoder.forward_features(img_tensor.to(device))
    # features shape: (1, 7, 7, 1536)
    spatial_feat = features[0].cpu().numpy() # (7, 7, 1536)
    attention_grid = np.linalg.norm(spatial_feat, axis=-1)
    attention_grid = (attention_grid - attention_grid.min()) / (attention_grid.max() - attention_grid.min() + 1e-8)
    return cv2.resize(attention_grid, (224, 224))

def create_heatmap_overlay(img_np, heatmap_grid):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap_grid), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

def main():
    print("🌟 เริ่มกระบวนการ Stage 4: เปรียบเทียบสมรรถนะ (Model Comparison) 🌟")
    print("ศึกประชันความแม่นยำระหว่าง DINOv2 (Meta) vs Swin Transformer (Microsoft)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"รันบนอุปกรณ์: {device}\n")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dir = "Data200_Hybrid_Split/train"
    val_dir = "Data200_Hybrid_Split/val"
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"❌ ไม่พบข้อมูล {train_dir}")
        return

    eval_train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    
    # DataLoader
    eval_train_loader = DataLoader(eval_train_dataset, batch_size=64, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    results = {}
    
    # ------------------ ทดสอบ DINOv2 ------------------
    try:
        print("▶️ กำลังทดสอบผู้เข้าแข่งขันคนที่ 1: DINOv2 (Meta)")
        hybrid3 = load_module("hybrid3", "hybrid_03_finetune_dino.py")
        model_dino = hybrid3.HFVisionFinetuner().to(device)
        dino_weight = 'hybrid_hf_vision_finetuned.pth'
        if os.path.exists(dino_weight):
            model_dino.encoder.load_state_dict(torch.load(dino_weight, map_location=device))
            print("   ✅ โหลดน้ำหนักที่เทรนแล้วสำเร็จ")
        else:
            print("   ⚠️ ไม่พบน้ำหนักที่เทรน (กำลังใช้ Pre-trained ดั้งเดิม)")
            
        model_dino.eval()
        dino_train_feat, dino_train_labels = build_knowledge_base(model_dino, eval_train_loader, device)
        dino_acc = evaluate_model(model_dino, val_loader, dino_train_feat, dino_train_labels, device)
        results["DINOv2"] = dino_acc
        print(f"   🏆 ความแม่นยำ DINOv2: {dino_acc:.2f}%\n")
    except Exception as e:
        print(f"   ❌ ทดสอบ DINOv2 ล้มเหลว: {e}\n")

    # ------------------ ทดสอบ Swin Transformer ------------------
    try:
        print("▶️ กำลังทดสอบผู้เข้าแข่งขันคนที่ 2: Swin Transformer (Microsoft)")
        hybrid3_swin = load_module("hybrid3_swin", "hybrid_03_finetune_swin.py")
        model_swin = hybrid3_swin.SwinVisionFinetuner().to(device)
        swin_weight = 'hybrid_swin_vision_finetuned.pth'
        if os.path.exists(swin_weight):
            model_swin.encoder.load_state_dict(torch.load(swin_weight, map_location=device))
            print("   ✅ โหลดน้ำหนักที่เทรนแล้วสำเร็จ")
        else:
            print("   ⚠️ ไม่พบน้ำหนักที่เทรน (กำลังใช้ Pre-trained ดั้งเดิม)")
            
        model_swin.eval()
        swin_train_feat, swin_train_labels = build_knowledge_base(model_swin, eval_train_loader, device)
        swin_acc = evaluate_model(model_swin, val_loader, swin_train_feat, swin_train_labels, device)
        results["Swin Transformer"] = swin_acc
        print(f"   🏆 ความแม่นยำ Swin Transformer: {swin_acc:.2f}%\n")
    except Exception as e:
        print(f"   ❌ ทดสอบ Swin ล้มเหลว: {e}\n")

    # ------------------ สรุปผล ------------------
    print("==================================================")
    print("📊 สรุปผลการประชันความแม่นยำ (Validation Accuracy)")
    print("==================================================")
    for name, acc in results.items():
        print(f"  🟢 {name}: {acc:.2f}%")
        
    if len(results) == 2:
        winner = max(results, key=results.get)
        print(f"\n🎉 ผู้ชนะเลิศอันดับ 1 ได้แก่: **{winner}** 🥇")

    # ------------------ สร้าง Heatmap เปรียบเทียบ ------------------
    if 'model_dino' in locals() and 'model_swin' in locals():
        print("\n🔍 กำลังสร้างแผนผังโฟกัส (Attention Map) เปรียบเทียบ 2 โมเดล...")
        # สุ่มรูป 1 รูปจาก Validation
        val_samples = val_dataset.samples
        random_path, _ = random.choice(val_samples)
        
        img_pil = Image.open(random_path).convert("RGB")
        img_resized = img_pil.resize((224, 224))
        img_np = np.array(img_resized)
        
        img_tensor = transform(img_pil).unsqueeze(0)
        
        try:
            dino_hm = generate_dino_heatmap(model_dino, img_tensor, device)
            dino_overlay = create_heatmap_overlay(img_np, dino_hm)
            
            swin_hm = generate_swin_heatmap(model_swin, img_tensor, device)
            swin_overlay = create_heatmap_overlay(img_np, swin_hm)
            
            # วาดแบนเนอร์
            def add_banner(img, text):
                h, w, _ = img.shape
                banner = np.zeros((40, w, 3), dtype=np.uint8)
                cv2.putText(banner, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                return np.vstack([img, banner])
                
            dino_final = add_banner(dino_overlay, "DINOv2 Focus")
            swin_final = add_banner(swin_overlay, "Swin Focus")
            original_final = add_banner(img_np, "Original")
            
            comparison_img = np.hstack([original_final, dino_final, swin_final])
            save_path = "compare_attention_maps.png"
            Image.fromarray(comparison_img).save(save_path)
            print(f"   ✅ บันทึกภาพเปรียบเทียบ Attention Map ไว้ที่: {save_path} แล้วครับ!")
        except Exception as e:
            print(f"   ❌ ไม่สามารถสร้าง Attention Map ได้: {e}")

if __name__ == "__main__":
    main()
