import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.backends.cudnn as cudnn

try:
    import timm
except ImportError:
    print("❌ ไม่พบไลบรารี timm")
    print("กรุณาติดตั้งโดยรันคำสั่ง: pip install timm")
    exit(1)

cudnn.benchmark = True

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss"""
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        features = F.normalize(features, dim=1)
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mask_pos_sum = mask.sum(1)
        mask_pos_sum = torch.where(mask_pos_sum == 0, torch.ones_like(mask_pos_sum), mask_pos_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_sum
        loss = -mean_log_prob_pos
        return loss.mean()

class SwinVisionFinetuner(nn.Module):
    def __init__(self, model_name='swinv2_large_window12to16_192to256'):
        super().__init__()
        # ใช้ timm โหลด Swin Transformer แบบดึงเฉพาะ Feature (num_classes=0)
        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        # แช่แข็งพารามิเตอร์เกือบทั้งหมด
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # ปลดล็อกเฉพาะ 2 Blocks สุดท้ายของ Swin
        if hasattr(self.encoder, 'layers'):
            for param in self.encoder.layers[-1].blocks[-2:].parameters():
                param.requires_grad = True
        
        # Swin Large จะให้ output ขนาด 1536
        self.head = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)

    def get_features(self, x):
        return self.encoder(x)

def evaluate_knn(model, train_loader, val_loader, device):
    """ฟังก์ชันทดสอบความแม่นยำด้วย k-NN แบบรวดเร็ว (Accuracy-First)"""
    model.eval()
    
    # 1. สร้างคลังสมองจาก Train
    train_features = []
    train_labels = []
    with torch.inference_mode():
        for inputs, labels in tqdm(train_loader, desc="   [k-NN] Building KB", leave=False):
            inputs = inputs.to(device)
            with torch.amp.autocast('cuda'):
                features = model.get_features(inputs)
            features = F.normalize(features, dim=1)
            train_features.append(features.cpu())
            train_labels.append(labels)
            
    train_features = torch.cat(train_features, dim=0).to(device)
    train_labels = torch.cat(train_labels, dim=0).to(device)
    
    # 2. ทดสอบกับ Val
    correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, labels in tqdm(val_loader, desc="   [k-NN] Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                val_features = model.get_features(inputs)
            val_features = F.normalize(val_features, dim=1)
            
            # Dot Product หาระยะห่าง Cosine Similarity บน GPU (เร็วมาก)
            sim_matrix = torch.matmul(val_features, train_features.T)
            best_match_indices = torch.argmax(sim_matrix, dim=1)
            preds = train_labels[best_match_indices]
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return (correct / total) * 100

def main():
    print("🌟 เริ่มกระบวนการ Stage 3 (ทางเลือก): Feature Extraction ด้วย Swin Transformer 🌟")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"รันบนอุปกรณ์: {device}")
    
    # Transform สำหรับตอนเทรนปกติ (SwinV2 ตัวนี้รับภาพขนาด 256x256 ได้ดีที่สุด)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Transform สำหรับตอนทำข้อสอบย่อย k-NN (ห้ามบิดรูป)
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = "Data200_Raw_Split/train"
    val_dir = "Data200_Raw_Split/val"
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"❌ ไม่พบโฟลเดอร์ข้อมูล {train_dir} หรือ {val_dir} กรุณารันไฟล์ 02 ก่อน")
        return

    # โหลดชุดข้อมูล
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    print(f"ข้อมูลสำหรับสอน (Train): {len(train_dataset)} รูปภาพ")
    print(f"ข้อมูลสำหรับสอบย่อย (Val): {len(val_dataset)} รูปภาพ")
    
    # 1. Dataloader สำหรับอัปเดต Loss (มี Augment)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    
    # 2. Dataloader สำหรับคลังสมองตอนสอบย่อย (ไม่มี Augment จะได้แม่นๆ Batch Size อัดได้เยอะเพราะไม่ต้องหา Gradient)
    eval_train_dataset = datasets.ImageFolder(train_dir, transform=val_transform)
    eval_train_loader = DataLoader(eval_train_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    
    # 3. Dataloader สำหรับโจทย์ข้อสอบย่อย
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    
    print("\n📥 กำลังโหลดโมเดล SOTA ทางเลือก (Swin Transformer Large)...")
    model = SwinVisionFinetuner().to(device)
    
    weight_path = 'hybrid_swin_vision_finetuned.pth'
    if os.path.exists(weight_path):
        print(f"🔄 พบไฟล์น้ำหนักเดิม '{weight_path}' โหลดต่อเพื่อสานความฉลาด...")
        model.encoder.load_state_dict(torch.load(weight_path, map_location=device))
        
    criterion = SupConLoss(temperature=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    epochs = 15
    best_val_acc = 0.0
    
    print("\n🚀 เริ่มต้นการเทรน พร้อมระบบสอบย่อย (k-NN) ทุกสิ้นสุดการเรียนแต่ละรอบ...")
    for epoch in range(epochs):
        # --------------------- TRAINING PHASE ---------------------
        model.train()
        running_loss = 0.0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [เรียนรู้]")
        for inputs, labels in progress:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                features = model(inputs)
                loss = criterion(features, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            progress.set_postfix({'loss': f"{loss.item():.4f}"})
            
        epoch_loss = running_loss / len(train_loader)
        
        # --------------------- VALIDATION PHASE ---------------------
        print(f"   ⏳ กำลังสอบย่อยแบบ k-NN บนโฟลเดอร์ Val (เพื่อป้องกัน Overfitting)...")
        val_acc = evaluate_knn(model, eval_train_loader, val_loader, device)
        
        print(f"   📊 Epoch {epoch+1} สรุปผล | Train Loss: {epoch_loss:.4f} | 🎯 สอบย่อย (Val Accuracy): {val_acc:.2f}%")
        
        # เซฟโมเดลเฉพาะตอนที่ "คะแนนสอบย่อย" ทำสถิติใหม่เท่านั้น
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.encoder.state_dict(), weight_path)
            print(f"   🏆 New High Score! เซฟโมเดลเก็บไว้ที่ความแม่นยำ {val_acc:.2f}% ({weight_path})")

    print(f"\n🎉 ฝึกสอนโมเดล Swin Transformer สำเร็จ! (ความแม่นยำสูงสุดระหว่างเทรนที่เซฟไว้: {best_val_acc:.2f}%)")

if __name__ == "__main__":
    main()
