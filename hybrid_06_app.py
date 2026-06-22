import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import gradio as gr
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from rembg import remove
import importlib.util

# โหลดคลาส HFVisionFinetuner
spec = importlib.util.spec_from_file_location("hybrid3", "hybrid_03_finetune_dino.py")
hybrid3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hybrid3)
HFVisionFinetuner = hybrid3.HFVisionFinetuner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ตัวแปร Global เพื่อโหลดโมเดลครั้งเดียวตอนเปิดแอป
model = None
train_features = None
train_labels = None
classes = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def init_system():
    global model, train_features, train_labels, classes
    
    weights_path = "hybrid_hf_vision_finetuned.pth"
    train_dir = "Data200_Hybrid_Split/train"
    
    if not os.path.exists(weights_path) or not os.path.exists(train_dir):
        print(f"❌ ระบบไม่พร้อม: ไม่พบไฟล์ {weights_path} หรือ {train_dir}")
        return False
        
    print("📥 กำลังเปิดระบบ... โหลดโมเดล DINO (HF)...")
    model = HFVisionFinetuner().to(device)
    model.encoder.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    print("🧠 กำลังสร้างคลังสมองจาก Train Set (ทำครั้งเดียว)...")
    dataset = datasets.ImageFolder(train_dir, transform=transform)
    classes = dataset.classes
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    features_list = []
    labels_list = []
    with torch.inference_mode():
        for inputs, labels in tqdm(dataloader, desc="Loading KB"):
            inputs = inputs.to(device)
            features = model.get_features(inputs)
            features = F.normalize(features, dim=1)
            features_list.append(features.cpu())
            labels_list.append(labels)
            
    train_features = torch.cat(features_list, dim=0)
    train_labels = torch.cat(labels_list, dim=0)
    print("✅ โหลดความจำสำเร็จ! ระบบพร้อมใช้งาน")
    return True

def process_single_image(img_path, auto_segment):
    # โหลดรูปภาพ
    img = Image.open(img_path).convert("RGB")
    
    # 1. ระบบไดคัท (ถ้าเปิดใช้งาน)
    if auto_segment:
        img_np = np.array(img)
        # รัน rembg ตัดฉากหลัง (ได้ภาพ RGBA)
        img_seg = remove(img_np)
        img_pil = Image.fromarray(img_seg)
        # ซ้อนภาพลงบนพื้นหลังสีดำเพื่อป้องกันสัญญาณรบกวน
        background = Image.new("RGB", img_pil.size, (0, 0, 0))
        background.paste(img_pil, mask=img_pil.split()[3])
        img = background

    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # 2. ทำนายผลด้วย k-NN
    with torch.inference_mode():
        test_feature = model.get_features(input_tensor)
        test_feature = F.normalize(test_feature, dim=1).cpu()
        
        sim_matrix = torch.matmul(test_feature, train_features.T)
        best_match_idx = torch.argmax(sim_matrix, dim=1).item()
        confidence = torch.max(sim_matrix).item() * 100
        
        pred_label_idx = train_labels[best_match_idx].item()
        pred_class = classes[pred_label_idx]

    # 3. ดึงโครงสร้างสมองสร้าง Attention Heatmap
    with torch.inference_mode():
        outputs = model.encoder(pixel_values=input_tensor, output_attentions=True)
        
    attentions = outputs.attentions[-1]
    cls_attention = attentions[0, :, 0, 1:]
    mean_attention = torch.mean(cls_attention, dim=0).cpu().numpy()
    
    grid_size = int(np.sqrt(mean_attention.shape[0]))
    attention_grid = mean_attention.reshape(grid_size, grid_size)
    attention_grid = (attention_grid - attention_grid.min()) / (attention_grid.max() - attention_grid.min() + 1e-8)
    attention_grid = cv2.resize(attention_grid, (224, 224))
    
    img_resized = np.array(img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_grid), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # ซ้อนทับภาพ Heatmap เข้ากับรูปถ่าย
    alpha = 0.5
    superimposed = cv2.addWeighted(img_resized, alpha, heatmap, 1 - alpha, 0)
    
    # สร้างกรอบแบนเนอร์ด้านล่างเพื่อแสดงคำทำนาย
    h, w, _ = superimposed.shape
    banner = np.zeros((40, w, 3), dtype=np.uint8)
    # เขียนชื่อคลาสและ %
    text = f"{pred_class} ({confidence:.1f}%)"
    cv2.putText(banner, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # ประกบภาพกับแบนเนอร์
    final_img = np.vstack([superimposed, banner])
    
    return final_img, text

def process_batch(files, auto_segment):
    if not files:
        return []
        
    results = []
    # files จาก Gradio File component จะมาเป็น List ของออบเจกต์ไฟล์
    # เราใช้ .name เพื่อดึงพิกัด (path) ชั่วคราวของไฟล์ที่อัปโหลดมา
    for file in tqdm(files, desc="Processing Images"):
        file_path = file.name
        final_img, caption = process_single_image(file_path, auto_segment)
        # ส่งกลับไปเป็น Tuple (รูป, แคปชั่น) ให้ Gallery เอาไปแสดงผล
        results.append((final_img, caption))
        
    return results

def create_ui():
    with gr.Blocks(theme=gr.themes.Glass()) as app:
        gr.Markdown("# 🌿 ระบบจำแนกใบไม้อัจฉริยะแบบ Hybrid (Batch Mode)")
        gr.Markdown("โปรแกรมสำหรับอัปโหลดภาพชุดใหญ่ เพื่อดูว่าโมเดลทายถูกไหม และแอบดูสมอง AI (Heatmap) ว่ามันโฟกัสที่จุดไหน")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. ใส่ข้อมูลภาพ")
                file_input = gr.File(label="📂 อัปโหลดรูปภาพ (ลากวางได้ทีละหลายๆ รูป)", file_count="multiple", file_types=["image"])
                auto_seg = gr.Checkbox(label="✂️ ไดคัทภาพป่า/ฉากหลังอัตโนมัติ (บังคับเปิดสำหรับรูปดิบ)", value=True)
                submit_btn = gr.Button("🚀 ประมวลผล", variant="primary")
            
            with gr.Column(scale=2):
                gr.Markdown("### 2. ผลลัพธ์การสแกน")
                gallery_output = gr.Gallery(label="ผลลัพธ์ (Heatmap & คำทำนาย)", columns=3, height="auto", object_fit="contain")
                
        submit_btn.click(fn=process_batch, inputs=[file_input, auto_seg], outputs=gallery_output)
        
    return app

if __name__ == "__main__":
    success = init_system()
    if success:
        app = create_ui()
        print("\n🌐 สตาร์ทเซิร์ฟเวอร์เรียบร้อย! กรุณากดลิงก์ http://127.0.0.1:7860 ด้านล่างเพื่อเปิดโปรแกรมบนเบราว์เซอร์")
        app.launch(share=False, server_name="127.0.0.1", server_port=7860)
