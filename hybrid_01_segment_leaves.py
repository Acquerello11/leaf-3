import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image

try:
    from transformers import pipeline
except ImportError:
    print("❌ ไม่พบไลบรารี transformers")
    exit(1)

def get_best_mask(masks_list):
    """
    SAM 2 pipeline returns multiple masks.
    We want to find the largest mask that does NOT cover the entire image (i.e., background).
    """
    best_mask = None
    max_area = 0
    
    for mask_item in masks_list:
        if isinstance(mask_item, Image.Image):
            mask_np = np.array(mask_item)
        else:
            mask_np = np.array(mask_item)
            
        area = np.sum(mask_np > 0)
        total_area = mask_np.shape[0] * mask_np.shape[1]
        
        # Ignore masks that take up more than 95% of the image (likely the background)
        if area > 0.95 * total_area:
            continue
            
        if area > max_area:
            max_area = area
            best_mask = mask_np
            
    return best_mask

def main():
    print("🌟 เริ่มกระบวนการ Stage 1: Auto-Segmentation ไดคัทใบไม้อัตโนมัติ (SAM 2 GPU) 🌟")
    print("💡 โหมดนี้ใช้พลังการ์ดจอ (RTX 4080) กับโมเดล facebook/sam2-hiera-large ทะลวงความเร็วขั้นสุด!")
    
    input_dir = "Data200_Raw_Split"
    output_dir = "Data200_Segmented_Split"
    
    if not os.path.exists(input_dir):
        print(f"❌ ไม่พบโฟลเดอร์ {input_dir}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    all_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_files.append(os.path.join(root, file))
                
    print(f"พบรูปภาพทั้งหมด {len(all_files)} รูปภาพ กำลังเตรียมตัว...")
    
    device = 0 if torch.cuda.is_available() else -1
    print(f"📥 กำลังโหลดโมเดล SAM 2 (facebook/sam2-hiera-large) เข้าสู่ {'GPU' if device == 0 else 'CPU'}...")
    try:
        generator = pipeline("mask-generation", model="facebook/sam2-hiera-large", device=device, points_per_batch=64)
    except Exception as e:
        print(f"⚠️ โหลดโมเดลล้มเหลว: {e}")
        return
    
    print("🚀 เริ่มลุยไดคัทด้วย SAM 2...")
    for img_path in tqdm(all_files, desc="Segmenting (SAM 2 GPU)"):
        rel_path = os.path.relpath(img_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        out_path_png = os.path.splitext(out_path)[0] + '.png'
        
        os.makedirs(os.path.dirname(out_path_png), exist_ok=True)
        
        if os.path.exists(out_path_png):
            continue
            
        try:
            image = Image.open(img_path).convert("RGB")
            outputs = generator(image)
            
            masks = outputs.get("masks", [])
            best_mask = get_best_mask(masks)
            
            if best_mask is not None:
                img_np = np.array(image)
                alpha_channel = (best_mask > 0).astype(np.uint8) * 255
                rgba_img = np.dstack((img_np, alpha_channel))
                Image.fromarray(rgba_img).save(out_path_png)
            else:
                # Fallback: if no good mask, save original image but warn
                print(f"\n⚠️ ไม่พบ Mask ที่เหมาะสมสำหรับ {img_path}")
                img_np = np.array(image)
                alpha_channel = np.ones((img_np.shape[0], img_np.shape[1]), dtype=np.uint8) * 255
                rgba_img = np.dstack((img_np, alpha_channel))
                Image.fromarray(rgba_img).save(out_path_png)
                
        except Exception as e:
            print(f"\n❌ Error processing {img_path}: {e}")
            pass

    print(f"\n🎉 ไดคัทด้วย SAM 2 เสร็จสมบูรณ์! รูปทั้งหมดถูกเก็บไว้ที่ '{output_dir}'")

if __name__ == "__main__":
    main()
