import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from rembg import remove, new_session
except ImportError:
    print("❌ ไม่พบไลบรารี rembg")
    exit(1)

def main():
    print("🌟 เริ่มกระบวนการ Stage 1: Auto-Segmentation ไดคัทใบไม้อัตโนมัติ (GPU Mode) 🌟")
    print("💡 โหมดนี้ใช้พลังการ์ดจอ (RTX 4080) ทะลวงความเร็วขั้นสุด!")
    
    input_dir = "Data200_Cleaned"
    output_dir = "Data200_Segmented"
    
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
    
    # โหลดโมเดลพร้อมเปิดการใช้งาน GPU (CUDAExecutionProvider)
    print("📥 กำลังโหลดโมเดล u2net เข้าสู่การ์ดจอ VRAM...")
    try:
        session = new_session("u2net", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    except Exception as e:
        print(f"⚠️ โหลดเซสชันล้มเหลว: {e} (กำลังใช้ค่าเริ่มต้นแทน)")
        session = None
    
    print("🚀 เริ่มลุยไดคัทด้วยการ์ดจอ...")
    for img_path in tqdm(all_files, desc="Segmenting (GPU)"):
        rel_path = os.path.relpath(img_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        out_path_png = os.path.splitext(out_path)[0] + '.png'
        
        os.makedirs(os.path.dirname(out_path_png), exist_ok=True)
        
        # ข้ามถ้ารูปนี้ทำไปแล้ว
        if os.path.exists(out_path_png):
            continue
            
        try:
            with open(img_path, 'rb') as i:
                input_data = i.read()
                
            # ไดคัทบนการ์ดจอ
            if session:
                output_data = remove(input_data, session=session)
            else:
                output_data = remove(input_data)
            
            with open(out_path_png, 'wb') as o:
                o.write(output_data)
                
        except Exception as e:
            # ถ้ามีปัญหาบางภาพ ก็ข้ามไป
            pass

    print(f"\n🎉 ไดคัทเสร็จสมบูรณ์! รูปทั้งหมดถูกเก็บไว้ที่ '{output_dir}'")

if __name__ == "__main__":
    main()
