import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
try:
    from rembg import remove
except ImportError:
    print("❌ ไม่พบไลบรารี rembg")
    print("กรุณาติดตั้งโดยรันคำสั่ง: pip install rembg[gpu] หรือ pip install rembg")
    exit(1)

def main():
    print("🌟 เริ่มกระบวนการ Stage 1: Auto-Segmentation ไดคัทใบไม้อัตโนมัติ 🌟")
    
    input_dir = "Data200_Cleaned"
    output_dir = "Data200_Segmented"
    
    if not os.path.exists(input_dir):
        print(f"❌ ไม่พบโฟลเดอร์ {input_dir}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # นับจำนวนไฟล์ทั้งหมด
    all_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_files.append(os.path.join(root, file))
                
    print(f"พบรูปภาพทั้งหมด {len(all_files)} รูปภาพ กำลังเริ่มไดคัท...")
    
    for img_path in tqdm(all_files, desc="Segmenting Images"):
        # สร้างโฟลเดอร์ย่อยใน output_dir ให้เหมือนกับ input_dir
        rel_path = os.path.relpath(img_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # ข้ามถ้ารูปนี้ทำไปแล้ว
        if os.path.exists(out_path):
            continue
            
        try:
            # อ่านภาพ
            with open(img_path, 'rb') as i:
                input_data = i.read()
                
            # ลบฉากหลัง (กระบวนการเดียวกับ Auto-SAM แต่จัดการง่ายกว่าและไม่ต้องใช้ Prompts)
            output_data = remove(input_data)
            
            # บันทึกเป็น PNG (เพื่อเก็บพื้นหลังโปร่งใส)
            # เปลี่ยนนามสกุลไฟล์เป็น .png
            out_path_png = os.path.splitext(out_path)[0] + '.png'
            
            with open(out_path_png, 'wb') as o:
                o.write(output_data)
                
            # หากต้องการเซฟเป็นพื้นสีดำ (Black Background) แทนแบบโปร่งใส 
            # สามารถใช้ cv2 โหลดรูป PNG มาเปลี่ยน Alpha เป็น Black ได้
            
        except Exception as e:
            print(f"⚠️ เกิดข้อผิดพลาดกับภาพ {img_path}: {e}")

    print(f"\n🎉 ไดคัทเสร็จสมบูรณ์! รูปทั้งหมดถูกเก็บไว้ที่ '{output_dir}'")

if __name__ == "__main__":
    main()
