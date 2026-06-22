import os
import shutil
import random
from tqdm import tqdm

def main():
    print("🌟 เริ่มกระบวนการแบ่งข้อมูล (Train 70% / Val 15% / Test 15%) 🌟")
    
    input_dir = "Data200_Segmented"
    output_dir = "Data200_Hybrid_Split"
    
    if not os.path.exists(input_dir):
        print(f"❌ ไม่พบโฟลเดอร์ '{input_dir}' กรุณารันไฟล์ hybrid_01_segment_leaves.py เพื่อไดคัทรูปก่อนครับ")
        return
        
    # สร้างโฟลเดอร์หลักสำหรับ 3 splits
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
        
    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    print(f"พบทั้งหมด {len(classes)} คลาส (สายพันธุ์)")
    
    for cls in tqdm(classes, desc="กำลังแบ่ง Dataset"):
        cls_dir = os.path.join(input_dir, cls)
        
        # ดึงไฟล์รูปภาพทั้งหมดในคลาสนี้
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # สลับรูปภาพแบบสุ่ม เพื่อไม่ให้เกิดอคติในการแบ่ง
        random.shuffle(images)
        
        num_imgs = len(images)
        train_end = int(num_imgs * 0.70)
        val_end = train_end + int(num_imgs * 0.15)
        
        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]
        
        # ก๊อปปี้ไฟล์ไปยังโฟลเดอร์ใหม่
        for img_list, split in zip([train_imgs, val_imgs, test_imgs], ['train', 'val', 'test']):
            split_cls_dir = os.path.join(output_dir, split, cls)
            os.makedirs(split_cls_dir, exist_ok=True)
            for img in img_list:
                src = os.path.join(cls_dir, img)
                dst = os.path.join(split_cls_dir, img)
                shutil.copy2(src, dst)
                
    print(f"\n🎉 แบ่งข้อมูลสำเร็จ! รูปถูกแบ่งและจัดเก็บไว้ที่โฟลเดอร์ '{output_dir}'")
    print(f"- ข้อมูลสำหรับเทรน (Train): {output_dir}/train")
    print(f"- ข้อมูลสำหรับปรับแต่ง (Val): {output_dir}/val")
    print(f"- ข้อมูลสำหรับทดสอบ (Test): {output_dir}/test")

if __name__ == "__main__":
    main()
