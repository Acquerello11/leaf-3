import os
import cv2
import numpy as np
import concurrent.futures
from tqdm import tqdm

def process_single_image(seg_path, segmented_dir, cleaned_dir):
    try:
        img = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            return "Cannot read image"

        # ย่อขนาดรูปให้เล็กลงสุดๆ เพื่อความรวดเร็วในการคำนวณเปอร์เซ็นต์สี
        h, w = img.shape[:2]
        if h > 512 or w > 512:
            img_small = cv2.resize(img, (512, 512))
        else:
            img_small = img
            
        is_failed = False
        fail_reason = ""
        
        h_s, w_s = img_small.shape[:2]
        total_pixels = h_s * w_s

        if img_small.shape[2] == 4:
            alpha = img_small[:, :, 3]
            visible_pixels = np.sum(alpha > 0)
            
            if visible_pixels / total_pixels < 0.05:
                is_failed = True
                fail_reason = "Empty"
            else:
                bgr = img_small[:, :, :3]
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                
                lower_green = np.array([30, 40, 40])
                upper_green = np.array([90, 255, 255])
                
                green_mask = cv2.inRange(hsv, lower_green, upper_green)
                valid_green = cv2.bitwise_and(green_mask, green_mask, mask=alpha)
                green_pixels = np.sum(valid_green > 0)
                
                if green_pixels / visible_pixels < 0.10:
                    is_failed = True
                    fail_reason = "Not Green"

        if is_failed:
            rel_path = os.path.relpath(seg_path, segmented_dir)
            base_dir = os.path.dirname(os.path.join(cleaned_dir, rel_path))
            file_base = os.path.splitext(os.path.basename(seg_path))[0]
            
            raw_img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                guess_path = os.path.join(base_dir, file_base + ext)
                if os.path.exists(guess_path):
                    raw_img_path = guess_path
                    break
            
            if raw_img_path:
                # อันนี้ต้องอ่านรูปเต็ม เพื่อเซฟทับตัวเต็ม
                raw_img = cv2.imread(raw_img_path)
                if raw_img is not None:
                    cv2.imwrite(seg_path, raw_img)
            return fail_reason
        else:
            return "Success"
            
    except Exception as e:
        return f"Error: {e}"

def process_qc():
    print("🌟 เริ่มกระบวนการตรวจสอบคุณภาพไดคัท (Segmentation QC) 🌟")
    print("🚀 โหมดเทอร์โบ: ใช้ Threading และย่อขนาดภาพในการประมวลผล...")

    segmented_dir = "Data200_Segmented"
    cleaned_dir = "Data200_Cleaned"

    if not os.path.exists(segmented_dir):
        print(f"❌ ไม่พบโฟลเดอร์ {segmented_dir}")
        return

    all_files = []
    for root, _, files in os.walk(segmented_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_files.append(os.path.join(root, file))

    total_images = len(all_files)
    print(f"พบภาพไดคัททั้งหมด {total_images} ภาพ")

    # ใช้งาน ThreadPoolExecutor เพื่อให้อ่านรูปและประมวลผลพร้อมกันหลายๆ รูป (ดึงพลัง CPU ทุก Core)
    workers = min(32, os.cpu_count() * 2) if os.cpu_count() else 8
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_image, f, segmented_dir, cleaned_dir) for f in all_files]
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_images, desc="QC Checking (Turbo)"):
            results.append(future.result())

    failed_empty = results.count("Empty")
    failed_not_green = results.count("Not Green")
    success_count = results.count("Success")

    print("\n==============================================")
    print("📊 สรุปผลการตรวจสอบและซ่อมแซมคุณภาพไดคัท (QC)")
    print("==============================================")
    print(f"✅ ภาพที่ไดคัทสมบูรณ์ (เขียวชัดแจ๋ว): {success_count} ภาพ")
    print(f"🔄 ภาพที่พังและถูกแทนที่ด้วยรูปดิบต้นฉบับ: {failed_empty + failed_not_green} ภาพ")
    print(f"   👉 อาการลบจนขาวโพลน: {failed_empty} ภาพ")
    print(f"   👉 อาการติดมาแต่ลำต้นสีน้ำตาล: {failed_not_green} ภาพ")
    print("==============================================")
    print("ระบบพร้อมสำหรับการนำภาพไปเทรนต่อใน Stage ถัดไปแล้วครับ!")

if __name__ == "__main__":
    process_qc()
