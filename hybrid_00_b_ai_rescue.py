import os
import shutil
import re

# =========================================================
# 🟢 ตั้งค่าเกณฑ์ความใจดี (New Tolerant Thresholds) 🟢
# สามารถแก้ตัวเลขตรงนี้ได้เลยครับ เพื่อให้ AI ใจดีขึ้น
# =========================================================

# 1. เกณฑ์สำหรับภาพที่หลุดธีม (Outliers) 
# เดิมทีรูปจะถูกทิ้งถ้าคะแนนความเหมือน (sim) < 0.68 
# เกณฑ์ใหม่: ยอมรับภาพที่มีความคล้ายคลึง (sim) ตั้งแต่ 60% ขึ้นไป (0.60)
NEW_SIM_THRESHOLD = 0.60

# 2. เกณฑ์สำหรับภาพที่ซูมไกลเกินไป (Zoomed_out)
# เดิมทีรูปจะถูกทิ้งถ้า sim < 0.78 และมีเปอร์เซ็นต์สีเขียว (green) < 0.18 (18%)
# เกณฑ์ใหม่: ถ้ารูปมีสีเขียวมากกว่า 10% (0.10) ให้สอบผ่านได้เลย หรือถ้ามันคล้ายมากกว่า 70% ก็ให้ผ่าน
NEW_ZOOM_SIM_THRESHOLD = 0.70
NEW_ZOOM_GREEN_RATIO = 0.10

# =========================================================

def parse_outlier(filename):
    match = re.search(r"sim([0-9.]+)_", filename)
    if match:
        return float(match.group(1))
    return 0.0
    
def parse_zoom(filename):
    match = re.search(r"sim([0-9.]+)_gr([0-9.]+)_", filename)
    if match:
        return float(match.group(1)), float(match.group(2))
    return 0.0, 0.0

def get_original_class(filename, known_classes):
    for c in known_classes:
        if f"_{c}_" in filename:
            return c
    return None

def main():
    print("🤖 เข้าสู่โหมด AI ตรวจทานซ้ำ (AI Re-clean) 🤖")
    print(f"กำลังใช้เกณฑ์ที่ใจดีขึ้น:")
    print(f" - Outlier: ความเหมือน >= {NEW_SIM_THRESHOLD:.2f}")
    print(f" - Zoomed : ความเหมือน >= {NEW_ZOOM_SIM_THRESHOLD:.2f} หรือ มีสีเขียว >= {NEW_ZOOM_GREEN_RATIO:.2f}")
    
    clean_dir = "Data200_Cleaned"
    reject_dir = "Data200_Rejected"
    
    outliers_dir = os.path.join(reject_dir, "outliers")
    zoomed_dir = os.path.join(reject_dir, "zoomed_out")
    
    if not os.path.exists(clean_dir):
        print(f"❌ ไม่พบโฟลเดอร์ {clean_dir} กรุณารันไฟล์ 00 ก่อน")
        return
        
    known_classes = [d for d in os.listdir(clean_dir) if os.path.isdir(os.path.join(clean_dir, d))]
    
    success_count = 0
    fail_count = 0
    
    # ---------------- 1. กู้คืนจาก Outliers ----------------
    print("\n🔍 กำลังตรวจทานแฟ้ม Outliers...")
    if os.path.exists(outliers_dir):
        files = [f for f in os.listdir(outliers_dir) if os.path.isfile(os.path.join(outliers_dir, f))]
        for filename in files:
            if filename.startswith("corrupt_"):
                continue # ไฟล์เสีย ไม่กู้คืน
                
            sim = parse_outlier(filename)
            # ถ้าคะแนนสอบรอบใหม่ ผ่านเกณฑ์ใหม่
            if sim >= NEW_SIM_THRESHOLD:
                c = get_original_class(filename, known_classes)
                if c:
                    original_name = filename.split(f"_{c}_")[1]
                    dest_path = os.path.join(clean_dir, c, original_name)
                    # ป้องกันการตั้งชื่อไฟล์ซ้ำทับของเดิม
                    if os.path.exists(dest_path):
                        name, ext = os.path.splitext(original_name)
                        dest_path = os.path.join(clean_dir, c, f"{name}_rescued{ext}")
                    
                    shutil.move(os.path.join(outliers_dir, filename), dest_path)
                    success_count += 1
                else:
                    fail_count += 1
                    
    # ---------------- 2. กู้คืนจาก Zoomed_out ----------------
    print("🔍 กำลังตรวจทานแฟ้มภาพซูมไกล (Zoomed_out)...")
    if os.path.exists(zoomed_dir):
        files = [f for f in os.listdir(zoomed_dir) if os.path.isfile(os.path.join(zoomed_dir, f))]
        for filename in files:
            sim, gr = parse_zoom(filename)
            # ถ้าสีเขียวเยอะกว่าเกณฑ์ใหม่ หรือ ความเหมือนมากกว่าเกณฑ์ใหม่ ก็ให้ผ่านเลย
            if sim >= NEW_ZOOM_SIM_THRESHOLD or gr >= NEW_ZOOM_GREEN_RATIO:
                c = get_original_class(filename, known_classes)
                if c:
                    original_name = filename.split(f"_{c}_")[1]
                    dest_path = os.path.join(clean_dir, c, original_name)
                    if os.path.exists(dest_path):
                        name, ext = os.path.splitext(original_name)
                        dest_path = os.path.join(clean_dir, c, f"{name}_rescued{ext}")
                    
                    shutil.move(os.path.join(zoomed_dir, filename), dest_path)
                    success_count += 1
                else:
                    fail_count += 1

    print("\n================================================")
    print(f"🎉 ตรวจทานซ้ำเสร็จสิ้น: AI ใจดีขึ้นและปล่อยให้รูปสอบผ่านจำนวน {success_count} รูป!")
    print(f"รูปทั้งหมดถูกย้ายกลับเข้า {clean_dir} แยกตามโฟลเดอร์เรียบร้อย")
    print("================================================")
    if success_count > 0:
        print("👉 อย่าลืมรันไฟล์ไดคัท (01) และสุ่มแบ่งข้อมูล (02) ใหม่ด้วยนะครับ เพื่อให้รูปเซ็ตใหม่เข้าสู่ระบบ!")

if __name__ == "__main__":
    main()
