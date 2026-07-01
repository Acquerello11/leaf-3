import subprocess
import sys

def run_step(script_name):
    print(f"\n{'='*50}")
    print(f"🚀 Running {script_name}...")
    print(f"{'='*50}\n")
    try:
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error occurred while running {script_name}. Pipeline stopped.")
        sys.exit(e.returncode)

def main():
    print("🌟 Starting End-to-End Data Pipeline for DINOv2 🌟\n")
    
    steps = [
        "hybrid_01_segment_leaves.py",
        "hybrid_03_finetune_dino.py",
        "hybrid_04_evaluate.py",
        "hybrid_05_visualize_attention.py"
    ]
    
    for step in steps:
        run_step(step)
        
    print("\n🎉 DINOv2 Pipeline completed successfully! 🎉")

if __name__ == "__main__":
    main()
