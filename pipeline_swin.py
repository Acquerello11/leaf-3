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
    print("🌟 Starting End-to-End Data Pipeline for Swin Transformer 🌟\n")
    
    steps = [
        "hybrid_01_segment_leaves.py",
        "hybrid_03_finetune_swin.py",
        "hybrid_04_evaluate_swin.py",
        "hybrid_05_visualize_attention_swin.py"
    ]
    
    for step in steps:
        run_step(step)
        
    print("\n🎉 Swin Transformer Pipeline completed successfully! 🎉")

if __name__ == "__main__":
    main()
