
import os
import sys
import shutil

# Add local ffmpeg to PATH
cwd = os.getcwd()
ffmpeg_path = os.path.join(cwd, "tools", "ffmpeg", "bin")
os.environ["PATH"] += os.pathsep + ffmpeg_path

def check_import(module_name):
    try:
        __import__(module_name)
        print(f"✅ {module_name} imported successfully.")
        return True
    except ImportError as e:
        print(f"❌ {module_name} failed to import: {e}")
        return False

def check_command(cmd):
    if shutil.which(cmd):
        print(f"✅ {cmd} found in PATH.")
        return True
    else:
        print(f"❌ {cmd} not found in PATH.")
        return False

def main():
    print("Verifying KAVACH-AI Environment...")
    
    # Check Python modules
    deps = ["torch", "cv2", "mediapipe", "fastapi", "yt_dlp"]
    all_pass = True
    for d in deps:
        if not check_import(d):
            all_pass = False
            
    # Check FFmpeg
    if not check_command("ffmpeg"):
         all_pass = False
         print(f"DEBUG: PATH={os.environ['PATH']}")
         print(f"DEBUG: Expected ffmpeg at {ffmpeg_path}")

    if all_pass:
        print("\n🎉 Environment Setup COMPLETE!")
    else:
        print("\n⚠️ Environment Setup INCOMPLETE.")
        sys.exit(1)

if __name__ == "__main__":
    main()
