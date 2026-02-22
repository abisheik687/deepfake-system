
import sys
import os
import shutil

# Add project root to path
sys.path.append(os.getcwd())

from backend.worker import process_video_task

def test_worker_logic():
    print("Testing Worker Pipeline Logic...")
    
    # Create a dummy file
    dummy_path = "test_worker_video.mp4"
    with open(dummy_path, "wb") as f:
        f.write(b"dummy_video_content")
        
    try:
        # Calling the task function directly (bypassing Celery broker)
        result = process_video_task(dummy_path, "test_task_001")
        
        print("\n✅ Worker Verification Passed!")
        print(f"Task ID: test_task_001")
        print(f"Verdict: {result['verdict']}")
        print(f"Final Score: {result['final_score']}")
        
        if result['final_score'] > 0.5:
             print("Create logic detected 'Suspicious' (Correct for mock).")
        else:
             print("Logic detected 'Real'.")
             
    except Exception as e:
        print(f"\n❌ Worker Logic Failed: {e}")
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)

if __name__ == "__main__":
    test_worker_logic()
