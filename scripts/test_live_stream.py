
import cv2
import time
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from backend.ingestion.stream_loader import StreamLoader

def test_stream(url, duration=30):
    print(f"Testing stream ingestion for {duration} seconds...")
    loader = StreamLoader(url, sample_rate=5).start() # Sample every 5th frame
    
    start_time = time.time()
    frames_processed = 0
    
    try:
        while loader.running() and (time.time() - start_time) < duration:
            frame = loader.read()
            if frame is None:
                continue
                
            frames_processed += 1
            
            # Display (optional, might not work in headless env)
            # cv2.imshow("Stream Test", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            
            if frames_processed % 10 == 0:
                print(f"Processed {frames_processed} frames...", end='\r')
                
    except KeyboardInterrupt:
        print("\nStopping...")
        
    loader.stop()
    cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    print(f"\nTest Complete.")
    print(f"Time: {elapsed:.2f}s")
    print(f"Frames: {frames_processed}")
    print(f"Actual FPS Processed: {frames_processed/elapsed:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://www.youtube.com/watch?v=jfKfPfyJRdk", help="YouTube Live URL (default: Lofi Girl)")
    args = parser.parse_args()
    
    test_stream(args.url)
