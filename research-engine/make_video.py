import cv2
import numpy as np

print("Generating synthetic CCTV video...")

# Create a blank video file at 30 frames per second
out = cv2.VideoWriter('cctv_sample.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))

# Create 90 frames (3 seconds total)
for i in range(90):
    # Create a dark gray frame
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
    # Add some text that moves slightly
    cv2.putText(frame, f"CCTV CAM 01 - RECORDING", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Timestamp: 15:42:{i:02d}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # Draw a shape so CLIP has something to look at
    cv2.rectangle(frame, (300, 300), (400, 400), (0, 0, 255), -1) # A red square
    
    out.write(frame)

out.release()
print("✅ Created cctv_sample.mp4 in your folder!")