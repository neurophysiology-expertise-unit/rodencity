import cv2
import numpy as np
import os

width, height = 640, 480
fps = 10
out = cv2.VideoWriter('test_video.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

for i in range(50):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # create a moving circle to emulate a mouse
    x = 100 + i * 8
    y = 240 + int(30 * np.sin(i * 0.5))
    cv2.circle(frame, (x, y), 20, (200, 200, 200), -1)
    out.write(frame)
    
out.release()
print("Generated test_video.avi successfully.")
