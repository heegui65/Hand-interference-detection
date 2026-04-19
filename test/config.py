# config.py

# Original video parameters
ORIGINAL_VIDEO_WIDTH = 2592
ORIGINAL_VIDEO_HEIGHT = 1944

# Input resolution scaling (to improve inference speed, 0.5 means scale to 50%)
INPUT_SCALE = 0.5  # Adjust this value to change inference input size: 0.25/0.33/0.5/0.66 etc.

# Center crop region (original video coordinates)
CROP_X1 = (ORIGINAL_VIDEO_WIDTH - ORIGINAL_VIDEO_HEIGHT) // 2  # 324
CROP_Y1 = 0
CROP_X2 = CROP_X1 + ORIGINAL_VIDEO_HEIGHT  # 2268
CROP_Y2 = ORIGINAL_VIDEO_HEIGHT  # 1944
CROP_SIZE = ORIGINAL_VIDEO_HEIGHT  # 1944

# Display window size
DISPLAY_WIDTH = 720
DISPLAY_HEIGHT = 720

# Detection thresholds
CONF_THRESH = 0.6
IOU_THRESH = 0.5
ALARM_FRAME_THRESH = 2  # Trigger alarm only if detected in 2 consecutive frames
IOU_INTERVENT_THRESH = 0.3  # IOU threshold between hand and fixed region

# Video parameters
VIDEO_WIDTH = 1944  # Width after cropping
VIDEO_HEIGHT = 1944  # Height after cropping
ROTATE_ANGLE = 0  # No rotation needed for Raspberry Pi camera, adjust as needed
TARGET_SIZE = 640  # Model input size

# Video source settings
VIDEO_SOURCE = "test.mp4"  # None for camera, or specify video file path like "test_video.mp4"