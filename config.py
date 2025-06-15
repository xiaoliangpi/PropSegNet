import os

class Config:
    CUBLAS_WORKSPACE_CONFIG = ":4096:8"
    
    PATCH_H = 224  # height of the patch
    PATCH_W = 224  # width of the patch  
    STRIDE_H = 150  # stride applied along the height_dim
    STRIDE_W = 150  # stride applied along the width_dim
    
    PADDING_SIZE = 37  # Padding size
    PADDING_FILL = 255  # Fill value for white in RGB images
    
    WORKING_DIR = './'
    SOURCE_DIRS = ['A', 'B', 'D', 'non_predominant']
    CLASS_NAMES = ['HB', 'LB', 'SN', 'non_predominant']
    N_CLASSES = 3
    
    BATCH_SIZE = 4
    N_EPOCHS = 50
    LEARNING_RATE = 3e-5
    NUM_THREADS = 2
    
    MODEL_NAME = 'swin_tiny_patch4_window7_224'
    MODEL_INPUT_FEATURES = 768
    MODEL_OUTPUT_FEATURES = 3
    
    ROTATION_ANGLES = [0, 30, 60, 90, 120, 150]
    
    BACKGROUND_THRESHOLD = 245
    CROP_THRESHOLD = 240
    MORPHOLOGY_KERNEL_SIZE = (7, 7)
    
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    TARGET_SIZE = (1500, 1500)
    
    BRIGHTNESS_FACTOR = 0.05
    VERTICAL_FLIP_PROB = 0.5
    HORIZONTAL_FLIP_PROB = 0.5
    
    PRINT_FREQ = 10  # Print every 10 iterations
    
    @staticmethod
    def setup_environment():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = Config.CUBLAS_WORKSPACE_CONFIG