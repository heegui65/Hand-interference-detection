from ultralytics import YOLO
import os
import torch # Import torch for gradient operations

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Added: Sparsity training callback function ---
def sldh_sparsity_step(trainer):
    """
    After backpropagation, apply L1 regularization penalty to the weights of all BN layers.
    The larger the SR (Sparsity Rate), the more aggressive the pruning. Typically set to 0.001.
    """
    SR = 0.001 
    for m in trainer.model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            # Apply gradient to BN layer weights (gamma) to push them towards 0
            m.weight.grad.data.add_(SR * torch.sign(m.weight.data))

def main():
    cfg_path = os.getenv("YOLO_CFG", "yolo11-SLDH.yaml") # Ensure to load your SLDH configuration file
    model = YOLO(cfg_path).load('yolo11n.pt') # Load pretrained weights
    # Note: This callback will run after each backpropagation
    model.add_callback("on_after_backward", sldh_sparsity_step)
    print("Successfully registered BN sparsity training callback (Sparsity Training).")

    model.train(
        data='data.yaml',
        epochs=100,
        batch=16,
        imgsz=640,
        device='0',
        patience=50,
        workers=4
    )
  
    metrics = model.val(data='data.yaml', split='test')
    print("Test set mAP50-95:", metrics.box.map)
    print("Test set mAP50:", metrics.box.map50)
    print("Test set Recall   :", metrics.box.mr)      
    print("Test set Precision:", metrics.box.mp)      

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()