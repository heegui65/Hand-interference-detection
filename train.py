from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    # model = YOLO('yolov8n.pt')
    # model=YOLO('yolov7-tiny.pt')
    cfg_path = os.getenv("YOLO_CFG", "yolo11n.pt")
    model=YOLO(cfg_path)
    model.train(
        data='data.yaml',
        epochs=100,
        batch=16,
        imgsz=640,
        device='0',
        patience=50,
        workers=4          # If still error, set workers=0 to disable multiprocessing
    )

    metrics = model.val(data='data.yaml', split='test')  # Key: split='test'
    print("Test set mAP50-95:", metrics.box.map)
    print("Test set mAP50:", metrics.box.map50)
    print("Test set Recall   :", metrics.box.mr)      
    print("Test set Precision:", metrics.box.mp)      
if __name__ == '__main__':
    # Must add this on Windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()
