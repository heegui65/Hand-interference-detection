from ultralytics import YOLO

model = YOLO('best.pt')

path = model.export(format='onnx', imgsz=640, simplify=True,opset=13)

print(f"Model successfully exported to: {path}")                                                                 

