import cv2 
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np

ZONE_POLYGON = np.array([
    [0,0],
    [1280 //2,0],
    [1280 //2,720],
    [0,720]
])

def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280,720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_argument()
    frame_width,frame_height = args.webcam_resolution
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
    
    model = YOLO("yolov8s.pt")
    
    box_annonator = sv.BoxAnnotator(
        thickness = 2,
        text_thickness = 2 ,
        text_scale = 1
    )
    
    zone = sv.PolygonZone(polygon=ZONE_POLYGON,frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone,color= sv.Color.red())
    
    while True:
        ret, frame = cap.read()
        
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.class_id != 0] # disble person class
        labels = [
            # f"{model.model.names[b]} {c:0.2f}"
            f"{a[2]:0.2f} {model.model.names[a[3]]}"
            for a in detections
            
        ]
        
        frame = box_annonator.annotate(scene=frame,detections = detections,labels=labels)
        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame,)
        
        cv2.imshow("yolov8",frame)
        
        # print(frame.shape)
        # break
        if(cv2.waitKey(30) == 97):
            break

if __name__ == "__main__":
    main()