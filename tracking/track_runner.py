import cv2
import glob
from tracking.deepsort_tracker import DeepSortTracker

class TrackingRunner:
    def __init__(self ,detector, tracker: DeepSortTracker):
        self.detector = detector
        self.tracker = tracker

    def run(self,image_dir):
        image_paths  = sorted(glob.glob(f"{image_dir}/*.jpg"))
        
        all_tracks= []
        frame_idx = 0

        for img_path in image_paths:
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Frame is None: {frame}")
                continue
            
            # YOLO inference
            results = self.detector(frame, verbose=False)[0]

            detections = []
            for box in results.boxes:
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                conf = float(box.conf)
                cls = int(box.cls)
                detections.append([x1,y1,x2,y2,conf,cls])

            #Tracker (DeepSort) Update
            tracks = self.tracker.update(detections= detections, frame= frame)
            
            for t in tracks:
                t["frame"] = frame_idx
                t["image"] = img_path
                all_tracks.append(t)
            
            frame_idx +=1

        return all_tracks
    
    def draw_tracks(self, frame, tracks,frame_idx):
        for track in tracks:
            track_id = track.get("track_id")
            x1, y1, x2, y2 = map(int, track.get('bbox'))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"ID {track_id}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        cv2.imwrite(f"C:\\Users\\neokg\\Coding_Projects\\yolo-detector\\car_crash_dataset\\tracked_images_results\\frame_{frame_idx:05d}.jpg", frame)
        return
