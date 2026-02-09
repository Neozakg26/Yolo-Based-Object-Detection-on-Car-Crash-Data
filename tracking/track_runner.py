import cv2
import glob
from tracking.deepsort_tracker import DeepSortTracker
import numpy as np

class TrackingRunner:
    def __init__(self ,detector, tracker: DeepSortTracker):
        self.detector = detector
        self.tracker = tracker

    def run(self,image_dir,metadata):
        limit = int(metadata.get('accident_start_frame'))

        if limit is not None:
            image_paths  = sorted(glob.glob(f"{image_dir}*.jpg"))[:limit]
        else:
            image_paths  = sorted(glob.glob(f"{image_dir}*.jpg"))

        all_tracks= []
        frame_idx = 0
        prev_frame = None
        prev_speed = (0,0)

        for img_path in image_paths:
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Frame is None: {frame}")
                frame_idx +=1
                continue
            
            #Extract Ego Motion    
            ego_dx, ego_dy, ego_speed, ego_accel = self.__estimate_ego_motion__(prev_speed=prev_speed,prev_frame=prev_frame, curr_frame=frame)
            prev_speed = (ego_dx,ego_dy)
            
            # YOLO inference
            detected_results = self.detector(frame, verbose=False)[0]
            detections = []
            for box in detected_results.boxes:
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                conf = float(box.conf)
                cls = int(box.cls)
                detections.append([x1,y1,x2,y2,conf,cls])

            #Tracker (DeepSort) Update
            self.tracker.update(detections= detections, frame= frame, 
                                         frame_idx= frame_idx,
                                         ego_motion=(ego_dx,ego_dy,ego_speed,ego_accel),
                                         metadata=metadata,
                                         all_tracks=all_tracks)
            
            
            prev_frame = frame
            frame_idx +=1
        print(f" Frame ID: {frame_idx}")
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

    def __estimate_ego_motion__(self,prev_speed,prev_frame, curr_frame):

        if prev_frame is None:
            ego_dx, ego_dy, ego_speed, ego_accel= 0.0, 0.0, 0.0, 0.0
            return  ego_dx, ego_dy, ego_speed, ego_accel
        
            
        pre_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame,cv2.COLOR_BGR2GRAY)

        pts_prev = cv2.goodFeaturesToTrack(pre_gray,maxCorners=500
                                           ,qualityLevel=0.01, minDistance=7)
        
        pts_curr, status, err = cv2.calcOpticalFlowPyrLK(
            prevImg=pre_gray,nextImg=curr_gray,prevPts=pts_prev,nextPts=None)
        
        good_prev = pts_prev[status==1]
        good_curr = pts_curr[status==1]

        flow  = good_curr - good_prev

        ego_dx = np.median(flow[:,0])
        ego_dy = np.median(flow[:,1])

        ego_speed = np.sqrt(ego_dx**2 + ego_dy**2) #Euclidean 

        ax_t = ego_dx - prev_speed[0]
        ay_t = ego_dy -prev_speed[1]

        ego_accel = np.sqrt(ax_t**2 + ay_t**2)

        return ego_dx, ego_dy, ego_speed, ego_accel