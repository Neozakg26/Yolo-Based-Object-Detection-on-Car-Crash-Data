from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortTracker:

    def __init__(self, max_age=30,
                 n_init=3,
                 max_iou_distance=0.7,
                 embedder="mobilenet",
                 half= True,
                 bgr=True):
        
        self.tracker = DeepSort(max_age = max_age,
                                n_init = n_init,
                                max_iou_distance = max_iou_distance,
                                embedder=embedder,
                                half=half,
                                bgr=bgr
                                )

    def update(self, detections, frame, frame_idx, ego_motion:tuple, metadata,all_tracks):
        
    
        # Ego Information 
        ego_dx = float(ego_motion[0])
        ego_dy = float(ego_motion[1])
        ego_speed = float(ego_motion[2])
        ego_accel = float(ego_motion[3])
        ego_involve = "Yes" == metadata.get('egoinvolve')  

        formatted_dets =[]

        for det in detections:
            x1,y1,x2,y2, conf, cls = det
            w = float(x2-x1)
            h= float(y2-y1)
            formatted_dets.append(
                ([float(x1),float(y1),w,h],float(conf), int(cls))
            )

        tracks = self.tracker.update_tracks(
            formatted_dets,
            frame=frame
        )

        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            l,t,r,b = track.to_ltrb()

            x = float((l+r)/2)
            y = float((t+b)/2)
            w = float(r-l)
            h = float(b-t)

            vx= float(track.mean[4])
            vy=float(track.mean[5])
            all_tracks.append({
                    "frame": int(frame_idx),
                    "track_id": int(track_id),
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "vx": vx,
                    "vy": vy,
                    "class_id": int(track.get_det_class()),
                    "ego_dx":ego_dx,
                    "ego_dy":ego_dy,
                    "ego_speed":ego_speed,
                    "ego_accel":ego_accel,
                    "ego_involve":ego_involve
                })
            