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

    def update(self, detections, frame):

        formatted_dets =[]

        for det in detections:
            x1,y1,x2,y2, conf, cls = det
            formatted_dets.append(
                ([x1,y1,x2-x1,y2-y1],conf, cls)
            )

        tracks = self.tracker.update_tracks(
            formatted_dets,
            frame=frame
        )

        results = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l,t,w,h = track.to_ltrb()
            cls = track.get_det_class()

            results.append({
                    "track_id": track_id,
                    "bbox": [l,t,w,h],
                    "class_id": cls
                })
            
        return results