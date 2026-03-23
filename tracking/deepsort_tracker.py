from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np


class DeepSortTracker:

    def __init__(self, max_age=30,
                 n_init=3,
                 max_iou_distance=0.7,
                 embedder="mobilenet",
                 half=True,
                 bgr=True):

        self.tracker = DeepSort(max_age=max_age,
                                n_init=n_init,
                                max_iou_distance=max_iou_distance,
                                embedder=embedder,
                                half=half,
                                bgr=bgr)

        # Track history for computing acceleration and ttc_rate
        self.track_history = {}

    def update(self, detections, frame, frame_idx, ego_motion: tuple, all_tracks):

        frame_height = frame.shape[0]

        # Ego Information
        ego_dx = float(ego_motion[0])
        ego_dy = float(ego_motion[1])
        ego_speed = float(ego_motion[2])
        ego_accel = float(ego_motion[3])
        # ego_involve = "Yes" == metadata.get('egoinvolve')

        formatted_dets = []

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            w = float(x2 - x1)
            h = float(y2 - y1)
            formatted_dets.append(
                ([float(x1), float(y1), w, h], float(conf), int(cls))
            )

        tracks = self.tracker.update_tracks(
            formatted_dets,
            frame=frame
        )

        for track in tracks:
            if not track.is_confirmed():
                print("ERROR: Track  is NOt confirmed!")
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()

            x = float((l + r) / 2)
            y = float((t + b) / 2)
            w = float(r - l)
            h = float(b - t)

            # Kalman-filtered state from DeepSort
            vx = float(track.mean[4])
            vy = float(track.mean[5])
            va = float(track.mean[6])  # velocity of aspect ratio
            vh = float(track.mean[7])  # velocity of height (Kalman-filtered)

            # TTC confidence from Kalman covariance
            vh_variance = float(track.covariance[7, 7])
            ttc_confidence = float(1.0 / (1.0 + np.sqrt(vh_variance)))

            # Initialize track history if new track
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    'prev_vx': vx,
                    'prev_vy': vy,
                    'prev_ttc': None
                }
            history = self.track_history[track_id]

            # Speed: velocity magnitude (Kalman-filtered)
            speed = float(np.sqrt(vx**2 + vy**2))

            # Acceleration: change in Kalman-filtered velocity
            ax = float(vx - history['prev_vx'])
            ay = float(vy - history['prev_vy'])

            # Relative velocity (object - ego motion)
            rel_vx = float(vx - ego_dx)
            rel_vy = float(vy - ego_dy)

            # Proximity: h / frame_height (scale-based depth proxy)
            proximity = float(h / frame_height) if frame_height > 0 else 0.0

            # Distance proxy: inverse of proximity
            distance_proxy = float(1.0 / proximity) if proximity > 0.01 else 100.0

            # Closing rate: from Kalman-filtered vh (vh > 0 = approaching)
            closing_rate = float(vh / frame_height) if frame_height > 0 else 0.0

            # Risk speed: combined speed-proximity risk
            risk_speed = float(speed * proximity)

            # TTC proxy: time-to-collision using Kalman-filtered closing_rate
            if closing_rate > 1e-4:
                ttc_height = (1.0 - proximity) / closing_rate
            else:
                ttc_height = 999.0

            # Area-based TTC (more robust than height alone)
            area = w * h
            area_rate = w * vh + h * vx  # d(area)/dt approximation
            if area_rate > 1.0:
                ttc_area = area / area_rate
            else:
                ttc_area = 999.0

            # Use minimum of both estimates (conservative)
            ttc_proxy = float(np.clip(min(ttc_height, ttc_area), 0.0, 999.0))

            # TTC smoothed: same as ttc_proxy (vh is already Kalman-filtered)
            ttc_smoothed = ttc_proxy

            # TTC relative: based on relative velocity magnitude
            rel_speed = np.sqrt(rel_vx**2 + rel_vy**2)
            if rel_speed > 0.5:
                ttc_relative = distance_proxy / rel_speed
            else:
                ttc_relative = 999.0
            ttc_relative = float(np.clip(ttc_relative, 0.0, 999.0))

            # TTC rate: change in TTC (negative = danger increasing)
            prev_ttc = history['prev_ttc']
            if prev_ttc is not None:
                ttc_rate = float(ttc_proxy - prev_ttc)
            else:
                ttc_rate = 0.0

            # Update history for next frame
            history['prev_vx'] = vx
            history['prev_vy'] = vy
            history['prev_ttc'] = ttc_proxy

            all_tracks.append({
                "frame": int(frame_idx),
                "track_id": int(track_id),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "vx": vx,
                "vy": vy,
                "va": va,
                "vh": vh,
                "ax": ax,
                "ay": ay,
                "speed": speed,
                "rel_vx": rel_vx,
                "rel_vy": rel_vy,
                "proximity": proximity,
                "distance_proxy": distance_proxy,
                "closing_rate": closing_rate,
                "risk_speed": risk_speed,
                "ttc_proxy": ttc_proxy,
                "ttc_smoothed": ttc_smoothed,
                "ttc_relative": ttc_relative,
                "ttc_rate": ttc_rate,
                "ttc_confidence": ttc_confidence,
                "class_id": int(track.get_det_class()),
                "ego_dx": ego_dx,
                "ego_dy": ego_dy,
                "ego_speed": ego_speed,
                "ego_accel": ego_accel
                # ,"ego_involve": ego_involve
            })
            