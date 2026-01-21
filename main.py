import cv2
import numpy as np
import os


class MotionBagCounter:
    def __init__(
        self,
        video_path,
        line_position=0.5,
        min_area=500,
        max_area=50000,
        roi=None,
        use_diagonal_line=False,
        line_length_ratio=1.0,
    ):
        self.video_path = video_path
        self.line_position = line_position
        self.min_area = min_area
        self.max_area = max_area
        self.roi = roi
        self.use_diagonal_line = use_diagonal_line
        self.line_length_ratio = line_length_ratio
        self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
            history=300, dist2Threshold=500, detectShadows=True
        )
        self.tracked_objects = {}
        self.next_id = 0
        self.counted_ids = set()
        self.counted_times = {}
        self.total_count = 0
        self.max_distance = 80

    def get_centroids(self, frame):
        if self.roi:
            x1, y1, x2, y2 = self.roi
            fg_mask = self.bg_subtractor.apply(frame[y1:y2, x1:x2].copy())
            full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = fg_mask
            fg_mask = full_mask
        else:
            fg_mask = self.bg_subtractor.apply(frame)

        fg_mask[fg_mask == 127] = 0
        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

        centroids, boxes = [], []
        for contour in cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w // 2, y + h // 2
                if not self.roi or (
                    self.roi[0] <= cx <= self.roi[2]
                    and self.roi[1] <= cy <= self.roi[3]
                ):
                    centroids.append((cx, cy))
                    boxes.append((x, y, w, h))
        return centroids, boxes

    def update_tracks(self, centroids, boxes):
        if not self.tracked_objects:
            for centroid, box in zip(centroids, boxes):
                self.tracked_objects[self.next_id] = {
                    "centroid_history": [centroid],
                    "box": box,
                    "frames_missing": 0,
                }
                self.next_id += 1
            return

        matched_tracks, matched_centroids = set(), set()
        for tid in list(self.tracked_objects.keys()):
            prev_c = self.tracked_objects[tid]["centroid_history"][-1]
            min_dist, min_idx = float("inf"), -1
            for i, c in enumerate(centroids):
                if i not in matched_centroids:
                    d = np.sqrt((c[0] - prev_c[0]) ** 2 + (c[1] - prev_c[1]) ** 2)
                    if d < min(min_dist, self.max_distance):
                        min_dist, min_idx = d, i

            if min_idx != -1:
                self.tracked_objects[tid]["centroid_history"].append(centroids[min_idx])
                self.tracked_objects[tid]["box"] = boxes[min_idx]
                self.tracked_objects[tid]["frames_missing"] = 0
                matched_tracks.add(tid)
                matched_centroids.add(min_idx)
            else:
                self.tracked_objects[tid]["frames_missing"] += 1

        for tid in list(self.tracked_objects.keys()):
            if self.tracked_objects[tid]["frames_missing"] > 10:
                del self.tracked_objects[tid]

        for i, (c, b) in enumerate(zip(centroids, boxes)):
            if i not in matched_centroids:
                self.tracked_objects[self.next_id] = {
                    "centroid_history": [c],
                    "box": b,
                    "frames_missing": 0,
                }
                self.next_id += 1

    def check_line_crossing(self, line_y, frame_count, w):
        crossed = []
        for tid, track in self.tracked_objects.items():
            if tid in self.counted_ids or len(track["centroid_history"]) < 2:
                continue

            px, py = track["centroid_history"][-2]
            cx, cy = track["centroid_history"][-1]
            crossed_line = False

            if self.use_diagonal_line and w:
                if py < self.line_position * px and cy >= self.line_position * cx:
                    crossed_line = True
            elif py < line_y <= cy:
                crossed_line = True

            if crossed_line:
                is_dup = any(
                    np.sqrt(
                        (cx - self.tracked_objects[oid]["centroid_history"][-1][0]) ** 2
                        + (cy - self.tracked_objects[oid]["centroid_history"][-1][1])
                        ** 2
                    )
                    < 80
                    and frame_count - self.counted_times.get(oid, 0) < 10
                    for oid in self.counted_ids
                    if oid in self.tracked_objects
                )

                if not is_dup:
                    self.counted_ids.add(tid)
                    self.counted_times[tid] = frame_count
                    self.total_count += 1
                    crossed.append(tid)
        return crossed

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open {self.video_path}")
            return 0

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        line_y = int(h * self.line_position)

        os.makedirs("output", exist_ok=True)
        out = cv2.VideoWriter(
            "output/counted_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

        print("=" * 60)
        print("BAG COUNTER")
        print("=" * 60)
        print(f"Video: {w}x{h} @ {fps}fps, {total_frames} frames")
        print(
            f"Line: {'Diagonal' if self.use_diagonal_line else 'Horizontal'} | Areas: {self.min_area}-{self.max_area}"
        )
        print("-" * 60)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            centroids, boxes = self.get_centroids(frame)
            self.update_tracks(centroids, boxes)
            crossed_ids = self.check_line_crossing(line_y, frame_count, w)

            if crossed_ids:
                for i, cid in enumerate(crossed_ids):
                    print(
                        f"✅ Bag #{self.total_count - len(crossed_ids) + i + 1} (ID:{cid}) | Frame {frame_count}"
                    )

            # Draw line
            margin = int(w * (1 - self.line_length_ratio) / 2)
            x1, x2 = margin, w - margin
            if self.use_diagonal_line:
                y1, y2 = int(self.line_position * x1), int(self.line_position * x2)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            else:
                cv2.line(frame, (x1, line_y), (x2, line_y), (0, 255, 255), 3)

            # Draw ROI
            if self.roi:
                cv2.rectangle(
                    frame,
                    (self.roi[0], self.roi[1]),
                    (self.roi[2], self.roi[3]),
                    (255, 0, 255),
                    2,
                )

            # Draw tracked objects
            for tid, track in self.tracked_objects.items():
                x, y, tw, th = track["box"]
                c = track["centroid_history"][-1]
                color = (0, 255, 0) if tid in self.counted_ids else (255, 0, 0)
                cv2.rectangle(frame, (x, y), (x + tw, y + th), color, 2)
                cv2.circle(frame, c, 5, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"ID:{tid}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            cv2.putText(
                frame,
                f"Count: {self.total_count}",
                (10, h - 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )
            cv2.putText(
                frame,
                f"Frame: {frame_count}/{total_frames}",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            out.write(frame)

            if frame_count % 30 == 0:
                print(
                    f"Progress: {frame_count}/{total_frames} | Count: {self.total_count} | Active: {len(self.tracked_objects)}"
                )

        cap.release()
        out.release()

        with open("output/count_log.txt", "w") as f:
            f.write(
                f"Results\nFrames: {frame_count}\nTotal: {self.total_count}\nIDs: {sorted(self.counted_ids)}\n"
            )

        print("-" * 60)
        print(f"DONE! Counted: {self.total_count} bags")
        print("=" * 60)
        return self.total_count


def main():
    VIDEO_PATH = "./input/رصيف التحميل .mp4"
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found: {VIDEO_PATH}")
        return
    counter = MotionBagCounter(
        VIDEO_PATH,
        line_position=0.5,
        min_area=800,
        max_area=50000,
        roi=(400, 300, 1550, 750),
        use_diagonal_line=True,
        line_length_ratio=0.6,
    )
    total = counter.run()
    print(f"\nFINAL COUNT: {total} bags\n")


if __name__ == "__main__":
    main()
