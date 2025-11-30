import cv2
from ultralytics import YOLO
import wikipedia
import threading
import time
import webbrowser
from collections import Counter
from wikipedia.exceptions import DisambiguationError, PageError, WikipediaException
import numpy as np

# --- CẤU HÌNH ---
wikipedia.set_lang("en")

MODEL_PATH = r"D:\best_new.pt"
TRACKER_CONFIG_PATH = r"D:\bytetrack.yaml"

model = YOLO(MODEL_PATH)

STABILITY_THRESHOLD = 5
AUTO_FREEZE_THRESHOLD = 5
CONF_FREEZE_LIMIT = 0.60
YOLO_CONF_THRESHOLD = 0.35
MAX_MISSES = 10

# --- BIẾN TOÀN CỤC ---
tracked_fishes = {}
current_stable_label = "No Fish Detected"
last_update_time = time.time()
last_max_conf = 0.0
last_stable_box = None
wiki_cache = {}
wiki_threads = set()
cache_lock = threading.Lock()
is_frozen = False
frozen_frame = None
current_link = ""

# --- HÀM TRA CỨU WIKIPEDIA ---
def fetch_wiki(label):
    try:
        results = wikipedia.search(label)
        with cache_lock:
            if results:
                page = wikipedia.page(results[0], auto_suggest=False)
                wiki_cache[label] = page.url
            else:
                wiki_cache[label] = "No Wiki"
    except (DisambiguationError, PageError):
        with cache_lock:
            wiki_cache[label] = "No Wiki"
    except WikipediaException:
        with cache_lock:
            wiki_cache[label] = "Wiki API Error"
    except Exception:
        with cache_lock:
            wiki_cache[label] = "General Error"
    finally:
        wiki_threads.discard(label)

# --- HÀM XỬ LÝ YOLO VÀ HIỂN THỊ ---
def process_yolo_and_display(frame, is_frozen_mode):
    global tracked_fishes, current_stable_label, last_update_time
    global last_max_conf, current_link, last_stable_box, wiki_cache, wiki_threads

    freeze_signal = False
    results = [] if is_frozen_mode else model.track(
        source=frame,
        conf=YOLO_CONF_THRESHOLD,
        persist=True,
        tracker=TRACKER_CONFIG_PATH,
        verbose=False
    )

    best_conf_seen_in_frame = 0.0
    best_stable_label_in_frame = "No Fish Detected"
    best_box_in_frame = None
    best_track_id_in_frame = None
    current_ids = set()

    # Cập nhật trạng thái theo track_id
    for r in results:
        boxes = r.boxes
        if boxes is not None and boxes.id is not None:
            for box, cls_id, conf, track_id in zip(boxes.xyxy, boxes.cls, boxes.conf, boxes.id):
                track_id = int(track_id.item())
                confidence = float(conf.item())
                label_en = model.names[int(cls_id.item())]
                current_ids.add(track_id)
                box_float = [float(b.item()) for b in box]

                if track_id not in tracked_fishes:
                    tracked_fishes[track_id] = {
                        "stable_label": label_en,
                        "conf": confidence,
                        "box": [int(b) for b in box_float],
                        "history": [label_en],
                        "high_conf_frames": 0,
                        "best_conf_seen": confidence,
                        "box_history": [box_float],
                        "missed_frames": 0
                    }
                else:
                    fish_data = tracked_fishes[track_id]
                    fish_data["conf"] = confidence
                    fish_data["missed_frames"] = 0
                    fish_data["box_history"].append(box_float)
                    if len(fish_data["box_history"]) > STABILITY_THRESHOLD:
                        fish_data["box_history"].pop(0)

                    avg_box = np.mean(fish_data["box_history"], axis=0).astype(int)
                    fish_data["box"] = avg_box.tolist()

                    if confidence > fish_data["best_conf_seen"]:
                        fish_data["best_conf_seen"] = confidence

                    fish_data["history"].append(label_en)
                    if len(fish_data["history"]) > STABILITY_THRESHOLD:
                        fish_data["history"].pop(0)

                    most_common = Counter(fish_data["history"]).most_common(1)[0]
                    if most_common[1] >= STABILITY_THRESHOLD / 2:
                        fish_data["stable_label"] = most_common[0]

                    if confidence >= CONF_FREEZE_LIMIT:
                        fish_data["high_conf_frames"] += 1
                        if fish_data["high_conf_frames"] >= AUTO_FREEZE_THRESHOLD:
                            freeze_signal = True
                    else:
                        fish_data["high_conf_frames"] = 0

                if tracked_fishes[track_id]["best_conf_seen"] > best_conf_seen_in_frame:
                    best_conf_seen_in_frame = tracked_fishes[track_id]["best_conf_seen"]
                    best_stable_label_in_frame = tracked_fishes[track_id]["stable_label"]
                    best_box_in_frame = tracked_fishes[track_id]["box"]
                    best_track_id_in_frame = track_id

    # Xử lý các track bị mất dấu
    ids_to_remove = []
    for track_id, fish_data in list(tracked_fishes.items()):
        if track_id not in current_ids:
            fish_data["missed_frames"] += 1
            if fish_data["missed_frames"] > MAX_MISSES:
                ids_to_remove.append(track_id)
            elif track_id == best_track_id_in_frame:
                best_conf_seen_in_frame = fish_data["best_conf_seen"]
                best_stable_label_in_frame = fish_data["stable_label"]
                best_box_in_frame = fish_data["box"]

    for id_remove in ids_to_remove:
        del tracked_fishes[id_remove]

    current_stable_label = best_stable_label_in_frame
    last_max_conf = best_conf_seen_in_frame
    last_stable_box = best_box_in_frame if best_box_in_frame else None

    # Tra cứu Wikipedia
    label_raw = current_stable_label
    label_search = label_raw.split('-')[0].strip() if '-' in label_raw else label_raw
    is_parent_search = '-' in label_raw

    if label_search != "No Fish Detected":
        if time.time() - last_update_time > 1.0 or is_frozen_mode or freeze_signal:
            last_update_time = time.time()
            if label_search not in wiki_cache and label_search not in wiki_threads:
                wiki_threads.add(label_search)
                threading.Thread(target=fetch_wiki, args=(label_search,), daemon=True).start()

    # Vẽ bounding box
    if current_stable_label != "No Fish Detected" and last_stable_box:
        x1, y1, x2, y2 = last_stable_box
        color = (255, 0, 255) if is_frozen_mode or freeze_signal else (0, 255, 0)
        conf_display = f"Conf: {last_max_conf:.2f}"
        status_text = f"{current_stable_label} | {conf_display}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(status_text, font, font_scale, thickness)
        p1 = (x1, y1)
        p2 = (x1 + text_width, y1 - text_height - baseline - 5)

        cv2.rectangle(frame, p1, p2, color, -1)
        cv2.putText(frame, status_text, (x1, y1 - baseline - 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    with cache_lock:
        current_link = wiki_cache.get(label_search, "")

    fish_name_text = f"Class: {label_raw} (Conf: {last_max_conf:.2f})" if label_search != "No Fish Detected" else "Status: No Fish Detected"

    if label_search == "No Fish Detected":
        action_text = "Move fish into camera view."
    elif current_link in ["", "Wiki API Error", "General Error"]:
        action_text = f"Wiki Status: {current_link if current_link else 'Loading...'}"
    elif current_link == "No Wiki":
        action_text = f"Wiki Status: Page not found for '{label_search}'"
    else:
        action_text = "Press 'L' to Open Wiki Link" if not is_parent_search else f"Category: {label_search}. Press 'L' to Open Link"

    cv2.rectangle(frame, (10, frame.shape[0] - 70), (frame.shape[1] - 10, frame.shape[0] - 10), (0, 0, 0), -1)
    cv2.putText(frame, fish_name_text, (15, frame.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, action_text, (15, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame, current_link, freeze_signal

# --- HÀM MỞ CAMERA ---
def setup_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Error: Could not open webcam index 0. Trying index 1.")
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print("Final Error: Could not open webcam. Exiting.")
            return None
    print("Webcam successfully opened.")
    return cap

# --- MAIN LOOP ---
cap = setup_camera()
if cap is None:
    exit()

print("Press 'q' to quit. Press 'L' to open Wikipedia link. Press 'SPACE' to freeze/unfreeze.")

while cap.isOpened():
    if not is_frozen:
        ret, frame = cap.read()
        if not ret:
            break
        frame_display = frame.copy()
        frame_display, _, freeze_signal = process_yolo_and_display(frame_display, is_frozen)

        if freeze_signal:
            is_frozen = True
            frozen_frame = frame_display.copy()
            print("Mode: AUTO-FROZEN. Press SPACE to Unfreeze.")
            if current_stable_label != "No Fish Detected":
                label_search = current_stable_label.split('-')[0].strip()
                with cache_lock:
                    current_link = wiki_cache.get(label_search, "")
    else:
        if frozen_frame is None:
            key = cv2.waitKey(1) & 0xFF
            continue
        frame_display = frozen_frame.copy()

    cv2.imshow("YOLOv8 + Wikipedia", frame_display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    if key == ord('l'):
        if current_link and current_link not in ["", "No Wiki", "Wiki API Error", "General Error"]:
            print(f"Opening Wiki link for {current_stable_label}: {current_link}")
            webbrowser.open(current_link)
    if key == ord(' '):
        if not is_frozen:
            is_frozen = True
            frozen_frame = frame_display.copy()
            print("Mode: FROZEN (Manual). Press SPACE to Unfreeze.")
            if current_stable_label != "No Fish Detected":
                label_search = current_stable_label.split('-')[0].strip()
                with cache_lock:
                    current_link = wiki_cache.get(label_search, "")
        else:
            is_frozen = False
            frozen_frame = None
            print("Mode: LIVE. Processing real-time video.")
            tracked_fishes = {}
            wiki_cache = {}
            current_stable_label = "No Fish Detected"
            last_max_conf = 0.0
            last_stable_box = None

cap.release()
cv2.destroyAllWindows()
