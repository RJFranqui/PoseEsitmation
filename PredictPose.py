import cv2
import torch
import joblib
import numpy as np
from xgboost import XGBClassifier
from PoseModYOLO import load_yolo_model, PoseDetector
from FrameDataset import SquatKneeFrameDataset
from mlp import SquatMLP
from BaseClassifier import PhaseLSTM
import torch.nn.functional as F
from collections import deque

def load_models():
    mlp_model = SquatMLP(input_dim=4)
    mlp_model.load_state_dict(torch.load("models/mlp_model.pth", map_location='cpu'))
    mlp_model.eval()

    rf_model = joblib.load("models/best_rf_model.pkl")
    xgb_model = joblib.load("models/best_xgb_model.pkl")

    lstm_model = PhaseLSTM(input_size=8)
    lstm_model.load_state_dict(torch.load("models/lstm_model.pth", map_location='cpu'))
    lstm_model.eval()

    return mlp_model, rf_model, xgb_model, lstm_model

def extract_features(lmList):
    joints = {}
    for lm in lmList:
        id, _, _, x, y, z, world_x, world_y, world_z = lm
        joints[id] = np.array([world_x, world_y, world_z])

    try:
        left_thigh = joints[23] - joints[25]
        left_shin = joints[27] - joints[25]
        right_thigh = joints[24] - joints[26]
        right_shin = joints[28] - joints[26]
        torso = (joints[11] + joints[12]) / 2 - (joints[23] + joints[24]) / 2

        def safe_angle(v1, v2):
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            return np.clip(np.arccos(cos), 0, np.pi)

        left_knee = safe_angle(left_thigh, left_shin)
        right_knee = safe_angle(right_thigh, right_shin)
        left_hip = safe_angle(torso, left_thigh)
        torso_lean = np.arctan2(torso[1], abs(torso[2]))

        return np.array([left_knee, right_knee, left_hip, torso_lean])
    except:
        return None

lstm_buffer = deque(maxlen=30)
raw_feature_buffer = deque(maxlen=1)  # stores only latest raw features

def run_video(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = PoseDetector()
    yolo_net, output_layers, class_names = load_yolo_model()
    mlp_model, rf_model, xgb_model, lstm_model = load_models()

    scaler_4 = joblib.load("models/scaler.pkl")             # for MLP, RF, XGB (4 features)
    scaler_8 = joblib.load("models/lstm_scaler.pkl")        # for LSTM (8 features)
    print("Scalers loaded successfully!")

    frame_width, frame_height = 800, 600
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    output_path = video_path.split('/')[-1].replace('.mov', '_predictions.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        yolo_net.setInput(blob)
        outputs = yolo_net.forward(output_layers)

        height, width = frame.shape[:2]
        boxes = []

        for outp in outputs:
            for detection in outp:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_names[class_id] == "person":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    boxes.append([x, y, w, h])

        if boxes:
            frame_center = np.array([width // 2, height // 2])
            best_box = min(boxes, key=lambda b: np.linalg.norm(np.array([b[0]+b[2]//2, b[1]+b[3]//2]) - frame_center))

            if best_box:
                x, y, w, h = best_box
                x, y = max(0, x), max(0, y)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cropped = frame[y:y+h, x:x+w]
                detector.findPose(cropped)
                lmList = detector.findPosition(cropped)

                if lmList:
                    features = extract_features(lmList)
                    if features is not None:
                        # ==== Frame models (4 features)
                        scaled_features = scaler_4.transform(features.reshape(1, -1))
                        temp_features = scaled_features.flatten()

                        mlp_pred = torch.softmax(mlp_model(torch.tensor(scaled_features, dtype=torch.float32)), dim=1)
                        mlp_label = "UP" if mlp_pred.argmax() == 1 else "DOWN"

                        rf_label = "UP" if rf_model.predict(temp_features.reshape(1, -1))[0] == 1 else "DOWN"
                        xgb_label = "UP" if xgb_model.predict(temp_features.reshape(1, -1))[0] == 1 else "DOWN"

                        # ==== LSTM (8 features)
                        if len(raw_feature_buffer) > 0:
                            last_raw_features = raw_feature_buffer[-1]
                            delta = features - last_raw_features
                        else:
                            delta = np.zeros_like(features)

                        # Now store current raw features
                        raw_feature_buffer.append(features)

                        # Concatenate angles + deltas
                        full_features = np.concatenate([features, delta])

                        # Now scale the 8 features properly
                        scaled_lstm_input = scaler_8.transform(full_features.reshape(1, -1)).flatten()

                        # Append scaled to lstm_buffer
                        lstm_buffer.append(scaled_lstm_input)


                        if len(lstm_buffer) == 30:
                            lstm_seq = torch.tensor(np.stack(lstm_buffer)).unsqueeze(0).float()
                            lstm_out = torch.softmax(lstm_model(lstm_seq), dim=1)
                            confidence = lstm_out.max().item()
                            pred_class = lstm_out.argmax().item()
                            lstm_label = "UP" if pred_class == 1 else "DOWN"
                            # Optional: add confidence threshold
                            if confidence < 0.6:
                                lstm_label = "---"
                        else:
                            lstm_label = "---"

                        y_offset = 30
                        colors = {
                            "MLP": (255, 0, 0),
                            "RF": (0, 255, 0),
                            "XGB": (0, 0, 255),
                            "LSTM": (0, 255, 255)
                        }

                        for model_name, pred_label in zip(["MLP", "RF", "XGB", "LSTM"], [mlp_label, rf_label, xgb_label, lstm_label]):
                            color = colors.get(model_name, (255, 255, 255))
                            cv2.putText(frame, f"{model_name}: {pred_label}", (10, y_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            y_offset += 30

        resized = cv2.resize(frame, (800, 600))
        cv2.imshow("Squat Prediction", resized)
        out.write(resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_video("AllVids/Fort3.mov")
