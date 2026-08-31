import hashlib
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except Exception as exc:
    YOLO = None
    YOLO_IMPORT_ERROR = exc
else:
    YOLO_IMPORT_ERROR = None


BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "data" / "HelmetHead"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "best.pt"
DEFAULT_POSE_MODEL_PATH = BASE_DIR / "models" / "yolov26n-pose.pt"
TRAIN_REPORT_DIR = BASE_DIR / "assets" / "train"
VAL_REPORT_DIR = BASE_DIR / "assets" / "train"
DETECT_SAMPLE_DIR = BASE_DIR / "data" / "samples"
VIOLATION_SAVE_DIR = BASE_DIR / "violations"
EXPORT_DIR = BASE_DIR / "export"
UPLOAD_MODEL_DIR = BASE_DIR / "uploaded_models"

for directory in [VIOLATION_SAVE_DIR, EXPORT_DIR, UPLOAD_MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


RAW_CLASS_LABELS = {
    "head": "未佩戴安全帽",
    "no_hat": "未佩戴安全帽",
    "no-helmet": "未佩戴安全帽",
    "person_head": "未佩戴安全帽",
    "helmet": "已佩戴安全帽",
    "hat": "已佩戴安全帽",
    "hardhat": "已佩戴安全帽",
}
VIOLATION_KEYS = {"head", "no_hat", "no-helmet", "person_head"}
POSE_VIOLATION_LABELS = {"危险俯身/临边探空", "疑似倒地/跌倒"}
DEFAULT_POSE_CONF = 0.50
DEFAULT_BEND_ANGLE_THRESHOLD = 45
DEFAULT_FALL_ANGLE_THRESHOLD = 65
DEFAULT_FALL_WIDTH_HEIGHT_RATIO = 1.25
KEYPOINT_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 5), (0, 6),
]


st.set_page_config(
    page_title="智慧工地人员安全行为识别与预警系统",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.main {background-color: #ffffff; font-family: "Microsoft YaHei", "Segoe UI", sans-serif;}
.block-container {padding: 1rem 2rem 2rem;}
.system-title {text-align: center; font-size: 31px; font-weight: 700; color: #1f2937; margin: 0 0 1rem;}
.metric-card {background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 8px; padding: 0.8rem; text-align: center;}
.metric-card h4 {margin: 0; color: #64748b; font-size: 14px; font-weight: 600;}
.metric-card p {margin: 0.35rem 0 0; font-size: 18px; font-weight: 700; color: #0f172a;}
.alert-normal {background: #dcfce7; color: #166534; padding: 0.8rem; border-radius: 8px; text-align: center; font-weight: 700;}
.alert-warning {background: #fff7ed; color: #9a3412; padding: 0.8rem; border-radius: 8px; text-align: center; font-weight: 700;}
.alert-danger {background: #fee2e2; color: #991b1b; padding: 0.8rem; border-radius: 8px; text-align: center; font-weight: 700;}
.thin-note {color: #64748b; font-size: 13px;}
.path-box {font-family: Consolas, monospace; font-size: 12px; background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 8px; padding: 0.6rem; overflow-wrap: anywhere;}
.small-title {font-size: 18px; font-weight: 700; color: #111827; margin: 0.4rem 0;}
.stDataFrame {background: white; border-radius: 8px;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "menu": "图片检测",
        "model_path": str(DEFAULT_MODEL_PATH),
        "pose_model_path": str(DEFAULT_POSE_MODEL_PATH),
        "violation_logs": [],
        "total_violations": 0,
        "helmet_violation_count": 0,
        "pose_violation_count": 0,
        "processed_images": set(),
        "last_helmet_state": True,
        "last_pose_state": True,
        "last_pose_desc": "正常姿态",
        "continuous_violation_frames": 0,
        "continuous_pose_frames": 0,
        "last_alert_time": datetime.min,
        "current_fps": 0.0,
        "current_target_count": 0,
        "last_coordinates": {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0},
        "last_source_name": "",
        "last_detection_source": "",
        "last_detection_image": None,
        "last_detection_rows": pd.DataFrame(),
        "last_detection_violation": False,
        "last_detection_alert_text": "",
        "camera_error_logs": [],
        "detection_results": pd.DataFrame(
            columns=["序号", "图片/视频名称", "类别", "置信度", "坐标位置", "检测时间"]
        ),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def has_cuda() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    if YOLO is None:
        raise RuntimeError(f"ultralytics 导入失败：{YOLO_IMPORT_ERROR}")
    return YOLO(model_path)


def get_model(model_path: str):
    if not model_path or not Path(model_path).exists():
        return None, f"模型文件不存在：{model_path}"
    try:
        return load_model(model_path), "模型加载成功"
    except Exception as exc:
        return None, f"模型加载失败：{exc}"


@st.cache_resource(show_spinner=False)
def load_pose_model(model_path: str):
    if YOLO is None:
        raise RuntimeError(f"ultralytics 导入失败：{YOLO_IMPORT_ERROR}")
    return YOLO(model_path)


def get_pose_model(model_path: str):
    if not model_path or not Path(model_path).exists():
        return None, f"姿态模型文件不存在：{model_path}"
    try:
        return load_pose_model(model_path), "姿态模型加载成功"
    except Exception as exc:
        return None, f"姿态模型加载失败：{exc}"


def normalize_key(raw_name: str) -> str:
    text = str(raw_name).strip().lower().replace(" ", "_")
    if text in RAW_CLASS_LABELS:
        return text
    if "no" in text and ("hat" in text or "helmet" in text):
        return "no_hat"
    if "helmet" in text or "hardhat" in text or text == "hat":
        return "helmet"
    if "head" in text:
        return "head"
    return text


def chinese_label(raw_name: str) -> str:
    return RAW_CLASS_LABELS.get(normalize_key(raw_name), str(raw_name))


def class_color(raw_name: str) -> tuple[int, int, int]:
    key = normalize_key(raw_name)
    if key in VIOLATION_KEYS:
        return (36, 36, 220)
    if key == "helmet":
        return (38, 160, 70)
    return (220, 130, 30)


def pose_color(label: str) -> tuple[int, int, int]:
    if "倒地" in label or "跌倒" in label:
        return (128, 0, 220)
    if "俯身" in label or "临边" in label:
        return (0, 110, 255)
    return (38, 160, 70)


def read_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    uploaded_file.seek(0)
    return image


def image_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def append_detection_rows(rows: list[dict]) -> None:
    if not rows:
        return
    next_index = len(st.session_state.detection_results) + 1
    normalized = []
    for offset, row in enumerate(rows):
        normalized.append(
            {
                "序号": next_index + offset,
                "图片/视频名称": row["source"],
                "类别": row["label"],
                "置信度": f"{row['conf']:.3f}",
                "坐标位置": row["box"],
                "检测时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    st.session_state.detection_results = pd.concat(
        [st.session_state.detection_results, pd.DataFrame(normalized)],
        ignore_index=True,
    )


def add_violation_log(
    message: str,
    frame_bgr: np.ndarray | None,
    level: str = "一般预警",
    violation_type: str = "helmet",
) -> None:
    now = datetime.now()
    if (now - st.session_state.last_alert_time).total_seconds() < 1.5:
        return
    st.session_state.last_alert_time = now
    st.session_state.total_violations += 1
    if violation_type in {"helmet", "both"}:
        st.session_state.helmet_violation_count += 1
    if violation_type in {"pose", "both"}:
        st.session_state.pose_violation_count += 1
    save_path = ""
    if frame_bgr is not None:
        save_path = str(VIOLATION_SAVE_DIR / f"{now.strftime('%Y%m%d_%H%M%S')}_{violation_type}_violation.jpg")
        cv2.imwrite(save_path, frame_bgr)
    st.session_state.violation_logs.append(
        {
            "预警时间": now.strftime("%Y-%m-%d %H:%M:%S"),
            "预警级别": level,
            "违规内容": message,
            "截图路径": save_path,
            "处理状态": "未处理",
        }
    )


def add_camera_error(message: str) -> None:
    timestamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    logs = st.session_state.camera_error_logs
    if not logs or not logs[-1].endswith(message):
        logs.append(timestamped)
    st.session_state.camera_error_logs = logs[-50:]


def render_camera_error_logs(container) -> None:
    logs = st.session_state.camera_error_logs[-10:]
    if logs:
        container.text_area("系统错误日志（最近10条）", value="\n".join(logs), height=180, disabled=True)
    else:
        container.info("暂无摄像头错误日志。")


def open_camera(camera_index: int, frame_width: int):
    backends = [
        ("DirectShow", getattr(cv2, "CAP_DSHOW", None)),
        ("Media Foundation", getattr(cv2, "CAP_MSMF", None)),
        ("默认后端", None),
    ]
    for backend_name, backend_id in backends:
        cap = cv2.VideoCapture(camera_index) if backend_id is None else cv2.VideoCapture(camera_index, backend_id)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(frame_width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(frame_width * 9 / 16))
            cap.set(cv2.CAP_PROP_FPS, 30)
            return cap, backend_name
        cap.release()
    return None, ""


def model_names(result) -> dict:
    names = getattr(result, "names", None)
    if isinstance(names, dict):
        return names
    return {}


def keypoint_xyc(keypoints) -> np.ndarray:
    data = keypoints.data.detach().cpu().numpy()
    if data.ndim == 3:
        data = data[0]
    return data


def visible_points(kps: np.ndarray, indices: list[int], min_conf: float = 0.35) -> list[np.ndarray]:
    points = []
    for index in indices:
        if index < len(kps) and kps[index][2] >= min_conf:
            points.append(kps[index])
    return points


def midpoint(points: list[np.ndarray]) -> np.ndarray | None:
    if not points:
        return None
    return np.mean(np.array([[p[0], p[1]] for p in points]), axis=0)


def torso_angle_from_vertical(shoulder_center: np.ndarray, hip_center: np.ndarray) -> float:
    dx = abs(float(shoulder_center[0] - hip_center[0]))
    dy = abs(float(shoulder_center[1] - hip_center[1]))
    if dy < 1e-6:
        return 90.0
    return float(np.degrees(np.arctan(dx / dy)))


def check_dangerous_pose(
    keypoints,
    frame_height: int,
    bend_angle_threshold: float,
    fall_angle_threshold: float,
    fall_ratio_threshold: float,
) -> tuple[bool, str, float]:
    kps = keypoint_xyc(keypoints)
    if len(kps) < 17:
        return False, "正常姿态", 0.0

    shoulders = visible_points(kps, [5, 6], 0.35)
    hips = visible_points(kps, [11, 12], 0.35)
    core_points = visible_points(kps, [5, 6, 11, 12], 0.35)
    all_visible = visible_points(kps, list(range(17)), 0.30)
    if len(core_points) < 3 or len(all_visible) < 5:
        return False, "正常姿态", 0.0

    shoulder_center = midpoint(shoulders)
    hip_center = midpoint(hips)
    if shoulder_center is None or hip_center is None:
        return False, "正常姿态", 0.0

    torso_angle = torso_angle_from_vertical(shoulder_center, hip_center)
    xy = np.array([[p[0], p[1]] for p in all_visible])
    width = float(np.max(xy[:, 0]) - np.min(xy[:, 0]))
    height = float(np.max(xy[:, 1]) - np.min(xy[:, 1]))
    ratio = width / max(height, 1.0)

    nose = kps[0] if len(kps) > 0 and kps[0][2] >= 0.30 else None
    ankle_points = visible_points(kps, [15, 16], 0.30)
    knee_points = visible_points(kps, [13, 14], 0.30)
    foot_center = midpoint(ankle_points or knee_points)

    is_fallen = torso_angle >= fall_angle_threshold and ratio >= fall_ratio_threshold
    if foot_center is not None and nose is not None:
        vertical_span = abs(float(foot_center[1] - nose[1]))
        is_fallen = is_fallen or (ratio >= fall_ratio_threshold and vertical_span < frame_height * 0.35)
    if is_fallen:
        return True, "疑似倒地/跌倒", torso_angle

    is_bending = torso_angle >= bend_angle_threshold
    if nose is not None:
        is_bending = is_bending and (nose[1] >= shoulder_center[1] or shoulder_center[1] < frame_height * 0.35)
    if is_bending:
        return True, "危险俯身/临边探空", torso_angle

    return False, "正常姿态", torso_angle


def draw_pose_keypoints(frame_bgr: np.ndarray, keypoints, label: str, angle: float) -> None:
    kps = keypoint_xyc(keypoints)
    color = pose_color(label)
    visible_xy = []
    for point in kps:
        if point[2] < 0.30:
            continue
        x, y = int(point[0]), int(point[1])
        visible_xy.append((x, y))
        cv2.circle(frame_bgr, (x, y), 3, color, -1)

    for start, end in KEYPOINT_CONNECTIONS:
        if start >= len(kps) or end >= len(kps):
            continue
        if kps[start][2] >= 0.30 and kps[end][2] >= 0.30:
            cv2.line(
                frame_bgr,
                (int(kps[start][0]), int(kps[start][1])),
                (int(kps[end][0]), int(kps[end][1])),
                color,
                2,
            )

    if visible_xy:
        xs = [p[0] for p in visible_xy]
        ys = [p[1] for p in visible_xy]
        cv2.putText(
            frame_bgr,
            f"{label} {angle:.0f}deg",
            (min(xs), max(20, min(ys) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def predict_pose_frame(
    pose_model,
    frame_bgr: np.ndarray,
    source_name: str,
    pose_conf: float,
    bend_angle_threshold: float,
    fall_angle_threshold: float,
    fall_ratio_threshold: float,
    target_filter: str,
) -> tuple[np.ndarray, list[dict], bool, str]:
    if pose_model is None:
        st.session_state.last_pose_state = True
        st.session_state.last_pose_desc = "姿态模型未加载"
        return frame_bgr, [], False, "姿态模型未加载"

    predict_kwargs = {"source": frame_bgr, "conf": pose_conf, "verbose": False}
    if has_cuda():
        predict_kwargs["device"] = 0
    results = pose_model.predict(**predict_kwargs)

    pose_rows = []
    pose_violation_found = False
    pose_descs = []
    keypoints = getattr(results[0], "keypoints", None) if results else None
    if keypoints is None or len(keypoints.data) == 0:
        st.session_state.last_pose_state = True
        st.session_state.last_pose_desc = "未检测到人体姿态"
        return frame_bgr, pose_rows, False, "未检测到人体姿态"

    for index, kps in enumerate(keypoints):
        is_dangerous, desc, angle = check_dangerous_pose(
            kps,
            frame_bgr.shape[0],
            bend_angle_threshold,
            fall_angle_threshold,
            fall_ratio_threshold,
        )
        label = desc if is_dangerous else "正常姿态"
        draw_pose_keypoints(frame_bgr, kps, label, angle)
        if is_dangerous:
            pose_violation_found = True
            pose_descs.append(desc)
        if target_filter != "全部" and label != target_filter:
            continue
        pose_rows.append(
            {
                "source": source_name,
                "label": label,
                "conf": 1.0,
                "box": f"person#{index + 1}, torso_angle={angle:.1f}",
            }
        )

    st.session_state.last_pose_state = not pose_violation_found
    st.session_state.last_pose_desc = "、".join(sorted(set(pose_descs))) if pose_descs else "正常姿态"
    if pose_violation_found:
        st.session_state.continuous_pose_frames += 1
    else:
        st.session_state.continuous_pose_frames = 0

    return frame_bgr, pose_rows, pose_violation_found, st.session_state.last_pose_desc


def predict_frame(
    model,
    pose_model,
    frame_bgr: np.ndarray,
    source_name: str,
    conf_threshold: float,
    iou_threshold: float,
    line_width: int,
    pose_conf: float,
    bend_angle_threshold: float,
    fall_angle_threshold: float,
    fall_ratio_threshold: float,
    target_filter: str,
    save_violation: bool = True,
    append_results_to_history: bool = True,
) -> tuple[np.ndarray, pd.DataFrame, bool]:
    start = time.time()
    predict_kwargs = {
        "source": frame_bgr,
        "conf": conf_threshold,
        "iou": iou_threshold,
        "verbose": False,
    }
    if has_cuda():
        predict_kwargs["device"] = 0
    results = model.predict(**predict_kwargs)
    elapsed = max(time.time() - start, 1e-6)
    st.session_state.current_fps = 1.0 / elapsed

    annotated = frame_bgr.copy()
    rows = []
    helmet_violation_found = False
    names = model_names(results[0])

    boxes = getattr(results[0], "boxes", None)
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            raw_name = names.get(cls_id, str(cls_id))
            label = chinese_label(raw_name)
            key = normalize_key(raw_name)

            xmin, ymin, xmax, ymax = [int(v) for v in box.xyxy[0].detach().cpu().numpy()]
            color = class_color(raw_name)
            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, line_width)
            cv2.putText(
                annotated,
                f"{raw_name} {conf:.2f}",
                (xmin, max(20, ymin - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                max(1, line_width),
                cv2.LINE_AA,
            )

            if key in VIOLATION_KEYS:
                helmet_violation_found = True

            if target_filter != "全部" and label != target_filter:
                continue

            rows.append(
                {
                    "source": source_name,
                    "label": label,
                    "conf": conf,
                    "box": f"({xmin}, {ymin}, {xmax}, {ymax})",
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                }
            )

    annotated, pose_rows, pose_violation_found, pose_desc = predict_pose_frame(
        pose_model,
        annotated,
        source_name,
        pose_conf,
        bend_angle_threshold,
        fall_angle_threshold,
        fall_ratio_threshold,
        target_filter,
    )
    rows.extend(pose_rows)

    coordinate_rows = [row for row in rows if {"xmin", "ymin", "xmax", "ymax"}.issubset(row)]
    if coordinate_rows:
        last = coordinate_rows[-1]
        st.session_state.last_coordinates = {
            "xmin": last["xmin"],
            "ymin": last["ymin"],
            "xmax": last["xmax"],
            "ymax": last["ymax"],
        }
    elif not rows:
        st.session_state.last_coordinates = {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0}

    st.session_state.current_target_count = len(rows)
    st.session_state.last_helmet_state = not helmet_violation_found
    violation_found = helmet_violation_found or pose_violation_found
    if helmet_violation_found:
        st.session_state.continuous_violation_frames += 1
    else:
        st.session_state.continuous_violation_frames = 0

    if save_violation and violation_found:
        messages = []
        if helmet_violation_found:
            messages.append("检测到未佩戴安全帽目标")
        if pose_violation_found:
            messages.append(f"检测到{pose_desc}")
        level = "紧急告警" if pose_violation_found or st.session_state.continuous_violation_frames >= 3 else "一般预警"
        violation_type = "both" if helmet_violation_found and pose_violation_found else ("pose" if pose_violation_found else "helmet")
        add_violation_log("；".join(messages), annotated, level, violation_type)

    if append_results_to_history:
        append_detection_rows(rows)
    return annotated, pd.DataFrame(rows), violation_found


def image_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]


def dataset_stats() -> pd.DataFrame:
    rows = []
    for split in ["train", "valid", "test"]:
        image_dir = DATASET_ROOT / split / "images"
        label_dir = DATASET_ROOT / split / "labels"
        rows.append(
            {
                "数据划分": split,
                "图片数量": len(image_paths(image_dir)),
                "标签数量": len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0,
                "图片目录": str(image_dir),
            }
        )
    return pd.DataFrame(rows)


def latest_results_summary() -> pd.DataFrame:
    csv_path = TRAIN_REPORT_DIR / "results.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    row = df.iloc[-1].to_dict()
    picked = []
    for key, value in row.items():
        key_text = str(key).strip()
        if any(term in key_text.lower() for term in ["precision", "recall", "map", "box_loss", "cls_loss"]):
            picked.append({"指标": key_text, "数值": value})
    return pd.DataFrame(picked)


def show_report_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"未找到：{path.name}")


def export_dataframe(df: pd.DataFrame, stem: str) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = EXPORT_DIR / f"{stem}_{timestamp}.csv"
    xlsx_path = EXPORT_DIR / f"{stem}_{timestamp}.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception:
        xlsx_path = csv_path
    return csv_path, xlsx_path


def save_uploaded_model(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    digest = hashlib.md5(uploaded_file.getvalue()).hexdigest()[:10]
    target = UPLOAD_MODEL_DIR / f"{Path(uploaded_file.name).stem}_{digest}{suffix}"
    target.write_bytes(uploaded_file.getvalue())
    return str(target)


model, model_load_status = get_model(st.session_state.model_path)
pose_model, pose_model_load_status = get_pose_model(st.session_state.pose_model_path)
device_text = "GPU" if has_cuda() else "CPU"


with st.sidebar:
    st.markdown("### 系统功能导航")
    menu_list = [
        "系统设置",
        "图片检测",
        "摄像头检测",
        "文件夹批量检测",
        "上传模型",
        "预警记录管理",
        "结果展示",
        "结果导出",
    ]
    current_index = menu_list.index(st.session_state.menu) if st.session_state.menu in menu_list else 1
    st.session_state.menu = st.radio("系统功能导航", menu_list, index=current_index, label_visibility="collapsed")
    st.divider()

    st.markdown("### 安全帽检测参数")
    conf_threshold = st.slider("置信度 Conf", 0.05, 0.95, 0.25, step=0.05)
    iou_threshold = st.slider("交并比 IOU", 0.10, 0.95, 0.45, step=0.05)
    line_width = st.slider("检测框线宽", 1, 8, 2, step=1)
    st.divider()

    st.markdown("### 人体姿态检测参数")
    pose_conf = st.slider("姿态置信度", 0.10, 0.95, DEFAULT_POSE_CONF, step=0.05)
    bend_angle_threshold = st.slider("俯身角度阈值", 10, 80, DEFAULT_BEND_ANGLE_THRESHOLD, step=5)
    fall_angle_threshold = st.slider("倒地角度阈值", 40, 90, DEFAULT_FALL_ANGLE_THRESHOLD, step=5)
    fall_ratio_threshold = st.slider("倒地宽高比阈值", 0.80, 2.50, DEFAULT_FALL_WIDTH_HEIGHT_RATIO, step=0.05)
    st.divider()

    st.markdown("### 系统状态")
    st.metric("累计违规次数", st.session_state.total_violations)
    st.metric("安全帽违规", st.session_state.helmet_violation_count)
    st.metric("姿态违规", st.session_state.pose_violation_count)
    st.info(model_load_status)
    st.info(pose_model_load_status)
    st.write(f"运行设备：{device_text}")
    st.divider()

    if st.button("重置运行数据", use_container_width=True, type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


st.markdown('<div class="system-title">智慧工地人员安全行为识别与预警系统</div>', unsafe_allow_html=True)

top_col1, top_col2, top_col3, top_col4, top_col5 = st.columns(5)
with top_col1:
    status_color = "#16a34a" if st.session_state.last_helmet_state else "#dc2626"
    status_text = "合规" if st.session_state.last_helmet_state else "违规"
    st.markdown(
        f'<div class="metric-card"><h4>安全帽状态</h4><p style="color:{status_color};">{status_text}</p></div>',
        unsafe_allow_html=True,
    )
with top_col2:
    pose_color_text = "#16a34a" if st.session_state.last_pose_state else "#dc2626"
    st.markdown(
        f'<div class="metric-card"><h4>人员姿态状态</h4><p style="color:{pose_color_text};">{st.session_state.last_pose_desc}</p></div>',
        unsafe_allow_html=True,
    )
with top_col3:
    st.markdown(
        f'<div class="metric-card"><h4>当前目标数量</h4><p>{st.session_state.current_target_count} 个</p></div>',
        unsafe_allow_html=True,
    )
with top_col4:
    st.markdown(
        f'<div class="metric-card"><h4>实时 FPS</h4><p>{st.session_state.current_fps:.1f}</p></div>',
        unsafe_allow_html=True,
    )
with top_col5:
    st.markdown(
        f'<div class="metric-card"><h4>检测模型</h4><p>{"yolo26s best.pt" if Path(st.session_state.model_path).name == "best.pt" else Path(st.session_state.model_path).name}</p></div>',
        unsafe_allow_html=True,
    )

st.divider()
main_col, param_col = st.columns([3, 1])

with param_col:
    st.markdown("### 目标筛选")
    target_class = st.selectbox("检测目标类别", ["全部", "已佩戴安全帽", "未佩戴安全帽", "正常姿态", "危险俯身/临边探空", "疑似倒地/跌倒"])
    st.divider()
    st.markdown("### 目标位置坐标")
    coords = st.session_state.last_coordinates
    st.number_input("XMIN", value=int(coords["xmin"]), disabled=True)
    st.number_input("YMIN", value=int(coords["ymin"]), disabled=True)
    st.number_input("XMAX", value=int(coords["xmax"]), disabled=True)
    st.number_input("YMAX", value=int(coords["ymax"]), disabled=True)
    st.divider()
    st.markdown("### 当前模型")
    st.caption("安全帽检测")
    st.code(Path(st.session_state.model_path).name, language=None)
    st.caption("人体姿态")
    st.code(Path(st.session_state.pose_model_path).name, language=None)


if st.session_state.menu == "系统设置":
    with main_col:
        st.subheader("系统设置与检测原理")
        st.markdown("#### 运行方式")
        st.code(
            "streamlit run app.py\n"
            "默认模型：models/best.pt\n"
            "默认姿态模型：models/yolov26n-pose.pt\n"
            "默认数据：data/HelmetHead",
            language="powershell",
        )

        st.markdown("#### 检测原理")
        st.markdown(
            """
本系统采用 YOLO 单阶段目标检测流程：输入工地图片或视频帧后，模型在一次前向推理中同时完成目标定位和类别判断。
数据集使用 YOLO 标注格式，每行标签由 `class x_center y_center width height` 构成，坐标均为相对图片尺寸的归一化数值。

当前数据集包含两个类别：`head` 表示裸露头部，可作为未佩戴安全帽预警依据；`helmet` 表示安全帽目标，可作为合规佩戴依据。
前端在推理后读取检测框类别和置信度，若出现 `head` 类目标，则记录违规、保存截图，并在顶部状态卡片中切换为违规状态。

人体安全姿态检测使用 YOLO Pose 模型输出人体 17 个关键点，结合肩部、髋部、鼻子、膝踝等关键点计算躯干相对竖直方向的倾斜角和人体关键点包围框宽高比。
当躯干倾角超过俯身阈值时判定为“危险俯身/临边探空”；当躯干接近水平且人体宽高比异常偏大，或头脚垂直跨度过小时，判定为“疑似倒地/跌倒”。
            """
        )

        st.markdown("#### 数据集结构")
        st.dataframe(dataset_stats(), use_container_width=True, hide_index=True)

        st.markdown("#### 本地训练配置")
        st.code(
            "from ultralytics import YOLO\n\n"
            "model = YOLO('yolo26s.pt')\n"
            "model.train(data='safety_dataset.yaml', epochs=100, imgsz=640, batch=16, device=0, workers=8)",
            language="python",
        )

        current_path = st.text_input("当前安全帽模型路径", value=st.session_state.model_path)
        current_pose_path = st.text_input("当前姿态模型路径", value=st.session_state.pose_model_path)
        if st.button("应用模型路径", use_container_width=True):
            st.session_state.model_path = current_path
            st.session_state.pose_model_path = current_pose_path
            load_model.clear()
            load_pose_model.clear()
            st.rerun()


elif st.session_state.menu == "图片检测":
    with main_col:
        st.subheader("工地图片安全帽与人体安全姿态检测")
        if model is None:
            st.error(model_load_status)
        if pose_model is None:
            st.warning(pose_model_load_status)
        uploaded_image = st.file_uploader("上传工地图片", type=["jpg", "jpeg", "png", "bmp", "webp"])
        sample_paths = image_paths(DATASET_ROOT / "test" / "images")[:30]
        sample_choice = st.selectbox(
            "或选择测试集样例",
            ["不使用样例"] + [str(path) for path in sample_paths],
            format_func=lambda value: value if value == "不使用样例" else Path(value).name,
        )

        source_image = None
        source_name = ""
        if uploaded_image is not None:
            source_image = read_uploaded_image(uploaded_image)
            source_name = uploaded_image.name
        elif sample_choice != "不使用样例":
            source_image = cv2.imread(sample_choice)
            source_name = Path(sample_choice).name

        if source_image is not None:
            left, right = st.columns(2)
            with left:
                st.markdown('<div class="small-title">原始图片</div>', unsafe_allow_html=True)
                st.image(image_to_rgb(source_image), use_container_width=True)
            if st.button("开始图片检测", use_container_width=True, type="primary", disabled=model is None):
                annotated, rows, violation_found = predict_frame(
                    model,
                    pose_model,
                    source_image,
                    source_name,
                    conf_threshold,
                    iou_threshold,
                    line_width,
                    pose_conf,
                    bend_angle_threshold,
                    fall_angle_threshold,
                    fall_ratio_threshold,
                    target_class,
                )
                st.session_state.last_detection_source = source_name
                st.session_state.last_detection_image = annotated
                st.session_state.last_detection_rows = rows
                st.session_state.last_detection_violation = violation_found
                st.session_state.last_detection_alert_text = st.session_state.last_pose_desc
                st.rerun()

            has_current_result = (
                st.session_state.last_detection_source == source_name
                and st.session_state.last_detection_image is not None
            )
            if has_current_result:
                with right:
                    st.markdown('<div class="small-title">检测结果</div>', unsafe_allow_html=True)
                    st.image(image_to_rgb(st.session_state.last_detection_image), use_container_width=True)
                if st.session_state.last_detection_violation:
                    st.markdown('<div class="alert-danger">检测到安全帽或人体姿态违规，已写入预警记录。</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-normal">未发现未佩戴安全帽、危险俯身或倒地风险。</div>', unsafe_allow_html=True)
                rows = st.session_state.last_detection_rows
                st.dataframe(rows[["label", "conf", "box"]] if not rows.empty else rows, use_container_width=True)
        else:
            st.info("请上传图片，或从测试集样例中选择一张图片。")


elif st.session_state.menu == "摄像头检测":
    with main_col:
        st.subheader("摄像头实时视频安全监测与预警")
        if model is None:
            st.error(model_load_status)
        if pose_model is None:
            st.warning(pose_model_load_status)

        control_col1, control_col2, control_col3 = st.columns([1, 1, 1])
        with control_col1:
            run_camera = st.checkbox("开启摄像头实时监测", value=False)
        with control_col2:
            camera_index = st.number_input("摄像头编号", min_value=0, max_value=10, value=0, step=1)
        with control_col3:
            record_interval = st.slider("结果记录间隔（帧）", 1, 30, 5)

        frame_width = st.select_slider("采集分辨率宽度", options=[640, 800, 960, 1280], value=960)
        frame_container = st.empty()
        alert_container = st.empty()
        status_container = st.empty()
        result_container = st.empty()
        error_log_container = st.empty()

        render_camera_error_logs(error_log_container)

        if run_camera and model is not None:
            cap, camera_backend = open_camera(int(camera_index), int(frame_width))
            if cap is None:
                add_camera_error(f"摄像头 {int(camera_index)} 打开失败，请检查设备连接或编号。")
                alert_container.markdown('<div class="alert-danger">摄像头打开失败，请检查设备连接或编号。</div>', unsafe_allow_html=True)
                render_camera_error_logs(error_log_container)
            else:
                frame_count = 0
                last_rows = pd.DataFrame()
                status_container.info(f"实时监测运行中，摄像头后端：{camera_backend}。取消勾选后会在下一次页面刷新时停止。")

                try:
                    while run_camera:
                        loop_start = time.time()
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            add_camera_error("摄像头帧读取失败，已跳过当前帧。")
                            time.sleep(0.1)
                            continue

                        frame_count += 1
                        try:
                            annotated, rows, violation_found = predict_frame(
                                model,
                                pose_model,
                                frame,
                                "实时摄像头",
                                conf_threshold,
                                iou_threshold,
                                line_width,
                                pose_conf,
                                bend_angle_threshold,
                                fall_angle_threshold,
                                fall_ratio_threshold,
                                target_class,
                                save_violation=True,
                                append_results_to_history=(frame_count % record_interval == 0),
                            )
                        except Exception as exc:
                            add_camera_error(f"模型实时推理失败，已跳过当前帧：{str(exc)[:120]}")
                            frame_container.image(image_to_rgb(frame), use_container_width=True)
                            render_camera_error_logs(error_log_container)
                            time.sleep(0.05)
                            continue

                        elapsed = max(time.time() - loop_start, 1e-6)
                        st.session_state.current_fps = 1.0 / elapsed
                        last_rows = rows
                        frame_container.image(image_to_rgb(annotated), use_container_width=True)

                        if violation_found:
                            alert_container.markdown(
                                f'<div class="alert-danger">实时预警：安全帽或人体姿态存在风险（{st.session_state.last_pose_desc}）。</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            alert_container.markdown(
                                '<div class="alert-normal">实时监测正常：未发现未佩戴安全帽、危险俯身或倒地风险。</div>',
                                unsafe_allow_html=True,
                            )

                        status_container.info(
                            f"已监测 {frame_count} 帧 | FPS {st.session_state.current_fps:.1f} | "
                            f"目标 {st.session_state.current_target_count} 个 | 姿态：{st.session_state.last_pose_desc}"
                        )
                        if not last_rows.empty:
                            result_container.dataframe(
                                last_rows[["label", "conf", "box"]],
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            result_container.info("当前帧未检测到目标。")
                        render_camera_error_logs(error_log_container)
                        time.sleep(0.01)
                finally:
                    cap.release()
        elif run_camera and model is None:
            alert_container.markdown('<div class="alert-danger">安全帽检测模型未加载，无法启动实时监测。</div>', unsafe_allow_html=True)
        else:
            alert_container.markdown('<div class="alert-normal">摄像头实时监测未开启。</div>', unsafe_allow_html=True)


elif st.session_state.menu == "文件夹批量检测":
    with main_col:
        st.subheader("文件夹批量检测")
        default_dir = str(DATASET_ROOT / "test" / "images")
        folder = st.text_input("待检测图片文件夹", value=default_dir)
        max_images = st.slider("最多检测图片数", 1, 300, 30)
        save_outputs = st.checkbox("保存标注后的检测图片", value=True)

        if st.button("开始批量检测", use_container_width=True, type="primary", disabled=model is None):
            paths = image_paths(Path(folder))[:max_images]
            if not paths:
                st.warning("没有找到可检测图片。")
            else:
                output_dir = BASE_DIR / "runs" / "detect" / f"streamlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if save_outputs:
                    output_dir.mkdir(parents=True, exist_ok=True)
                progress = st.progress(0)
                tables = []
                preview = None
                for idx, path in enumerate(paths, start=1):
                    frame = cv2.imread(str(path))
                    if frame is None:
                        continue
                    annotated, rows, _ = predict_frame(
                        model,
                        pose_model,
                        frame,
                        path.name,
                        conf_threshold,
                        iou_threshold,
                        line_width,
                        pose_conf,
                        bend_angle_threshold,
                        fall_angle_threshold,
                        fall_ratio_threshold,
                        target_class,
                    )
                    if save_outputs:
                        cv2.imwrite(str(output_dir / path.name), annotated)
                    if not rows.empty:
                        tables.append(rows)
                    preview = annotated
                    progress.progress(idx / len(paths))

                if preview is not None:
                    st.image(image_to_rgb(preview), caption="最后一张检测结果", use_container_width=True)
                if tables:
                    st.dataframe(pd.concat(tables, ignore_index=True), use_container_width=True)
                st.success(f"批量检测完成，处理 {len(paths)} 张图片。")
                if save_outputs:
                    st.markdown(f'<div class="path-box">{output_dir}</div>', unsafe_allow_html=True)


elif st.session_state.menu == "上传模型":
    with main_col:
        st.subheader("上传与切换检测模型")
        model_kind = st.radio("模型类型", ["安全帽检测模型", "人体姿态模型"], horizontal=True)
        uploaded_model = st.file_uploader("上传 YOLO 模型文件", type=["pt", "onnx"])
        if uploaded_model is not None:
            if st.button("保存并使用该模型", use_container_width=True, type="primary"):
                new_path = save_uploaded_model(uploaded_model)
                if model_kind == "安全帽检测模型":
                    st.session_state.model_path = new_path
                    load_model.clear()
                else:
                    st.session_state.pose_model_path = new_path
                    load_pose_model.clear()
                st.success("模型已保存并切换。")
                st.rerun()

        st.markdown("#### 已上传模型")
        uploaded_models = sorted(UPLOAD_MODEL_DIR.glob("*.*"))
        if uploaded_models:
            selected = st.selectbox("选择模型", [str(path) for path in uploaded_models])
            switch_kind = st.radio("切换到", ["安全帽检测模型", "人体姿态模型"], horizontal=True)
            if st.button("切换到选中模型", use_container_width=True):
                if switch_kind == "安全帽检测模型":
                    st.session_state.model_path = selected
                    load_model.clear()
                else:
                    st.session_state.pose_model_path = selected
                    load_pose_model.clear()
                st.rerun()
        else:
            st.info("暂无上传模型。")


elif st.session_state.menu == "预警记录管理":
    with main_col:
        st.subheader("预警记录管理")
        logs_df = pd.DataFrame(st.session_state.violation_logs)
        if logs_df.empty:
            st.info("暂无预警记录。")
        else:
            st.dataframe(logs_df, use_container_width=True, hide_index=True)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("全部标记为已处理", use_container_width=True):
                    for item in st.session_state.violation_logs:
                        item["处理状态"] = "已处理"
                    st.rerun()
            with col_b:
                if st.button("导出预警记录", use_container_width=True):
                    csv_path, xlsx_path = export_dataframe(logs_df, "预警记录")
                    st.success(f"已导出：{csv_path.name} / {xlsx_path.name}")
            with col_c:
                if st.button("清空预警记录", use_container_width=True):
                    st.session_state.violation_logs = []
                    st.rerun()

            latest_images = [row.get("截图路径", "") for row in st.session_state.violation_logs if row.get("截图路径")]
            if latest_images:
                st.markdown("#### 最近预警截图")
                preview_cols = st.columns(min(3, len(latest_images)))
                for col, image_path in zip(preview_cols, latest_images[-3:]):
                    with col:
                        if Path(image_path).exists():
                            st.image(image_path, use_container_width=True)


elif st.session_state.menu == "结果展示":
    with main_col:
        st.subheader("训练与验证结果展示")
        metrics = latest_results_summary()
        if not metrics.empty:
            st.markdown("#### 训练末轮关键指标")
            st.dataframe(metrics, use_container_width=True, hide_index=True)

        tab_train, tab_val, tab_samples = st.tabs(["训练曲线", "验证评估", "检测样例"])
        with tab_train:
            show_report_image(TRAIN_REPORT_DIR / "results.png", "训练损失与指标曲线")
            curve_cols = st.columns(2)
            with curve_cols[0]:
                show_report_image(TRAIN_REPORT_DIR / "BoxPR_curve.png", "PR 曲线")
                show_report_image(TRAIN_REPORT_DIR / "BoxP_curve.png", "Precision 曲线")
            with curve_cols[1]:
                show_report_image(TRAIN_REPORT_DIR / "BoxF1_curve.png", "F1 曲线")
                show_report_image(TRAIN_REPORT_DIR / "BoxR_curve.png", "Recall 曲线")
        with tab_val:
            cm_cols = st.columns(2)
            with cm_cols[0]:
                show_report_image(VAL_REPORT_DIR / "confusion_matrix.png", "混淆矩阵")
            with cm_cols[1]:
                show_report_image(VAL_REPORT_DIR / "confusion_matrix_normalized.png", "归一化混淆矩阵")
            batch_cols = st.columns(2)
            with batch_cols[0]:
                show_report_image(VAL_REPORT_DIR / "val_batch0_labels.jpg", "验证标签样例")
            with batch_cols[1]:
                show_report_image(VAL_REPORT_DIR / "val_batch0_pred.jpg", "验证预测样例")
        with tab_samples:
            samples = image_paths(DETECT_SAMPLE_DIR)[:12]
            if not samples:
                st.info("暂无检测样例图。")
            else:
                cols = st.columns(3)
                for idx, path in enumerate(samples):
                    with cols[idx % 3]:
                        st.image(str(path), caption=path.name, use_container_width=True)


elif st.session_state.menu == "结果导出":
    with main_col:
        st.subheader("结果导出")
        results_df = st.session_state.detection_results
        logs_df = pd.DataFrame(st.session_state.violation_logs)

        st.markdown("#### 检测明细")
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        if not results_df.empty:
            csv_path, xlsx_path = export_dataframe(results_df, "安全行为检测结果")
            with open(csv_path, "rb") as file:
                st.download_button("下载检测结果 CSV", file, file_name=csv_path.name, use_container_width=True)
            if xlsx_path.exists() and xlsx_path != csv_path:
                with open(xlsx_path, "rb") as file:
                    st.download_button("下载检测结果 Excel", file, file_name=xlsx_path.name, use_container_width=True)

        st.markdown("#### 预警日志")
        st.dataframe(logs_df, use_container_width=True, hide_index=True)
        if not logs_df.empty:
            csv_path, xlsx_path = export_dataframe(logs_df, "预警日志")
            with open(csv_path, "rb") as file:
                st.download_button("下载预警日志 CSV", file, file_name=csv_path.name, use_container_width=True)
            if xlsx_path.exists() and xlsx_path != csv_path:
                with open(xlsx_path, "rb") as file:
                    st.download_button("下载预警日志 Excel", file, file_name=xlsx_path.name, use_container_width=True)
