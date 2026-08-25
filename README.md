# 智慧工地安全行为识别与预警系统

用 YOLO 看有没有戴安全帽，再用 Pose 看人是不是弯太狠、像不像摔倒。页面是 Streamlit，图片、文件夹、摄像头都能试。

## 能做什么

- 单张图、整个文件夹、摄像头
- 戴没戴帽
- 17 个关键点判断俯身和疑似倒地
- 预警截图，能改处理状态，能导出 CSV / Excel
- 检测模型和姿态模型可以换

## 训练结果

HelmetHead 大约 1.06 万张训练图，两类：`head` / `helmet`。最后一轮：

| 指标 | 数值 |
| --- | --- |
| Precision | 0.917 |
| Recall | 0.830 |
| mAP@50 | 0.885 |
| mAP@50-95 | 0.604 |

曲线和混淆矩阵在 `assets/train/`。仓库只带测试样例，整套训练集太大，没放进来。

## 运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\run_app.ps1
```

浏览器开 `http://127.0.0.1:8502`。

默认用这些路径：

- `models/best.pt`：安全帽
- `models/yolov26n-pose.pt`：姿态
- `data/HelmetHead/test`：测试图
- `data/samples`：页面预览

第一次装 `torch` / `ultralytics` 会比较慢。没 GPU 就走 CPU，实时帧率会掉。

## 重新训练

把完整数据按 `train/images`、`valid/images`、`test/images` 放到 `data/HelmetHead/`，核对 `safety_dataset.yaml`，然后：

```powershell
python train.py
```

有 CUDA 就用显卡，没有就用 CPU。

Python、Ultralytics YOLO、OpenCV、Streamlit、Pandas

MIT。公开页只写 GitHub 账号 [h1neolzr7f](https://github.com/h1neolzr7f)。

总览：[dev-portfolio](https://github.com/h1neolzr7f/dev-portfolio)
