# 智慧工地安全行为识别与预警系统

施工现场安全监测原型。用 YOLO 检测未佩戴安全帽，用 YOLO Pose 识别危险俯身和疑似倒地，在 Streamlit 页面里完成实时预警、记录和导出。

## 功能

- 图片检测、文件夹批量检测、摄像头实时监测
- 安全帽合规 / 未佩戴判定
- 人体 17 关键点规则：危险俯身、疑似倒地
- 预警截图、处理状态、CSV / Excel 导出
- 检测模型与姿态模型上传切换

## 训练结果

在 HelmetHead 数据集（约 1.06 万张训练图，两类：`head` / `helmet`）上训练后，末轮指标：

| 指标 | 数值 |
| --- | --- |
| Precision | 0.917 |
| Recall | 0.830 |
| mAP@50 | 0.885 |
| mAP@50-95 | 0.604 |

曲线和混淆矩阵在 `assets/train/`。本仓库只带测试样例，完整训练集未放入，避免体积过大。

## 运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\run_app.ps1
```

浏览器打开 `http://127.0.0.1:8502`。

默认读取：

- `models/best.pt`：安全帽检测权重
- `models/yolov26n-pose.pt`：人体姿态权重
- `data/HelmetHead/test`：测试样例
- `data/samples`：页面预览样例

首次安装 `torch` / `ultralytics` 可能较慢。没有 GPU 时自动走 CPU，实时帧率会下降。

## 重新训练

把完整数据集按 `train/images`、`valid/images`、`test/images` 放到 `data/HelmetHead/`，确认 `safety_dataset.yaml` 后执行：

```powershell
python train.py
```

脚本会按是否有 CUDA 自动选择设备。

## 技术栈

Python、Ultralytics YOLO、OpenCV、Streamlit、Pandas

## License

MIT. Public attribution uses GitHub account [h1neolzr7f](https://github.com/h1neolzr7f) only.

Portfolio index: [dev-portfolio](https://github.com/h1neolzr7f/dev-portfolio)
