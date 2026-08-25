# 智慧工地安全行为识别与预警系统

使用 YOLO 检测未佩戴安全帽，使用 YOLO Pose 识别危险俯身与疑似倒地。检测入口包括单张图像、文件夹与摄像头，结果在 Streamlit 页面中展示、记录并导出。

## 功能

- 单张检测、文件夹批量检测、摄像头实时监测
- 安全帽佩戴判定（`head` / `helmet`）
- 基于 17 个关键点的危险俯身与疑似倒地判定
- 预警截图、处理状态管理，以及 CSV / Excel 导出
- 检测模型与姿态模型可切换

## 训练结果

HelmetHead 数据集约 1.06 万张训练图像，类别为 `head` / `helmet`。末轮指标：

| 指标 | 数值 |
| --- | --- |
| Precision | 0.917 |
| Recall | 0.830 |
| mAP@50 | 0.885 |
| mAP@50-95 | 0.604 |

训练曲线与混淆矩阵见 `assets/train/`。本仓库仅包含测试样例，完整训练集未纳入，以避免体积过大。

## 运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\run_app.ps1
```

浏览器访问 `http://127.0.0.1:8502`。

默认路径：

- `models/best.pt`：安全帽检测权重
- `models/yolov26n-pose.pt`：人体姿态权重
- `data/HelmetHead/test`：测试样例
- `data/samples`：页面预览样例

首次安装 `torch` / `ultralytics` 可能耗时较长。无 GPU 时自动使用 CPU，实时帧率会下降。

## 重新训练

将完整数据按 `train/images`、`valid/images`、`test/images` 放入 `data/HelmetHead/`，确认 `safety_dataset.yaml` 后执行：

```powershell
python train.py
```

脚本根据是否存在 CUDA 选择计算设备。

**技术栈** Python、Ultralytics YOLO、OpenCV、Streamlit、Pandas

许可：MIT。公开署名仅使用 GitHub 账号 [h1neolzr7f](https://github.com/h1neolzr7f)。

作品集：[dev-portfolio](https://github.com/h1neolzr7f/dev-portfolio)
