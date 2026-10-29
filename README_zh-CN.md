<div align="center">

<img src="assets/wechat-logo.png" height="40" alt="WeChat Vision">

# WeVisOne-VL

### 从目标到像素的统一视觉感知模型

<p>
  <a href="https://tencent.github.io/WeVisOne-VL"><img src="https://img.shields.io/badge/Project-Page-4285F4?logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2610.31329"><img src="https://img.shields.io/badge/Paper-arXiv-EA4335?logo=arxiv&logoColor=white" alt="Paper"></a>
  <a href="https://huggingface.co/tencent/WeVisOne-VL-2B"><img src="https://img.shields.io/badge/🤗%20Model-WeVisOne--VL--2B-FBBC04" alt="WeVisOne-VL-2B"></a>
  <a href="https://huggingface.co/tencent/WeVisOne-VL-4B"><img src="https://img.shields.io/badge/🤗%20Model-WeVisOne--VL--4B-FBBC04" alt="WeVisOne-VL-4B"></a>
  <a href="https://huggingface.co/spaces/hugging-apps/wevisone-vl-2b"><img src="https://img.shields.io/badge/🎮%20Demo-2B-34A853" alt="2B Demo"></a>
  <a href="https://huggingface.co/spaces/hugging-apps/wevisone-vl-4b"><img src="https://img.shields.io/badge/🎮%20Demo-4B-34A853" alt="4B Demo"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License"></a>
</p>

[English](README.md) | **简体中文** | [日本語](README_ja.md)

</div>

> **WeVisOne-VL**（**We**Chat **Vis**ion All-in-**One**）是一个视觉-语言基础模型。它通过结构化的 next-token prediction 和一个仅 9.6M 参数的轻量像素解码器，统一了**目标级**感知（目标检测、指代定位、计数、指点和 OCR）与**像素级**感知（指代分割和可提示抠图）。模型可以按用户指定的任意粒度进行定位，从点、框到二值掩码和连续的 Alpha Matte，并能可靠地拒绝目标不存在的查询。

<p align="center">
  <img src="assets/teaser.png" width="100%" alt="WeVisOne-VL 能力总览">
</p>

## ✨ 亮点

- 🎯 **All-in-One**：检测、指代、计数、指点、OCR、分割和抠图，一个模型全部搞定。
- 🪶 **轻量像素解码器**：仅 9.6M 参数即可输出掩码和 Alpha Matte，无需 SAM。
- 🚫 **可靠拒识**：目标不存在时返回 `None`，而不是产生幻觉。

## 📢 新闻

- **[2026-10]** 🎮 Demo 可以在线体验了：[2B](https://huggingface.co/spaces/hugging-apps/wevisone-vl-2b) / [4B](https://huggingface.co/spaces/hugging-apps/wevisone-vl-4b)。
- **[2026-10] 🔥🔥 我们还开源了 WeVisOne-VL 数据引擎中使用的抠图打标模型 [Mask2Alpha](https://github.com/Tencent/WeVisOne-VL/tree/mask2alpha)，可将二值掩码转换为高质量的 Alpha Matte！** 请查看 [`mask2alpha`](https://github.com/Tencent/WeVisOne-VL/tree/mask2alpha) 分支。
- **[2026-10]** 🎉 我们发布了技术报告、模型权重（2B 和 4B）、推理工具包、Cookbooks 和评测流程！

## 📝 TODO

- [x] 推理工具包与 Cookbooks
- [x] Gradio 网页 Demo
- [x] Benchmark 评测流程
- [x] 在 Hugging Face 上发布模型权重（2B 和 4B）
- [x] 开源抠图打标模型
- [ ] 微调代码

## 🏆 性能速览

**WeVisOne-VL-4B** 在各 benchmark 上与最强开源基线的对比。基线包括 9B 以内的 Qwen3-VL / Qwen3.5、Rex-Omni、LocateAnything，以及专用的分割 / 抠图模型。

| 任务 | Benchmark | 指标 | 最强基线 | **WeVisOne-VL-4B** |
| :--- | :--- | :---: | :---: | :---: |
| 📦 目标检测 | LVIS | F1 Mean ↑ | 50.7 | **55.3** |
| 📦 密集检测 | Dense200 | F1 Mean ↑ | 58.7 | **71.3** |
| 💬 指代检测 | RefCOCOg test | F1 Mean ↑ | 77.6 | **81.8** |
| 🚫 拒识 | WeRef-Bench | 拒识率 ↑ | 73.0 | **96.4** |
| 🔢 计数 | PixMo-Count test | Acc ↑ | 82.8 | **89.6** |
| 📍 指点 | PointBench | Acc ↑ | 69.9 | **71.9** |
| 🔤 OCR | TotalText | F1 Mean ↑ | 43.3 | **52.0** |
| 🎭 指代分割 | ReasonSeg val | gIoU ↑ | 65.0 | **71.2** |
| ✂️ 框提示抠图 | P3M-500-NP | MAD ↓ | 15.4 | **7.3** |
| ✂️ 文本提示抠图 | WeRefMat (Referring) | MAD ↓ | 22.4 | **9.8** |

#### 📊 完整 benchmark 结果（点击各任务展开）

检测、指代和 OCR 的 F1 为 IoU 阈值 0.5:0.95 上的平均值（"Mean"）。推理使用贪心解码，每张图最多 1,960 个视觉 token。

<details>
<summary><b>目标检测（F1 Mean）</b></summary>

| 方法 | COCO | LVIS | Dense200 | VisDrone |
| :--- | :---: | :---: | :---: | :---: |
| GroundingDINO-T | 56.6 | 38.8 | 33.1 | 38.5 |
| Qwen3-VL-8B | 47.8 | 42.9 | 13.8 | 29.5 |
| Qwen3.5-9B | 49.5 | 43.7 | 18.0 | 31.3 |
| Rex-Omni-3B | 52.9 | 46.9 | 58.3 | 35.8 |
| LocateAnything-3B | 54.7 | 50.7 | 58.7 | 39.9 |
| **WeVisOne-VL-2B** | 55.2 | 53.3 | 61.4 | 40.4 |
| **WeVisOne-VL-4B** | **57.1** | **55.3** | **71.3** | **44.7** |

</details>

<details>
<summary><b>指代目标检测（F1 Mean / 拒识率 %）</b></summary>

| 方法 | RefCOCOg test | RefCOCOg Rej | WeRef Mean | WeRef Rej | HumanRef | Ref-L4 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-VL-8B | 74.9 | 22.5 | 36.7 | 18.3 | 71.6 | 72.4 |
| Qwen3.5-9B | 74.6 | 15.2 | 50.7 | 70.7 | 77.8 | 72.0 |
| Rex-Omni-3B | 74.3 | 0.5 | 50.3 | 43.3 | 79.9 | 65.2 |
| LocateAnything-3B | 77.6 | 0.6 | 51.7 | 73.0 | 78.7 | 74.3 |
| **WeVisOne-VL-2B** | 78.2 | 50.2 | 73.2 | 92.9 | 77.4 | 75.1 |
| **WeVisOne-VL-4B** | **81.8** | **67.4** | **78.7** | **96.4** | **80.7** | **78.1** |

</details>

<details>
<summary><b>目标计数（精确匹配准确率 %）</b></summary>

| 方法 | CountQA | PixMo-Count test | CLOC test | KubriCount testB | MixCount test |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen3.5-9B | 41.6 | 80.4 | 22.4 | 19.1 | 7.0 |
| Rex-Omni-3B | 37.6 | 71.9 | 22.3 | 18.8 | 11.4 |
| LocateAnything-3B | 52.4 | 82.8 | 23.9 | 22.4 | 15.2 |
| **WeVisOne-VL-2B** | 47.9 | 88.1 | 26.9 | 32.4 | 32.2 |
| **WeVisOne-VL-4B** | **57.1** | **89.6** | **28.3** | **40.1** | **40.4** |

</details>

<details>
<summary><b>目标指点</b></summary>

| 方法 | PointBench | PixMo-Points | COCO | LVIS | Dense200 | VisDrone |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-VL-8B | 62.5 | 72.2 | 77.5 | 67.7 | 42.2 | 46.4 |
| Rex-Omni-3B | 69.9 | 72.9 | 80.5 | 70.8 | 82.5 | 58.9 |
| LocateAnything-3B | 56.9 | 68.6 | 83.9 | 76.6 | **87.6** | 60.4 |
| **WeVisOne-VL-2B** | 62.9 | 73.2 | **85.8** | 78.9 | 83.2 | 62.4 |
| **WeVisOne-VL-4B** | **71.9** | **74.7** | 85.4 | **79.5** | 87.0 | **64.4** |

</details>

<details>
<summary><b>OCR：文本定位与识别（F1 Mean）</b></summary>

| 方法 | HierText | ICDAR2015 | TotalText | SROIE |
| :--- | :---: | :---: | :---: | :---: |
| PP-OCRv5（专用模型） | 30.5 | 25.6 | 25.7 | 58.6 |
| Qwen3-VL-8B | 25.7 | 27.1 | 35.4 | 43.3 |
| Rex-Omni-3B | 28.0 | 28.1 | 40.6 | 44.8 |
| LocateAnything-3B | 29.4 | 24.1 | 43.3 | 40.3 |
| **WeVisOne-VL-2B** | 30.5 | 42.6 | 42.5 | 62.1 |
| **WeVisOne-VL-4B** | **39.0** | **47.8** | **52.0** | **68.1** |

</details>

<details>
<summary><b>指代分割（cIoU；ReasonSeg 为 gIoU）</b></summary>

| 方法 | RefCOCO testA | RefCOCO+ testA | RefCOCOg test | ReasonSeg val | Ref-ZOM cIoU | Ref-ZOM Rej |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| LISA-13B (ft) | 78.8 | 70.2 | 70.5 | 65.0 | - | - |
| GSVA-7B | 78.9 | 69.6 | 73.3 | 50.5 | 66.6 | 41.9 |
| UniPixel-7B | 83.0 | 80.1 | 77.1 | 60.5 | 62.9 | 0.3 |
| Youtu-VL-4B | 82.0 | 79.6 | 76.6 | - | 61.0 | 1.3 |
| **WeVisOne-VL-2B** | 82.9 | 79.0 | 74.9 | 60.9 | **73.4** | 62.4 |
| **WeVisOne-VL-4B** | **84.5** | **81.1** | **77.8** | **71.2** | 72.2 | **86.0** |

</details>

<details>
<summary><b>可提示抠图（MAD ×10³ ↓）</b></summary>

| 方法 | P3M-500-NP (Box) | AM-2K (Box) | PPM-100 (Box) | WeRefMat (Category) | WeRefMat (Referring) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Matte Anything | 24.4 | 13.2 | 10.5 | - | - |
| SAM2-Matting (SAM3) | 15.4 | 12.1 | 10.8 | 70.8 | 51.9 |
| Qwen3.5-9B + SAM2-Matting | - | - | - | 77.8 | 22.4 |
| **WeVisOne-VL-2B** | **6.6** | 5.7 | **5.3** | 17.7 | 15.8 |
| **WeVisOne-VL-4B** | 7.3 | **5.0** | 6.1 | **16.4** | **9.8** |

</details>

全部数据划分和指标请见技术报告；公开 benchmark 的复现方法请见 [evaluation/README.md](evaluation/README.md)。


## 🏗️ 工作原理

**1. 数据引擎。** 由知识图谱驱动发现概念，用多个互补的专家模型定位目标并由 VLM 校验，再把校验后的框扩展为掩码和 Alpha Matte。共标注了超过 1 亿张图像。

<p align="center">
  <img src="assets/method/wde_overview.jpg" width="100%" alt="数据引擎">
</p>

**2. 统一任务格式。** 所有任务都写成结构化文本：框和点用坐标 token `<0>` … `<1000>` 表示，目标不存在时输出 `None`，像素级任务在每个框后输出一个 `<OBJ>` token。

<p align="center">
  <img src="assets/method/task_formulation.jpg" width="100%" alt="统一任务格式">
</p>

**3. 模型架构。** 9.6M 参数的像素解码器结合每个 `<OBJ>` token 的隐状态、对应的预测框和 Qwen3-VL ViT 的多层特征，预测全分辨率的掩码或 Alpha Matte。

<p align="center">
  <img src="assets/method/pixel_decoder.jpg" width="100%" alt="像素解码器结构">
</p>

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/Tencent/WeVisOne-VL.git && cd WeVisOne-VL
conda create -n wevisone-vl python=3.10 -y && conda activate wevisone-vl
pip install -r requirements.txt && pip install -v -e .
```

### 推理

```python
from PIL import Image
from wevisone_vl import WeVisOneVL
from wevisone_vl.visualization import save_result

model = WeVisOneVL("tencent/WeVisOne-VL-4B", backend="transformers")  # 或 backend="vllm"
# model = WeVisOneVL("tencent/WeVisOne-VL-2B", backend="transformers")  # 使用 2B 时改用此行

# 目标级：开放词汇目标检测
image = Image.open("demo/examples/images/livingroom.jpg").convert("RGB")
result = model.predict(image, task="detection", query=["sofa", "pillow", "coffee table", "lamp"])
save_result("outputs/detection", image, result)

# 像素级：可提示抠图
image = Image.open("demo/examples/images/seabreeze.jpg").convert("RGB")
result = model.predict(image, task="matting", query="person")
save_result("outputs/matting", image, result)
```

<table>
  <tr>
    <td align="center"><img src="assets/demo/seabreeze.jpg" width="100%"><br><sub>输入 · <code>query="person"</code></sub></td>
    <td align="center"><img src="assets/demo/seabreeze_alpha.jpg" width="100%"><br><sub>预测的 Alpha Matte</sub></td>
    <td align="center"><img src="assets/demo/seabreeze_cutout.jpg" width="100%"><br><sub>抠图结果</sub></td>
  </tr>
</table>

### 支持的任务

| | `task` | `query` | 输出 |
| :---: | :--- | :--- | :--- |
| 📦 | `detection` | 类别列表 | 框 |
| 💬 | `referring` | 指代描述 | 框或 `None` |
| 📍 | `point` | 类别 / 描述 | 点 |
| 🔢 | `count_box` / `count_point` | 类别 / 描述 | 框 / 点 + 数量 |
| 🔤 | `ocr` / `ocr_grounding` | – / 待查找的文本 | 文本 + 框 |
| 🎭 | `segmentation` | 类别 / 描述 | 框 + 二值掩码 |
| ✂️ | `matting` | 类别 / 描述 | 框 + Alpha Matte |

> [!TIP]
> 还可以设置 `device`（`"auto"`、`"cuda:0"`、`"npu:0"`、`"cpu"`）、`dtype`、`min_pixels` / `max_pixels`，以及常用的解码参数（`max_new_tokens`、`temperature` 等）。[Cookbooks](#-cookbooks) 中有每个任务的详细用法。

## 📓 Cookbooks

| 教程 | 内容 |
| :--- | :--- |
| 📦 [目标级流程](tutorials/01_object_level_pipeline.ipynb) | 检测、指代、计数、指点 |
| 🔤 [OCR 流程](tutorials/02_ocr_pipeline.ipynb) | OCR、文本定位 |
| 🎭 [像素级流程](tutorials/03_segmentation_matting_pipeline.ipynb) | 分割、抠图 |

## 🎮 Gradio Demo

在线体验：[2B](https://huggingface.co/spaces/hugging-apps/wevisone-vl-2b) / [4B](https://huggingface.co/spaces/hugging-apps/wevisone-vl-4b)

在本地启动 Demo：

```bash
CUDA_VISIBLE_DEVICES=0 python demo/app.py --model tencent/WeVisOne-VL-4B
# 或使用 2B：
# CUDA_VISIBLE_DEVICES=0 python demo/app.py --model tencent/WeVisOne-VL-2B
```

<details>
<summary>更多启动选项</summary>

```bash
# 使用 vLLM 后端、自定义端口，并生成公开分享链接
CUDA_VISIBLE_DEVICES=0 python demo/app.py \
    --model tencent/WeVisOne-VL-4B --backend vllm \
    --server-name 0.0.0.0 --server-port 8080 --share
```

</details>

## 📊 评测

请参考 [evaluation/README.md](evaluation/README.md)。

## 🖼️ 代表性可视化

与强基线的并排对比，点击任务名可展开或折叠。

<details open>
<summary><b>指代检测</b>：根据自然语言描述定位对应的目标</summary>
<br>
<img src="assets/case_study/referring.jpg" width="100%" alt="指代检测对比">
</details>

<details open>
<summary><b>框提示抠图</b>：根据给定的框预测框内目标的 Alpha Matte</summary>
<br>
<img src="assets/case_study/matting_box.jpg" width="100%" alt="框提示抠图对比">
</details>

<details open>
<summary><b>文本提示抠图</b>：根据类别名或指代描述预测对应目标的 Alpha Matte</summary>
<br>
<img src="assets/case_study/matting_text.jpg" width="100%" alt="文本提示抠图对比">
</details>

<details>
<summary><b>目标检测</b>：用框定位给定类别的所有实例</summary>
<br>
<img src="assets/case_study/detection.jpg" width="100%" alt="目标检测对比">
</details>

<details>
<summary><b>目标计数</b>：统计符合类别或描述的实例数量</summary>
<br>
<img src="assets/case_study/counting.jpg" width="100%" alt="目标计数对比">
</details>

<details>
<summary><b>目标指点</b>：用点标出每个被查询的目标</summary>
<br>
<img src="assets/case_study/pointing.jpg" width="100%" alt="目标指点对比">
</details>

<details>
<summary><b>OCR</b>：定位并识别图中的文字</summary>
<br>
<img src="assets/case_study/ocr.jpg" width="100%" alt="OCR 对比">
</details>

<details>
<summary><b>指代分割</b>：根据自然语言描述分割对应的目标</summary>
<br>
<img src="assets/case_study/refseg.jpg" width="100%" alt="指代分割对比">
</details>

## 📜 许可证

本仓库中由腾讯编写的代码基于 [Apache License 2.0](LICENSE) 发布。第三方组件保留其原有许可证，请查阅相应源文件。

## 📖 引用

如果 WeVisOne-VL 对你的工作有帮助，欢迎点个 ⭐ 并引用：

```bibtex
@article{liang2026wevisone,
  title={WeVisOne-VL: Unified Visual Perception from Objects to Pixels},
  author={Liang, Tianming and Pan, Junyi and Li, Yuxi and Fu, Canmiao and Huang, Zhipeng and Sun, Chong and Li, Chen and Lyu, Jing},
  journal={arXiv preprint arXiv:2610.31329},
  year={2026}
}
```
