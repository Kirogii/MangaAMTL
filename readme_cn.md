<p align="center">
  <b>Languages</b><br>
  🇺🇸 <a href="README.md">English</a> |
  🇨🇳 <a href="README_cn.md">简体中文</a> |
  🇰🇷 <a href="README_ko.md">한국어</a>
</p>
<div align="center">

<img src="https://github.com/Kirogii/MangaAMTL/blob/main/Images/English2.png?raw=true" alt="Preview1" width="45%" />
<img src="https://github.com/Kirogii/MangaAMTL/blob/main/Images/Raw.png?raw=true" alt="Preview2" width="45%" />

<br>

<b>V3.1.0 翻译展示</b>
<i>（目前排版的唯一问题是文字尺寸）</i>
# MangaAMTL
### 注意：首次启动会安装多个 GB 的模型
</div>

> [!WARNING]
> 不要在生产环境中运行此程序，除非你清楚自己在做什么（如果你不了解其用途，请勿转发给他人或开启端口转发）。

> [!CAUTION]
> 这仍处于 Alpha 测试阶段，预期会出现 BUG。如果遇到问题，请在 [问题报告](https://github.com/Kirogii/MangaAMTL/issues) 中提供相关信息，包括来自控制台的日志：http://localhost:8000/console

<div align="center">
</div>

<div align="center">

# 本地环境设置（Localhost Setup）
需要 Python 3.12。
</div>

- 下载 C++ 构建工具：  
  https://visualstudio.microsoft.com/visual-cpp-build-tools/  
  <sub>（请确保勾选了 C++ 构建工具）</sub>
- 下载最新版本：[最新发布版](https://github.com/Kirogii/MangaAMTL/releases/latest)
- 3：进入项目目录  
  `cd MangaAMTL`
- 4：安装依赖项  
  `pip install -r requirements.txt`  
  或（支持 CUDA）  
  `pip install -r cudarequirements.txt`
- 5：运行开发服务器  
  `python app.py`

  应用程序将在以下地址启动：  
  `http://localhost:8000`

<div align="center">
</div>

<div align="center">

# 致谢与参考
- https://huggingface.co/Qwen/Qwen3.5-0.8B  
- https://huggingface.co/sharky172/manga-light-colorizer/  
- https://huggingface.co/Kirogii/Yolo-Manga_Textbox-Region_Detect/  
- https://huggingface.co/zai-org/GLM-OCR  

# Linux 系统设置
### 本指南使用 wget，请勿使用 curl
- 运行：`apt install wget`
- 运行：`wget https://raw.githubusercontent.com/Kirogii/MangaAMTL/refs/heads/main/Ubuntu.sh`
- 赋予执行权限：`chmod +x Ubuntu.sh`
- 执行脚本：`./Ubuntu.sh`

### 设置完成后，直接运行 `Manga` 即可启动服务器。若有更新，系统会提示你输入 `update` 进行更新。
</div>

<div align="center">

# 功能特性
- 多语言翻译（支持原文语言）：中文、韩语、日语、俄语、印尼语、英语  
- 模型选择（可轻松安装或添加新模型，目前仅支持 GGUF 格式）  
- 支持图像上色功能  

</div>

<div align="center">

# 常见问题
- Llama-Cpp-Python 报错（CUDA 版本）：  
  请安装 CUDA：https://developer.nvidia.com/cuda-downloads  
  然后重新运行上述安装命令。
- 未正确设置 OCR/AI 的 CUDA 环境：  
  运行命令：  
  `pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu126 `
