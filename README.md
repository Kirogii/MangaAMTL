<p align="center">
  <b>Languages  (MTL)</b><br>
  🇺🇸 <a href="README.md">English</a> |
  🇨🇳 <a href="README_cn.md">简体中文</a> |
  🇰🇷 <a href="README_ko.md">한국어</a>
</p>
<div align="center">

<img src="https://github.com/Kirogii/MangaAMTL/blob/main/Images/English2.png?raw=true" alt="Preview1" width="45%" />
<img src="https://github.com/Kirogii/MangaAMTL/blob/main/Images/Raw.png?raw=true" alt="Preview2" width="45%" />

<br>

<b>V3.1.0 Translation Showcase</b>
<i>(Text size is the only issue in typesetting currently)</i>
# MangaAMTL
### Note: On First launch this installs multiple gbs of models
</div>

> [!WARNING]
> Do not run this for production unless you know what you are doing (If you do not know what that is do not portforward this for other people)

> [!CAUTION]
> This is in alpha testing bugs are to be expected if you have a problem make a [Report](https://github.com/Kirogii/MangaAMTL/issues) including data from your console http://localhost:8000/console

<div align="center">
</div>

<div align="center">

# Localhost Setup
Python 3.12 Is Needed for this
</div>

- Download C++ Buildtools: https://visualstudio.microsoft.com/visual-cpp-build-tools/ <sub>(Make sure to select c++ buildtools)</sub>
- Download Latest Release: [Latest Version](https://github.com/Kirogii/MangaAMTL/releases/latest)
- 3: Cd inside
`cd MangaAMTL`
- 4: Install dependencies
`pip install -r requirements.txt` or for cuda `pip install -r cudarequirements.txt`
- 5: Run the development server
`python app.py`

` The application will start on:
 http://localhost:8000`
</div>

<div align="center">

</div>

<div align="center">


# Credits
- https://huggingface.co/Qwen/Qwen3.5-0.8B
- https://huggingface.co/sharky172/manga-light-colorizer/
- https://huggingface.co/Kirogii/Yolo-Manga_Textbox-Region_Detect/
- https://huggingface.co/zai-org/GLM-OCR

# Linux Setup
### This uses wget not curl
- Run `apt install wget`
- Run `wget https://raw.githubusercontent.com/Kirogii/MangaAMTL/refs/heads/main/Ubuntu.sh`
- Run `chmod +x Ubuntu.sh`
- Run `./Ubuntu.sh`

### After you've done the setup just run 'Manga' to start the server if theres an update it will prompt you to update by typing 'update'
</div>

<div align="center">

# Features
- Multi language translation from source support (Chinese,Korean,Japanese,Russian,Indonesian,English)
- Model selection (You can install/add new models without much effort GGUF ONLY)
- Colorizing Support
  
</div>

<div align="center">

# Problems
- Llama-Cpp-Python erroring (ON Cuda version): Install: https://developer.nvidia.com/cuda-downloads (Then rerun the install command from above)
- Cuda not being set on ocr/ai: `pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu126 `
