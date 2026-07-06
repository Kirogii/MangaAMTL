#      **           MangaAMTL is an open-source manga translation tool mainly for japanese manga**
Take note this is mostly a project for myself so if theres any bugs it will be fixed in the next version so im not fixing a bunch of problems (Although im testing the installation/Usage of the program fully so there shouldnt be too many problems for you

<p align="center">
  <img src="https://github.com/Kirogii/MangaAMTL/blob/main/Images/English.png?raw=true" alt="Preview1" width="45%" />
  <img src="https://github.com/Kirogii/MangaAMTL/blob/main/Images/Raw.png?raw=true" alt="Preview2" width="45%" />
</p>

> [!WARNING]  
> Do not run this for production unless you know what you are doing (If you do not know what that is do not portforward this)

> [!CAUTION]
> This is in alpha testing bugs are to be expected
# Tree Guide
- [Installation+Setup (For Localhost)](#LocalhostSetup)
- [Huggingface Setup](#HuggingfaceSetup)
- [Credits](#Credits)
- [Features](#Features)
- [Problems+Solutions](#Problems)
_________________________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________________________

#  Localhost Setup
### Python 3.12 Is Needed for this
### For V3+ You also need rust: https://rustup.rs
- 1: Download C++ Buildtools: https://visualstudio.microsoft.com/visual-cpp-build-tools/<sub> Make sure to select c++ buildtools</sub>
- 2: Download Latest Release: [Latest Version](https://github.com/Kirogii/MangaAMTL/releases/latest)
- 3: Cd inside
`cd MangaAMTL`
- 4: Install dependencies
`pip install -r requirements.txt` or for cuda `pip install -r cudarequirements.txt`
- 5: Run the development server
`python app.py`

` The application will start on:
 http://localhost:7860`
_________________________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________________________

# Huggingface Setup (Dead for now)
- Clone the space https://huggingface.co/spaces/Kirogii/Manga_AMTL


_________________________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________________________


# Credits
- https://huggingface.co/Qwen/Qwen3.5-0.8B
- https://huggingface.co/sharky172/manga-light-colorizer/
- https://huggingface.co/Kirogii/Yolo-Manga_Textbox-Region_Detect/
- https://huggingface.co/zai-org/GLM-OCR
_________________________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________________________

# Features
- Multi language translation from source support (Chinese,Korean,Japanese,Russian,Indonesian,English)
- Model selection (You can install/add new models without much effort GGUF ONLY)
- Colorizing Support
- If your looking for logs they are all inside http://localhost:7860/console
  
_________________________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________________________

# Problems
- Llama-Cpp-Python erroring (ON Cuda version): Install: https://developer.nvidia.com/cuda-downloads (Then rerun the install command from above)
- Cuda not being set on ocr/ai: `pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu126 `
