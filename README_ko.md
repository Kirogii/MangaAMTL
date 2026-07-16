<p align="center">
  <b>언어 (MTL)</b><br>
  🇺🇸 <a href="README.md">English</a> |
  🇨🇳 <a href="README_cn.md">简体中文</a> |
  🇰🇷 <a href="README_ko.md">한국어</a>
</p>
<div align="center">

<img src="https://github.com/Kirogii/MangaAMTL/blob/main/Images/English2.png?raw=true" alt="Preview1" width="45%" />
<img src="https://github.com/Kirogii/MangaAMTL/blob/main/Images/Raw.png?raw=true" alt="Preview2" width="45%" />

<br>

<b>V3.1.0 번역 샘플</b>
<i>(현재 조판 문제는 텍스트 크기만 존재)</i>
# MangaAMTL
### 참고: 첫 실행 시 여러 GB의 모델이 설치됩니다
</div>

> [!WARNING]
> 문제를 모르는 이상 생산 환경에서 실행하지 마세요 (그게 무엇인지 모르면 이 것을 포트포워딩하지 마세요)

> [!CAUTION]
> 이는 알파 테스트 단계이므로 버그가 예상됩니다. 문제가 있으면 [이슈](https://github.com/Kirogii/MangaAMTL/issues)를 남겨주세요. 콘솔 데이터를 포함해 주시면 더욱 도움이 됩니다 (http://localhost:8000/console)

<div align="center">
</div>

<div align="center">

# 로컬 서버 설정
파이썬 3.12가 필요합니다
</div>

- C++ 빌드 도구 다운로드: https://visualstudio.microsoft.com/visual-cpp-build-tools/ <sub>(C++ 빌드 도구를 반드시 선택하여 다운로드)</sub>
- 최신 릴리즈 다운로드: [최신 버전](https://github.com/Kirogii/MangaAMTL/releases/latest)
- 3: 내부로 이동
`cd MangaAMTL`
- 4: 의존성 설치
`pip install -r requirements.txt` 또는 CUDA를 사용하는 경우 `pip install -r cudarequirements.txt`
- 5: 개발 서버 실행
`python app.py`

`애플리케이션은 다음 주소에서 시작됩니다:
 http://localhost:8000`
</div>

<div align="center">
</div>

<div align="center">

# 크레딧
- https://huggingface.co/Qwen/Qwen3.5-0.8B
- https://huggingface.co/sharky172/manga-light-colorizer/
- https://huggingface.co/Kirogii/Yolo-Manga_Textbox-Region_Detect/
- https://huggingface.co/zai-org/GLM-OCR

# 리눅스 설정
### 이 방법은 curl 대신 wget을 사용합니다
- `apt install wget` 실행
- `wget https://raw.githubusercontent.com/Kirogii/MangaAMTL/refs/heads/main/Ubuntu.sh` 실행
- `chmod +x Ubuntu.sh` 실행
- `./Ubuntu.sh` 실행

### 위 설정을 완료했다면 'Manga' 명령어로 서버를 시작하세요. 업데이트가 있으면 'update' 입력 시 업데이트가 진행됩니다.
</div>

<div align="center">

# 기능
- 소스 지원 다국어 번역 (중국어, 한국어, 일본어, 러시아어, 인도네시아어, 영어)
- 모델 선택 (GGUF 전용) 쉽게 모델을 추가하거나 설치 가능
- 컬러라이징 지원
  
</div>

<div align="center">

# 문제점
- Llama-Cpp-Python 오류 (CUDA 버전): CUDA 다운로드 후 안내에 따라 재설치 필요 - https://developer.nvidia.com/cuda-downloads
- CUDA가 OCR/AI에 제대로 설정되지 않은 경우: `pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu126 `
