# 종이 한 장 — 10월 키네틱 타이포 모션

10월의 고요한 아침을 찢는 30초 키네틱 타이포그래피 영상. [HyperFrames](https://github.com/heygen-com/hyperframes)로 HTML을 MP4로 렌더했다.

- 완성본: [`film/renders/october-paper.mp4`](film/renders/october-paper.mp4) (1920×1080, 30fps, 30.0초)
- 기획: [`film/STORYLINES.md`](film/STORYLINES.md) (3안) → [`film/SPEC.md`](film/SPEC.md) (B안 연출 명세)
- 검수 기록: [`film/REVIEW.md`](film/REVIEW.md)

## 구조

```
film/                  HyperFrames 프로젝트
  index.html           루트: 씬 4개를 겹쳐 쌓고 스코어를 붙인다
  compositions/        s1-hanji · s2-kraft · s3-news · s4-poster
  assets/fonts/        OFL 폰트 4종 (OFL.txt)
  assets/paper/        종이 텍스처 (tools/make_textures.py 생성)
  assets/audio/        score.wav (tools/make_score.py 생성)
tools/                 텍스처·스코어 생성 스크립트 (numpy, scipy, Pillow)
index.html             GitHub Pages 랜딩 페이지
```

## 다시 만들기

Node 22+, FFmpeg, Python 3.13이 필요하다.

```bash
cd film
npx hyperframes preview    # Studio 미리보기
npx hyperframes check      # 검사
npx hyperframes render -o renders/october-paper.mp4
```

텍스처·스코어 재생성:

```bash
python -m venv .venv && .venv/Scripts/pip install numpy scipy pillow
.venv/Scripts/python tools/make_textures.py film/assets/paper
.venv/Scripts/python tools/make_score.py film/assets/audio/score.wav <media-use sfx 폴더> <ffmpeg 경로>
```

마지막 줄 문구는 `film/compositions/s4-poster.html`의 변수 `closing`으로 바꾼다.

## 라이선스 메모

- 폰트: 나눔붓·나눔펜, Noto Serif KR, Pretendard — SIL OFL 1.1 (`film/assets/fonts/OFL.txt`).
- 라이저·임팩트 효과음: Pixabay Content License (HyperFrames media-use 번들).
- 음악·텍스처: 이 레포의 스크립트로 직접 생성.
