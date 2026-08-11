# 새 컴퓨터 이전 및 복구

이 저장소와 `baseline-v1` GitHub Release만 있으면 개발·테스트·공개 데모와
결과보고서 작업을 다른 컴퓨터에서 이어갈 수 있다. 실제 환자 데이터와 비밀값은
저장소에 포함하지 않는다.

## 1. 저장소 복구

Windows PowerShell과 Python 3.12를 기준으로 한다.

```powershell
git clone https://github.com/yuraira/ecg-guard.git
cd ecg-guard
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

CPU 전용 환경에서는 PyTorch 공식 설치 안내에 맞는 CPU wheel을 선택할 수 있다.
Docker를 사용하는 경우 저장소 루트에서 다음 명령으로 검증한다.

```powershell
docker build -t ecg-guard:local .
docker run --rm -p 127.0.0.1:8501:8501 ecg-guard:local
```

## 2. 동결 체크포인트 복구

`best_model.pt`는 Git 저장소가 아니라 공개
[`baseline-v1` Release](https://github.com/yuraira/ecg-guard/releases/tag/baseline-v1)에
보관한다. 앱의 검증 다운로드 명령을 사용하면 파일 크기와 SHA-256을 함께 확인한다.

```powershell
.\.venv\Scripts\ecg-guard-fetch-checkpoint.exe `
  --output .\checkpoints\best_model.pt
```

기대 SHA-256은
`44a8ecc96f1ac084db2ef6921bf8e438c1130da6be140d7fc3ac7fe3ecfa2ead`이다.

## 3. 결과보고서 복구

최종 DOCX와 PDF는 `deliverables/`에 있다. 접수번호와 YouTube 시연 영상 URL은
제출 전에 문서에서 직접 입력한다. 보고서를 다시 생성해야 하면 다음 파일들이 함께
버전 관리된다.

- `outputs/result-report-work/build_result_report.py`
- `outputs/result-report-work/official-template.docx`
- `outputs/baseline-evaluation/evaluation_curves.png`
- `requirements-report.txt`

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-report.txt
.\.venv\Scripts\python.exe .\outputs\result-report-work\build_result_report.py
```

생성 스크립트는 Windows의 맑은 고딕 글꼴을 사용한다. 다른 운영체제에서는 글꼴
경로를 수정하거나 최종 DOCX를 직접 편집한다.

## 4. 의도적으로 GitHub에 올리지 않는 항목

- `.env`, `.streamlit/secrets.toml` 등 인증정보와 비밀값
- `.venv/`, 캐시, 빌드 디렉터리와 로컬 Docker 이미지
- `data/`의 PTB-XL 원본·가공 데이터
- 로컬 학습 체크포인트와 임시 업로드 파일
- 재생성 가능한 일반 `outputs/` 및 로컬 SBOM

PTB-XL은 원 배포처의 라이선스와 이용 조건에 따라 새 컴퓨터에서 다시 내려받는다.
공개 서비스는 `https://ecg-guard.onrender.com`, 소스와 CI는 이 GitHub 저장소,
동결 가중치는 `baseline-v1` Release를 기준으로 한다.
