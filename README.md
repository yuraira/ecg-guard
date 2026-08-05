# ECG Guard

신호 품질과 예측 불확실성을 함께 제공하는 심전도 인공지능 평가 플랫폼입니다.

> 연구 및 교육용 프로젝트이며 실제 의료 진단을 목적으로 하지 않습니다.

[모델 카드](docs/model_card.md)에서 허용된 사용 범위, 동결된 시험 성능과
알려진 한계를 확인할 수 있습니다.

## 현재 구현 범위

- PTB-XL 1.0.3 메타데이터 및 SCP 코드 파싱
- NORM, MI, STTC, CD, HYP 진단 상위군 다중 라벨 생성
- 공식 `strat_fold` 기반 학습, 검증, 시험 분할
- 환자 중복에 의한 데이터 누수 검사
- 100Hz 12유도 파형 로딩 및 형태 검사
- 12유도 예제 파형 그림 생성
- 학습 폴드 전용 정규화 및 지연 로딩 PyTorch Dataset
- 다중 라벨 residual 1D CNN 기준 모델과 재현 가능한 학습 CLI
- 학습 분포 기준의 설명 가능한 파형 기술 품질 지표
- 예측 경계 uncertainty와 validation-locked 선택적 판정 분석
- 체크포인트 무결성 검증과 검토 사유를 포함한 단일 기록 추론 CLI
- 안전한 WFDB 업로드와 결과 시각화를 제공하는 Streamlit 웹 데모

## 모델링 정책

- 공식 분할을 그대로 사용합니다: `strat_fold` 1~8은 학습, 9는 검증, 10은 시험입니다.
- 다섯 진단 상위군이 하나도 없는 411건은 원본 메타데이터에는 보존하되,
  현재 다섯 라벨 분류 모델의 학습·평가 대상에서는 제외합니다.
- NORM과 다른 진단 상위군이 함께 표시된 기록도 원자료의 다중 라벨을 수정하지 않습니다.
- 정규화 평균과 표준편차는 학습 코호트에서만 계산하고 검증 데이터에 재사용합니다.
- 시험 폴드 파형과 라벨은 정규화, 학습, 검증, 모델 선택에 사용하지 않습니다.

## 환경 구성

Python 3.12 환경에서 검증했습니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

`requirements.txt`는 현재 개발 장비의 NVIDIA GPU에 맞춰 공식 PyTorch CUDA 13.0
패키지 인덱스를 사용합니다. 다른 운영체제나 CPU 전용 환경에서는
[PyTorch 설치 안내](https://pytorch.org/get-started/locally/)에 맞는 패키지를 선택하세요.

## 데이터 준비

[PhysioNet PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)에서
다음 파일을 `data/raw/ptb-xl` 아래에 준비합니다.

```text
data/raw/ptb-xl
├─ records100
├─ ptbxl_database.csv
├─ scp_statements.csv
└─ LICENSE.txt
```

PTB-XL 파일은 CC BY 4.0 조건을 따르며 저장소에는 포함하지 않습니다.

## 데이터 파이프라인 실행

```powershell
.\.venv\Scripts\python.exe -m ecg_guard.data.prepare_ptbxl `
  --data-dir .\data\raw\ptb-xl `
  --limit 100 `
  --plot-out .\outputs\ptbxl_sample.png
```

실행 시 전체 메타데이터와 클래스 수, 공식 분할별 건수, 환자 중복 여부,
로드한 파형 및 라벨 배열 크기를 출력합니다.

## 기준 모델 학습

```powershell
.\.venv\Scripts\python.exe -m ecg_guard.training.train_baseline `
  --data-dir .\data\raw\ptb-xl `
  --output-dir .\outputs\baseline `
  --epochs 15 `
  --batch-size 128
```

검증 macro AUROC가 가장 높은 체크포인트와 설정, 학습 폴드 정규화 통계,
epoch별 검증 지표가 출력 폴더에 저장됩니다. 모델 선택이 끝날 때까지 시험 폴드는
사용하지 않습니다.

전체 기준 실험의 설정, 환경, 검증 결과와 한계는
[`docs/experiments/baseline_v1.md`](docs/experiments/baseline_v1.md)에 기록했습니다.

시험 폴드를 사용하기 전에 확률 보정, 임계값, patient-cluster bootstrap과
하위집단 정의를 [`docs/evaluation_protocol_v1.md`](docs/evaluation_protocol_v1.md)에
고정합니다. 평가는 다음처럼 명시적 허용 플래그가 있어야 실행됩니다.

```powershell
.\.venv\Scripts\python.exe -m ecg_guard.evaluation.evaluate_baseline `
  --checkpoint .\outputs\baseline\best_model.pt `
  --data-dir .\data\raw\ptb-xl `
  --output-dir .\outputs\baseline-evaluation `
  --allow-test-evaluation
```

잠긴 기준 모델의 시험 결과는 macro AUROC 0.91627
(95% patient-cluster bootstrap 구간 0.90871~0.92378), macro average
precision 0.81057(0.79269~0.82625)이었습니다. 이 값은 PTB-XL 내부
벤치마크이며 임상적 유효성이나 외부 일반화를 의미하지 않습니다.

동결된 모델의 사후 오류 분석은 다음처럼 실행합니다. 이 결과는 모델 재튜닝이
아닌 설명과 다음 연구 가설 수립에만 사용합니다.

```powershell
.\.venv\Scripts\python.exe -m ecg_guard.analysis.error_analysis `
  --predictions .\outputs\baseline-evaluation\test_predictions.csv `
  --data-dir .\data\raw\ptb-xl `
  --output-dir .\outputs\error-analysis
```

파형 기반 기술 품질 점수는 학습 분포의 설명 가능한 SQI 극단값을 표시합니다.
임상적 품질 정답이나 자동 입력 거부 기준이 아니며 상세 정의는
[`docs/quality_protocol_v1.md`](docs/quality_protocol_v1.md)에 있습니다.

```powershell
.\.venv\Scripts\python.exe -m ecg_guard.quality.analyze_quality `
  --data-dir .\data\raw\ptb-xl `
  --output-dir .\outputs\quality-analysis
```

예측 경계에 가까운 기록의 판정을 보류하는 선택적 판정 분석은 검증 fold에서
90%, 80%, 70% coverage cutoff를 고정합니다. 단일 모델의 proxy이므로
epistemic uncertainty로 해석하지 않습니다. 자세한 규칙은
[`docs/selective_prediction_protocol_v1.md`](docs/selective_prediction_protocol_v1.md)에
기록했습니다.

```powershell
.\.venv\Scripts\python.exe -m ecg_guard.uncertainty.analyze_selective `
  --evaluation-dir .\outputs\baseline-evaluation `
  --quality-dir .\outputs\quality-analysis `
  --output-dir .\outputs\selective-analysis
```

시험에서 uncertainty의 any-label 오류 탐지 AUROC는 0.7573이었습니다. 검증
80% 목표 cutoff의 시험 실제 coverage는 77.4%였고 Hamming 오류율은 전체
0.1504에서 0.1270으로 감소했습니다. 이는 선택된 하위집단의 trade-off이며
모델 자체의 성능 향상이나 임상적 안전성을 뜻하지 않습니다.

## 단일 기록 추론

동결된 baseline v1 체크포인트와
[`baseline_v1_inference.json`](src/ecg_guard/resources/baseline_v1_inference.json)의
해시가 일치할 때만 추론합니다. 입력은 100 Hz, 10초, mV 단위의 표준 12유도
WFDB 레코드여야 하며 `.hea` 또는 `.dat` 확장자는 생략할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m ecg_guard.inference.predict_record `
  --record .\data\raw\ptb-xl\records100\00000\00001_lr `
  --checkpoint .\outputs\baseline\best_model.pt `
  --output .\outputs\inference\00001.json
```

결과 JSON은 클래스별 보정 확률·잠긴 임계값뿐 아니라 기술 품질 상태,
결정 경계 불확실성, `auto_result`, `review_uncertain`,
`review_technical`, `review_both` 중 하나의 검토 라우팅을 함께 제공합니다.
기술 품질은 자동 입력 폐기 기준이 아니며, uncertainty cutoff 역시 임상적으로
승인된 운영점이 아닙니다. 체크포인트는 저장소에 직접 포함하지 않으며 위 학습
명령으로 재현하거나
[`baseline-v1` GitHub Release](https://github.com/yuraira/ecg-guard/releases/tag/baseline-v1)에서
받을 수 있습니다. 공개 파일의 SHA-256은
`44a8ecc96f1ac084db2ef6921bf8e438c1130da6be140d7fc3ac7fe3ecfa2ead`입니다.

## 웹 데모

웹 화면에서는 `.hea` 파일 1개와 해당 헤더가 참조하는 `.dat` 파일을 함께
업로드합니다. 100Hz, 10초, 표준 12유도 순서와 파일 참조를 분석 전에
검사하며 업로드 파일은 임시 폴더에서만 처리합니다. 실제 환자 파일은 허용하지
않으며 공개되었거나 적절하게 비식별 처리된 연구 데이터만 사용할 수 있습니다.
처리 범위와 삭제 방식은
[`DATA_HANDLING_POLICY.md`](DATA_HANDLING_POLICY.md)에 기록했습니다.

```powershell
$env:ECG_GUARD_CHECKPOINT="outputs/baseline/best_model.pt"
$env:ECG_GUARD_DEMO_RECORD="data/raw/ptb-xl/records100/00000/00001_lr"
.\.venv\Scripts\ecg-guard-web.exe
```

`ECG_GUARD_DEMO_RECORD`는 로컬 시연용 선택 설정입니다. 공개 호스트에서는
실제 환자 데이터가 아닌 합성 UI 샘플로도 전체 흐름을 확인할 수 있습니다.
합성 결과는 모델 성능 사례나 임상적으로 유효한 ECG 예시가 아닙니다. 사용법과
공개 배포 확인사항은 [`docs/web_demo.md`](docs/web_demo.md)에 기록했습니다.
현재 공개 HTTPS 서비스는 Render 계정 연결 전 단계이며 로컬 데모만 실행 중입니다.

동결 체크포인트의 공개 패키지를 만들고 GitHub Release에서 검증하는 절차는
[`docs/model_release.md`](docs/model_release.md)에 기록했습니다. `baseline-v1`
Release는 공개 상태이며 인증 없는 다운로드와 체크섬 검증을 완료했습니다.

CPU 전용 Docker 이미지와 로컬 전용 Compose 실행, 컨테이너 SBOM 추출 및
인터넷 공개 전 조건은 [`docs/deployment.md`](docs/deployment.md)에
기록했습니다. 기본 포트는 안전하게 `127.0.0.1`에만 바인딩합니다.
공개 호스팅용 Infrastructure-as-Code는 [`render.yaml`](render.yaml)에 있으며,
실제 URL이 발급되기 전까지 데이터 처리 정책에는 `배포 예정 환경`으로 구분합니다.

파이프라인만 빠르게 검증할 때는 다음처럼 제한된 표본으로 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m ecg_guard.training.train_baseline `
  --data-dir .\data\raw\ptb-xl `
  --output-dir .\outputs\baseline-smoke `
  --epochs 1 `
  --batch-size 64 `
  --num-workers 0 `
  --train-limit 512 `
  --validation-limit 256
```

## 테스트

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 라이선스와 기여

소스 코드는 Apache-2.0으로 공개합니다. PTB-XL은 별도의 CC BY 4.0 조건을
따르며 자세한 출처는 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)에
기록했습니다. 기여 절차는 [`CONTRIBUTING.md`](CONTRIBUTING.md)를 참고하세요.
