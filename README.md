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
