# ECG Guard 평가 프로토콜 v1

고정일: 2026-07-27
대상 모델: `outputs/baseline/best_model.pt`, epoch 5

이 문서는 시험 폴드 결과를 확인하기 전에 고정한 평가 규칙이다. 시험 결과를
확인한 뒤 임계값, 보정 방법, 하위집단 정의 또는 주 평가 지표를 바꾸지 않는다.

## 데이터 사용

- PTB-XL 1.0.3 공식 권고에 따라 fold 1~8은 학습, fold 9는 검증,
  fold 10은 시험으로 사용한다.
- 다섯 진단 상위군 중 하나도 없는 기록은 현재 분류 과제에서 제외한다.
- 모델과 정규화 통계는 다시 학습하지 않는다.
- 확률 보정과 임계값은 검증 fold 9에서만 결정한다.
- 시험 fold 10은 잠긴 평가 코드를 한 번 실행하는 데만 사용한다.

## 확률 보정과 임계값

- 검증의 모든 클래스·기록에 대해 binary cross-entropy를 최소화하는 하나의
  양수 scalar temperature를 결정한다.
- 클래스별 임계값은 검증 ROC에서 Youden's J
  (`sensitivity + specificity - 1`)를 최대화한다.
- 최대값이 같은 임계값이 여러 개면 민감도를 우선해 가장 낮은 임계값을 쓴다.
- 시험에서는 temperature와 임계값을 변경하지 않는다.

## 주 평가 결과

- 클래스별 및 macro AUROC
- 클래스별 및 macro average precision
- 잠긴 임계값에서 민감도, 특이도, precision, F1
- binary negative log-likelihood, Brier score, 15-bin ECE
- 보정 전후 시험 calibration 지표

## 불확실성

- 동일 환자의 여러 ECG가 독립이라는 잘못된 가정을 피하기 위해 환자를
  cluster로 하여 복원추출한다.
- 1,000회 patient-cluster percentile bootstrap, seed 42를 사용한다.
- 95% 구간은 bootstrap 분포의 2.5와 97.5 percentile이다.
- AUROC, average precision, 민감도, 특이도, F1, Brier score의 macro 및
  클래스별 구간을 계산한다.
- 구간은 고정된 모델, temperature, 임계값에 조건부이며 학습과 보정 파라미터
  추정 자체의 변동성은 포함하지 않는다.

## 하위집단

- 성별은 데이터의 원시 코드 0과 1로 보고한다. 코드 의미를 근거 없이
  남성·여성으로 재명명하지 않는다.
- 연령은 `<40`, `40~59`, `60+`로 나눈다.
- PTB-XL에서 89세 초과를 비식별화한 `age=300`은 실제 연령으로 취급하지 않고
  연령 하위집단 비교에서 제외하며 제외 건수를 보고한다.
- 신호 품질은 `baseline_drift`, `static_noise`, `burst_noise`,
  `electrodes_problems` 중 하나 이상의 주석 존재 여부로 나눈다.
- artifact 주석 부재를 깨끗한 신호의 확정적 근거로 해석하지 않는다.

## 해석 제한

- PTB-XL 내부 시험 결과이며 외부 일반화나 임상적 유효성을 입증하지 않는다.
- 검증 fold는 모델 선택과 사후 보정에 함께 사용되어 보정 파라미터가 해당
  fold에 과적합될 가능성이 있다. 최종 수치는 독립 시험 fold에서 보고한다.
- 하위집단 비교는 기술적 탐색 분석이며 인과적 공정성 결론으로 해석하지 않는다.
- 단일 학습 seed 모델이므로 학습 변동성은 bootstrap 구간에 포함되지 않는다.

## 근거

- PTB-XL 1.0.3: https://physionet.org/content/ptb-xl/1.0.3/
- PTB-XL 공식 benchmark:
  https://github.com/helme/ecg_ptbxl_benchmarking
- Temperature scaling:
  https://proceedings.mlr.press/v70/guo17a.html
