# ECG Guard 선택적 판정 프로토콜 v1

고정일: 2026-07-27
대상: 동결된 baseline v1의 보정 확률과 검증 고정 임계값

## 목적

예측이 클래스 결정 경계에 가까운 기록을 사람이 재검토하도록 라우팅했을 때
coverage와 오류의 trade-off를 측정한다. 이 기능은 의료적 안전이나 임상 효용이
검증된 자동 보류 시스템이 아니다.

## 불확실성 proxy

- 각 클래스에서 보정 확률과 잠긴 임계값을 log-odds로 변환한다.
- 절대 log-odds 거리를 `exp(-distance)`로 변환해 경계에서 1, 멀수록 0이
  되도록 한다.
- 한 기록의 `decision_uncertainty`는 다섯 클래스 값의 최댓값이다.
- 보조 정보로 클래스별 binary predictive entropy의 최댓값과 평균을 제공한다.

단일 모델 확률로 계산하므로 epistemic uncertainty를 추정하지 못한다. Deep
ensemble, 외부 분포 이동 검출 또는 임상 위험 확률로 해석하지 않는다.

## Coverage와 평가

- 검증 fold의 uncertainty quantile로 90%, 80%, 70% 목표 coverage cutoff를
  고정한다.
- 시험에서는 uncertainty가 cutoff 이하인 기록만 자동 결과 대상으로 간주하고
  실제 coverage, Hamming 오류율, any-label 오류율, macro AUROC, 민감도,
  특이도를 보고한다.
- 10~100% coverage의 risk-coverage curve를 제공한다.
- 80% 목표 coverage를 UI 데모 운영점으로 사용하되 임상적으로 승인된
  threshold라고 표현하지 않는다.

## 기술 품질과의 관계

기술 품질 SQI는 모델 오류 탐지력이 확인되지 않았으므로 uncertainty와 가중
합산하지 않는다. `review_uncertain`, `review_technical`, `review_both`를
독립적인 사유로 표시한다.

## 제한

- 보류 집단과 자동 결과 집단의 질병 유병률과 하위집단 구성이 달라질 수 있다.
- 선택된 일부 기록의 성능 향상이 전체 환자 집단의 임상 효용을 의미하지 않는다.
- baseline v1 시험 결과를 확인한 뒤 수행하는 사후 기술 분석이다. 동일 시험
  결과를 이용해 분류 모델이나 cutoff를 재튜닝하지 않는다.
