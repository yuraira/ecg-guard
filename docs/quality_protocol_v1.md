# ECG Guard 기술 품질 프로토콜 v1

고정일: 2026-07-27

## 목적

파형이 PTB-XL 학습 분포에서 기술적으로 극단적인지 표시해 사람의 재검토
우선순위를 제공한다. 점수를 임상적 신호 품질 정답, 질병 판정 또는 자동 입력
거부 기준으로 사용하지 않는다.

## 입력과 특징

100 Hz, 10초, 12유도 파형에서 다음 record-level SQI를 계산한다.

- 0.1~0.7 Hz 저주파 에너지 비율의 유도별 최댓값
- 35~49 Hz 고주파 에너지 비율의 유도별 최댓값
- 연속 표본 변화량이 `1e-5 mV` 미만인 비율의 최댓값
- 유도별 0.5~99.5 percentile 진폭 범위의 최댓값과 최솟값
- 표준편차로 정규화한 1차 차분 RMS의 최댓값
- 유도별 표준편차의 최솟값

질병 형태와 artifact가 동일 SQI에 영향을 줄 수 있으므로 개별 특징을 진단
원인으로 해석하지 않는다. 100 Hz 파형의 Nyquist 한계 때문에 50/60 Hz
전원선 간섭 지표는 포함하지 않는다.

## 학습 참조 분포와 점수

- 참조 quantile은 학습 fold 1~8의 17,084개 파형에서만 계산한다.
- 각 SQI의 사전 정의된 위험 방향에 대해 단측 empirical tail probability를
  계산한다.
- record의 `technical_review_score`는 일곱 tail probability의 최댓값이다.
- 99 percentile 이상은 `review`, 99.9 percentile 이상은
  `extreme_outlier`로 표시한다.
- `technical_quality_score` 0~100은 95~100 percentile의 초과 정도를
  가독성을 위해 역변환한 값이며 임상적으로 보정된 확률이 아니다.

## 타당성 점검

- 검증과 시험에서 PTB-XL의 `baseline_drift`, `static_noise`, `burst_noise`,
  `electrodes_problems` 중 하나 이상의 주석 존재 여부와 AUROC/AP를 보고한다.
- 주석 부재는 artifact 음성 정답으로 취급할 수 없으므로 약한 연관성 점검으로만
  해석한다.
- 동결된 baseline v1의 분류 오류와 review score의 연관성을 기술적으로
  보고하지만 인과관계를 주장하지 않는다.

## 사용 제한

- `review` 또는 `extreme_outlier`만으로 ECG를 자동 폐기하지 않는다.
- 필터링이나 파형 수정은 수행하지 않는다.
- PTB-XL 이외 데이터에는 새 참조 분포와 별도 검증이 필요하다.
- 시험 결과를 보고 SQI 정의나 threshold를 수정한 뒤 동일 결과를 최종 성능으로
  다시 주장하지 않는다.

## 참고

- PTB-XL 1.0.3:
  https://physionet.org/content/ptb-xl/1.0.3/
- ECG SQI robustness review:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9006023/
- ECG artifact와 병적 형태 구분의 한계:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4975422/
