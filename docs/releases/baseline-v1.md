# ECG Guard Baseline v1

ECG Guard의 동결된 residual 1D CNN 기준 모델과 재현성 자료를 공개합니다.

## 포함 내용

- SHA-256으로 잠긴 PyTorch 체크포인트
- 모델 카드와 추론 프로토콜
- 파일별 체크섬과 release manifest
- 직접 의존성 및 완성 Docker 이미지 CycloneDX SBOM
- 재현 가능하게 생성한 전체 ZIP 패키지
- Apache-2.0 라이선스와 PTB-XL 제3자 고지

## 사용 범위

이 모델은 공개 심전도 데이터의 연구·교육과 소프트웨어 재현성 검토를 위한
것입니다. 실제 환자의 진단, 선별, 치료 또는 응급 의사결정에 사용할 수
있도록 검증된 의료기기가 아닙니다.

## 무결성

다운로드 후 `SHA256SUMS.txt`로 파일을 검증하세요. 체크포인트의 고정
SHA-256은 다음과 같습니다.

`44a8ecc96f1ac084db2ef6921bf8e438c1130da6be140d7fc3ac7fe3ecfa2ead`
