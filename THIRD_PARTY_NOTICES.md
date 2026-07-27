# 제3자 자료 및 라이선스

## PTB-XL 1.0.3

ECG Guard의 기준 실험은 다음 공개 데이터셋을 사용합니다.

Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T.
(2022). *PTB-XL, a large publicly available electrocardiography dataset*
(version 1.0.3). PhysioNet. https://doi.org/10.13026/kfzx-aw45

- 데이터 페이지: https://physionet.org/content/ptb-xl/1.0.3/
- 데이터 라이선스: Creative Commons Attribution 4.0 International
- ECG Guard 저장소는 PTB-XL 원자료를 재배포하지 않습니다.
- 사용자는 PhysioNet에서 직접 데이터를 내려받고 해당 라이선스와 인용 조건을
  준수해야 합니다.

ECG Guard의 Apache-2.0 라이선스는 PTB-XL 데이터의 CC BY 4.0 조건을
대체하거나 변경하지 않습니다.

## Python 패키지

실행에 사용하는 Python 패키지와 정확한 버전은 `requirements.txt`와
`requirements-dev.txt`에 기록합니다. 각 패키지는 해당 프로젝트가 정한
개별 라이선스를 따릅니다.

직접 의존성의 버전·공식 저장소·사용 목적은 [`docs/sbom.md`](docs/sbom.md),
기계 판독 정보는 `sbom/direct-dependencies.cdx.json`에 기록합니다. 실제
Docker 실행 이미지의 Python 전이 의존성은 빌드 중
`/app/sbom/python-runtime.cdx.json`으로 생성합니다. Debian 운영체제
패키지까지 포함한 최종 이미지 SBOM은 고정 버전 Syft로 완성 이미지를
스캔해 `sbom/container-runtime.cdx.json`으로 추출합니다.

웹 데모는 Apache License 2.0으로 배포되는 Streamlit을 사용합니다.

- 버전: 1.60.0
- 공식 저장소: https://github.com/streamlit/streamlit
- 라이선스: https://github.com/streamlit/streamlit/blob/develop/LICENSE
