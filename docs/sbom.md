# ECG Guard SBOM

## 작성 범위

소스 저장소에는 제출 서식과 모델 Release에서 사용할 직접 의존성 CycloneDX
SBOM을 포함한다. Docker 이미지를 빌드할 때는 컨테이너에 실제 설치된 전이
Python 의존성을 다시 조사해 `/app/sbom/python-runtime.cdx.json`을 생성한다.
완성 이미지에서는 Syft로 Python과 Debian 운영체제 패키지를 모두 조사해
`sbom/container-runtime.cdx.json`을 생성한다.

직접 의존성 SBOM과 실행 이미지 SBOM을 구분하는 이유는 CUDA 개발 환경과 CPU
배포 환경의 PyTorch 빌드 및 전이 패키지가 다르기 때문이다. 최종 공개 서비스의
공급망 검토에는 완성 이미지를 스캔한 `container-runtime.cdx.json`을
사용해야 한다.

## 제출 서식용 직접 의존성

| 라이브러리 | 버전 | 라이선스 | 공식 저장소 | 사용 목적 및 주요 기능 |
|---|---:|---|---|---|
| PyTorch | 2.12.1 | BSD-3-Clause | https://github.com/pytorch/pytorch | residual 1D CNN 정의, 학습 및 추론 |
| NumPy | 2.5.1 | BSD-3-Clause 및 배포본 내 제3자 고지 | https://github.com/numpy/numpy | ECG 배열 처리와 수치 연산 |
| pandas | 3.0.5 | BSD-3-Clause | https://github.com/pandas-dev/pandas | PTB-XL 메타데이터와 평가표 처리 |
| scikit-learn | 1.9.0 | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn | AUROC·average precision 등 평가 지표 |
| WFDB | 4.3.1 | MIT | https://github.com/MIT-LCP/wfdb-python | WFDB 심전도 헤더 및 파형 로딩 |
| Matplotlib | 3.11.1 | PSF 기반 Matplotlib License | https://github.com/matplotlib/matplotlib | ECG 파형과 확률 시각화 |
| Streamlit | 1.60.0 | Apache-2.0 | https://github.com/streamlit/streamlit | 연구용 웹 데모 UI |
| pytest | 9.1.1 | MIT | https://github.com/pytest-dev/pytest | 개발·검증용 자동 테스트, 운영 이미지 제외 |

버전은 `pyproject.toml`, `requirements.txt` 및 `requirements-dev.txt`에서
고정한다. 위 라이선스는 각 프로젝트의 설치 메타데이터와 공식 저장소를 기준으로
기록했으며, 각 배포본에 포함된 전이 의존성과 번들 구성요소는 기계 판독 SBOM 및
해당 패키지의 고지를 따른다.

## 생성 방법

현재 설치 환경의 직접 의존성:

```powershell
.\.venv\Scripts\python.exe .\scripts\generate_sbom.py `
  --direct-only `
  --output .\sbom\direct-dependencies.cdx.json
```

현재 설치 환경의 전체 실행 의존성:

```powershell
.\.venv\Scripts\python.exe .\scripts\generate_sbom.py `
  --output .\sbom\runtime-local.cdx.json
```

`runtime-local.cdx.json`은 로컬 검증 산출물이며 Git에 커밋하지 않는다. 최종
배포 SBOM은 Docker 빌드가 끝난 뒤 다음 명령으로 추출한다.

```powershell
.\scripts\extract_container_sbom.ps1
```

스크립트는 digest로 고정된 Syft `v1.49.0` 이미지를 사용하고, 결과에 PyPI와
Debian 패키지가 모두 있는지 검증한다. Docker image ID, Syft image digest와
SBOM SHA-256은 별도의 로컬 provenance 파일에 남긴다. Python 전이 버전은
`requirements-container.lock`에도 고정해 SBOM이 단순한 사후 목록에 그치지
않도록 한다.

검증을 통과한 Release 후보의 source commit, image ID, base image와 Syft
digest, SBOM 해시 및 헬스체크 결과는
`sbom/container-runtime.provenance.json`에 고정한다. `.local.json` 파일은
매 실행 진단용이고, 확정된 provenance JSON만 모델 Release에 포함한다.

## 모델 Release와 현재 배포 이미지의 버전 경계

`baseline-v1` GitHub Release의 SBOM·provenance·ZIP은 태그
`1468ccaa6d3f819660d342a67589efdfb4e79109`에서 함께 생성한 불변 모델 패키지다.
Release manifest와 `SHA256SUMS.txt`가 서로 참조하므로 이후 배포 코드 변경을
이 자산에 덮어쓰지 않는다.

현재 `main`의 컨테이너 SBOM은 시작 명령 호환 처리를 포함한 이미지
`sha256:110ead502269e264286f90e276212b8a83f388903bfbaa0eb5e0982f84b592a3`에서
다시 추출했으며 SHA-256은
`43cc83f8399271bcc34d82975fecdf7e199f95d7e946fe451764b722abd3abf4`다.
모델 가중치 SHA-256은 두 버전 모두
`44a8ecc96f1ac084db2ef6921bf8e438c1130da6be140d7fc3ac7fe3ecfa2ead`로
동일하다. 모델 Release 자체를 재현할 때는 Release 자산을, 현재 배포 이미지의
공급망을 검토할 때는 `main`의 SBOM과 provenance를 사용한다.
