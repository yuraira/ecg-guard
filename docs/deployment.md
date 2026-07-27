# Docker 실행과 공개 배포 조건

## 기본 원칙

기본 Compose 설정은 포트를 `127.0.0.1`에만 바인딩한다. 따라서 개발 PC의
브라우저에서만 접속할 수 있으며 인터넷 공개 서비스가 아니다. 체크포인트도
이미지에 포함하지 않고 읽기 전용 볼륨으로 주입한다.

## 로컬 Docker 실행

다음 조건을 먼저 확인한다.

- Docker Engine 또는 Docker Desktop 설치
- `outputs/baseline/best_model.pt` 존재
- 체크포인트 SHA-256이 잠긴 추론 프로토콜과 일치

PowerShell에서 빌드 출처를 기록하고 실행한다.

```powershell
$env:BUILD_DATE=(Get-Date).ToUniversalTime().ToString("o")
$env:VCS_REF=(git rev-parse HEAD)
docker compose build
docker compose up
```

컨테이너의 시작 명령은 웹 서버를 열기 전에 마운트된 체크포인트의 SHA-256을
잠긴 추론 프로토콜과 비교한다. 파일이 없거나 해시가 다르면 프로세스가
종료되므로 Streamlit HTTP 헬스체크도 정상 상태가 될 수 없다.

브라우저에서 `http://127.0.0.1:8501`을 열고 다음 헬스체크를 확인한다.

```powershell
docker inspect `
  --format "{{json .State.Health}}" `
  ecg-guard
```

종료:

```powershell
docker compose down
```

## 완성 이미지 SBOM

Docker 빌드는 CPU 실행 환경에 실제 설치된 Python 전이 의존성을 조사해 이미지
내부 `/app/sbom/python-runtime.cdx.json`에 기록한다. 이것만으로는 Debian
운영체제 패키지를 포함하지 못하므로 최종 공급망 SBOM으로 사용하지 않는다.

```powershell
.\scripts\extract_container_sbom.ps1
```

이 스크립트는 Syft `v1.49.0` 컨테이너로 실제 완성 이미지를 스캔해 Python과
Debian 패키지가 모두 존재하는 CycloneDX JSON만 허용한다. 결과는
`sbom/container-runtime.cdx.json`, 이미지 ID·Syft 이미지 digest·SBOM 해시는
무시된 로컬 검증 파일 `sbom/container-runtime.provenance.local.json`에
기록한다.

`container-runtime.cdx.json`은 실제로 배포할 이미지와 같은 고유 image ID에서
추출해야 한다. 직접 의존성 표나 이미지 내부 Python 전용 목록으로 최종
컨테이너 공급망 SBOM을 대체하지 않는다. Syft 사용법과 버전은
[Anchore 공식 저장소](https://github.com/anchore/syft) 및
[v1.49.0 Release](https://github.com/anchore/syft/releases/tag/v1.49.0)에서
확인할 수 있다.

## 인터넷 공개 전 필수 조건

다음 조건이 모두 충족되기 전에는 Compose의 포트 바인딩을
`0.0.0.0` 또는 공개 호스트로 변경하지 않는다.

1. `baseline-v1` GitHub Release와 체크포인트가 승인 없이 다운로드 가능하다.
2. Release 체크포인트, 추론 프로토콜 및 모델 카드의 SHA-256이 일치한다.
3. 배포 이미지를 고유 image ID 또는 registry digest로 고정하고 해당
   이미지에서 전체 SBOM을 추출한다.
4. HTTPS를 종료하는 신뢰 가능한 reverse proxy 또는 관리형 플랫폼을 사용한다.
5. 실제 환자 파일을 금지하고 공개·비식별 연구 데이터 확인 절차를 유지한다.
6. 호스팅 사업자, 접속 로그 항목·보유 기간, 쿠키 및 국외 이전 여부를
   `DATA_HANDLING_POLICY.md`에 반영한다.
7. 접근 로그와 오류 로그에 업로드 이름, WFDB 헤더 및 원문 파형이 기록되지
   않는지 검증한다.
8. 요청 속도 제한, 50MB 본문 제한, XSRF 보호, 컨테이너 비특권 실행 및
   보안 업데이트 절차를 적용한다.
9. 실제 공개 URL에서 업로드, 세션 삭제, 오류 처리와 헬스체크를 점검한다.

이 조건은 연구용 공개 데모의 최소 운영 기준이다. 실제 환자 의료정보나 의료적
의사결정을 처리할 수 있는 인증·법적 동의·임상 검증 체계를 의미하지 않는다.

현재 `Dockerfile`의 `python:3.12-slim`은 유지보수되는 이동 태그다. 최종
배포 후보를 검증한 뒤에는 해당 빌드가 사용한 base image digest와 완성 이미지
digest를 배포 기록에 고정해야 한다.
