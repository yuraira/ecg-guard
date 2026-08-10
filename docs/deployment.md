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

이 스크립트는 digest로 고정한 Syft `v1.49.0` 컨테이너로 실제 완성 이미지를
스캔해 Python과 Debian 패키지가 모두 존재하는 CycloneDX JSON만 허용한다. 결과는
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

`Dockerfile`의 Python base image, Syft 검사기와 Python 전이 의존성은 각각
digest 또는 exact version으로 고정한다. `requirements-container.lock`은 CPU
배포 이미지에서 검증한 전이 의존성 집합이며, 직접 의존성 변경 시 실제 이미지
재빌드·테스트·SBOM 추출과 함께 갱신한다.

## Render 공개 데모 구성

### 현재 운영 상태 (2026-08-10)

`https://ecg-guard.onrender.com`은 Singapore 리전의 Free Web Service로 운영한다.
최초 공개 검증 배포는 커밋 `4e83474`이며, Render 로그에서 고정 체크포인트 준비,
`0.0.0.0:8501` 서버 기동과 live 전환을 확인했다. 외부에서는 HTTP→HTTPS 301,
HTTPS 루트와 `/_stcore/health`의 HTTP 200 응답을 확인했다. 실제 브라우저에서
합성 ECG 분석, 예측 확률, 12유도 파형, 품질·출처, 결과 삭제와 잘못된 WFDB
헤더의 사용자 오류 표시를 검증했으며 브라우저 오류 로그는 발생하지 않았다.

저장소 루트의 `render.yaml`은 Singapore 리전의 무료 Docker 웹 서비스를
선언한다. Render는 `Dockerfile`을 빌드하고 GitHub CI가 통과한 `main` 커밋만
자동 배포한다. 공개 요청은 Render 로드 밸런서에서 HTTPS로 종료되고 컨테이너는
이미지 자체 헬스체크와 같은 `PORT=8501`에 `0.0.0.0`으로 바인딩한다. Render는
해당 포트를 공개 HTTPS 엔드포인트로 전달한다.

체크포인트는 이미지나 저장소에 넣지 않는다. Render에서는
`ECG_GUARD_FETCH_CHECKPOINT_AT_STARTUP=1`을 설정한다. 컨테이너 진입점은
Docker의 기본 `CMD`를 실행하기 전에 공개 `baseline-v1` GitHub Release에서
모델을 임시 디렉터리로 내려받고 다음 세 조건을 모두 검증한다.

- 응답 및 실제 파일 크기: `23,579,661` bytes
- 추론 프로토콜에 잠긴 SHA-256:
  `44a8ecc96f1ac084db2ef6921bf8e438c1130da6be140d7fc3ac7fe3ecfa2ead`
- 다운로드 상한: 64MiB

다운로드는 같은 디렉터리의 임시 파일에 기록하고 검증에 성공한 뒤에만 원자적으로
최종 경로로 교체한다. 그다음 웹 런처가 시작 전에 체크포인트 해시를 한 번 더
검증한다. 다운로드·크기·해시 검증 중 하나라도 실패하면 웹 서버는 열리지 않는다.

무료 인스턴스는 15분 동안 요청이 없으면 종료되고 다음 요청에서 다시 시작될 수
있다. 파일시스템은 일시적이므로 업로드 및 결과의 영구 저장소로 사용하지 않는다.
Hobby 워크스페이스의 Render 로그 보유기간은 7일이다. 현재 사업자·리전·URL,
로그 항목과 보유 기간, 임시 파일·세션 삭제 및 국외 처리 범위는
`DATA_HANDLING_POLICY.md`에 기록한다.

로컬에서 호스팅 시작 경로를 재현하려면 새 이미지에서 다음 명령을 실행한다.

```powershell
docker run --rm --read-only --tmpfs /tmp:size=256m,mode=1777 `
  --publish 127.0.0.1:10000:8501 `
  --env PORT=8501 `
  --env ECG_GUARD_CHECKPOINT=/tmp/ecg-guard/best_model.pt `
  --env ECG_GUARD_FETCH_CHECKPOINT_AT_STARTUP=1 `
  --env ECG_GUARD_VERIFY_CHECKPOINT_AT_STARTUP=1 `
  ecg-guard:render-smoke
```

이 명령은 실제 공개 Release 다운로드, 이중 해시 검증, 읽기 전용 루트
파일시스템 및 HTTP 헬스체크를 한 번에 검증한다.

`render.yaml`은 별도의 `dockerCommand`를 지정하지 않고 이미지의 exec-form
`CMD`를 사용한다. 따라서 체크포인트 준비와 웹 실행을 셸의 따옴표·연산자 파싱에
의존하지 않는다. `ecg-guard-container-entrypoint`는 과거 Render 설정에 남은
셸 명령도 호환 처리하지만, 이 경로 역시 체크포인트의 크기·SHA-256 검증이나
웹 런처의 시작 검증을 우회하지 않는다.
