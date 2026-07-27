# Baseline v1 체크포인트 공개 절차

## 공개 전 검증과 패키징

다음 명령은 동결된 추론 프로토콜의 SHA-256과 실제 체크포인트를 비교하고,
절대 로컬 경로와 필수 메타데이터를 검사한 뒤 공개용 파일을 `dist`에 모은다.
추적·미추적 변경사항이 있는 Git 작업트리에서는 공개 패키지 생성을 거부하므로
릴리스할 변경사항을 먼저 커밋해야 한다.

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_model_release.py
```

커밋 전 패키지 구조만 검사하려면 `--allow-dirty`를 사용할 수 있지만, 이때
생성된 manifest에는 `source_dirty: true`가 기록되며 해당 파일은 게시하면
안 된다.

생성되는 공개 패키지는 다음 파일을 포함한다.

```text
dist/ecg-guard-baseline-v1
├─ best_model.pt
├─ baseline_v1_inference.json
├─ MODEL_CARD.md
├─ README.md
├─ release-manifest.json
├─ SHA256SUMS.txt
├─ direct-dependencies.cdx.json
├─ container-runtime.cdx.json
├─ LICENSE
└─ THIRD_PARTY_NOTICES.md
```

동일 내용을 고정 파일 순서·권한·커밋 시각으로 묶은
`dist/ecg-guard-baseline-v1.zip`도 생성한다. 따라서 같은 커밋과 같은 입력
파일에서 다시 실행하면 ZIP 해시가 동일하다.

PTB-XL 원자료, 시험 예측 원본, 로컬 환경 변수, 비밀키 및 사용자 업로드 파일은
포함하지 않는다.

## GitHub Release 게시

먼저 패키지의 체크섬과 내용을 직접 확인한다. 그 다음 GitHub CLI에 로그인된
환경에서 다음과 같이 공개 Release를 생성할 수 있다.

```powershell
$assets = @(
  Get-ChildItem .\dist\ecg-guard-baseline-v1 -File |
    ForEach-Object FullName
)
$assets += (Resolve-Path .\dist\ecg-guard-baseline-v1.zip).Path
gh release create baseline-v1 @assets `
  --repo yuraira/ecg-guard `
  --target (git rev-parse HEAD) `
  --title "ECG Guard Baseline v1" `
  --notes-file .\docs\releases\baseline-v1.md
```

게시는 외부 공개 작업이다. 실행 전 저장소가 공개 상태인지, 태그가 올바른
소스 커밋을 가리키는지, Release 초안이 아니라 공개 상태인지 확인한다.

게시 후 로그아웃 상태의 브라우저에서 다음 주소와 `best_model.pt` 다운로드가
승인 없이 열리는지 확인한다.

`https://github.com/yuraira/ecg-guard/releases/tag/baseline-v1`

인증 헤더를 보내지 않는 별도 임시 디렉터리에서도 다음처럼 검증한다.

```powershell
$url = "https://github.com/yuraira/ecg-guard/releases/download/" +
  "baseline-v1/best_model.pt"
Invoke-WebRequest -Uri $url -OutFile .\best_model.pt
(Get-FileHash -Algorithm SHA256 .\best_model.pt).Hash.ToLowerInvariant()
```

다운로드한 파일의 SHA-256을 `SHA256SUMS.txt` 및 추론 프로토콜과 일치시킨 뒤
대회 AI 모델 명세서에 실제 Release URL을 기재한다.
