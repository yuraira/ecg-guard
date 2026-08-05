[CmdletBinding()]
param(
    [string]$Image = "ecg-guard:baseline-v1",
    [string]$Output = "sbom/container-runtime.cdx.json",
    [string]$SyftImage = "anchore/syft:v1.49.0@sha256:13b53ebabe3d215268c90cf8fb9b875f0183908245f376fd4b3a2cb69d21d484"
)

$ErrorActionPreference = "Stop"
$repository = Split-Path -Parent $PSScriptRoot
$outputPath = [System.IO.Path]::GetFullPath(
    (Join-Path $repository $Output)
)
$outputDirectory = Split-Path -Parent $outputPath
$outputName = Split-Path -Leaf $outputPath
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

& docker image inspect $Image | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker image is unavailable: $Image"
}

& docker pull $SyftImage
if ($LASTEXITCODE -ne 0) {
    throw "Unable to pull the pinned Syft image: $SyftImage"
}

$outputVolume = "${outputDirectory}:/output"
& docker run --rm `
    --volume "/var/run/docker.sock:/var/run/docker.sock" `
    --volume $outputVolume `
    $SyftImage `
    scan $Image `
    --output "cyclonedx-json=/output/$outputName"
if ($LASTEXITCODE -ne 0) {
    throw "Syft failed to scan $Image"
}

$bom = Get-Content -Raw -Encoding utf8 -LiteralPath $outputPath |
    ConvertFrom-Json
$components = @($bom.components)
$pythonComponents = @(
    $components | Where-Object { $_.purl -like "pkg:pypi/*" }
)
$debianComponents = @(
    $components | Where-Object { $_.purl -like "pkg:deb/debian/*" }
)
if ($bom.bomFormat -ne "CycloneDX" -or $components.Count -eq 0) {
    throw "The generated file is not a non-empty CycloneDX SBOM"
}
if ($pythonComponents.Count -eq 0 -or $debianComponents.Count -eq 0) {
    throw (
        "The final SBOM must include both Python and Debian packages; " +
        "python=$($pythonComponents.Count), debian=$($debianComponents.Count)"
    )
}

$imageInspection = docker image inspect --format "{{json .}}" $Image |
    ConvertFrom-Json
$syftInspection = docker image inspect --format "{{json .}}" $SyftImage |
    ConvertFrom-Json
$provenance = [ordered]@{
    schema_version = "1.0"
    image = $Image
    image_id = $imageInspection.Id
    image_repo_digests = @($imageInspection.RepoDigests)
    syft_image = $SyftImage
    syft_image_id = $syftInspection.Id
    syft_repo_digests = @($syftInspection.RepoDigests)
    sbom_file = $outputName
    sbom_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $outputPath
    ).Hash.ToLowerInvariant()
    component_count = $components.Count
    python_component_count = $pythonComponents.Count
    debian_component_count = $debianComponents.Count
}
$provenancePath = Join-Path `
    $outputDirectory `
    "container-runtime.provenance.local.json"
$provenance |
    ConvertTo-Json -Depth 6 |
    Set-Content -Encoding utf8 -LiteralPath $provenancePath

Write-Output "sbom=$outputPath"
Write-Output "sbom_sha256=$($provenance.sbom_sha256)"
Write-Output "components=$($components.Count)"
Write-Output "python_components=$($pythonComponents.Count)"
Write-Output "debian_components=$($debianComponents.Count)"
Write-Output "image_id=$($imageInspection.Id)"
Write-Output "provenance=$provenancePath"
