Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ReportsDir = Join-Path $ProjectRoot "prototype\reports"
$RunsRoot = Join-Path $ProjectRoot "pilot\passive_baseline\runs"
$DownloadsRoot = Join-Path $ProjectRoot "workspace\downloads\rgb_datasets"

function Resolve-ExistingPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Assert-UnderRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Root
    )
    $resolvedPath = Resolve-ExistingPath -Path $Path
    if ($null -eq $resolvedPath) {
        throw "Path does not exist: $Path"
    }
    $resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
    if (-not $resolvedPath.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to operate outside project root. Path=$resolvedPath Root=$resolvedRoot"
    }
    return $resolvedPath
}

function Get-DirectoryFileCount {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )
    return (Get-ChildItem -LiteralPath $Path -Recurse -Force -File -ErrorAction SilentlyContinue | Measure-Object).Count
}

$removed = New-Object System.Collections.Generic.List[object]
$skipped = New-Object System.Collections.Generic.List[object]
$totalBytesRemoved = [int64]0

# 1) Remove per-epoch checkpoints, keep best.pt and any non-epoch artifacts.
$runsResolved = Assert-UnderRoot -Path $RunsRoot -Root $ProjectRoot
$epochFiles = Get-ChildItem -LiteralPath $runsResolved -Recurse -Force -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "epoch_*.pt" -and $_.FullName -match "\\epochs\\" }

foreach ($file in $epochFiles) {
    $resolvedFile = Assert-UnderRoot -Path $file.FullName -Root $runsResolved
    $sizeBytes = $file.Length
    Remove-Item -LiteralPath $resolvedFile -Force
    $totalBytesRemoved += $sizeBytes
    $removed.Add([pscustomobject]@{
        category   = "epoch_checkpoint"
        path       = $resolvedFile
        size_bytes = $sizeBytes
    }) | Out-Null
}

# 2) Remove stale partial download artifacts from rgb_datasets.
$downloadsResolved = Assert-UnderRoot -Path $DownloadsRoot -Root $ProjectRoot
$partFiles = Get-ChildItem -LiteralPath $downloadsResolved -Force -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "*.part" }

foreach ($file in $partFiles) {
    $resolvedFile = Assert-UnderRoot -Path $file.FullName -Root $downloadsResolved
    $sizeBytes = $file.Length
    Remove-Item -LiteralPath $resolvedFile -Force
    $totalBytesRemoved += $sizeBytes
    $removed.Add([pscustomobject]@{
        category   = "partial_download"
        path       = $resolvedFile
        size_bytes = $sizeBytes
    }) | Out-Null
}

# 3) Remove source archives only when the extracted raw dataset exists and is non-empty.
$archiveTargets = @(
    @{ archive = "workspace\downloads\rgb_datasets\UADFV.zip";               extracted = "workspace\datasets\raw\UADFV" },
    @{ archive = "workspace\downloads\rgb_datasets\Celeb-DF-v1.zip";         extracted = "workspace\datasets\raw\celebdf" },
    @{ archive = "workspace\downloads\rgb_datasets\FaceForensics++.zip";     extracted = "workspace\datasets\raw\FaceForensics++" },
    @{ archive = "workspace\downloads\rgb_datasets\Celeb-DF-v2.zip";         extracted = "workspace\datasets\raw\Celeb-DF-v2" },
    @{ archive = "workspace\downloads\rgb_datasets\DFDCP.zip";               extracted = "workspace\datasets\raw\DFDCP" },
    @{ archive = "workspace\downloads\rgb_datasets\DF40_deepfacelab.zip";    extracted = "workspace\datasets\raw\df40\deepfacelab" },
    @{ archive = "workspace\downloads\rgb_datasets\DF40_faceswap.zip";       extracted = "workspace\datasets\raw\df40\faceswap" },
    @{ archive = "workspace\downloads\rgb_datasets\DF40_simswap.zip";        extracted = "workspace\datasets\raw\df40\simswap" },
    @{ archive = "workspace\downloads\rgb_datasets\DF40_fsgan.zip";          extracted = "workspace\datasets\raw\df40\fsgan" },
    @{ archive = "workspace\downloads\rgb_datasets\DF40_inswap.zip";         extracted = "workspace\datasets\raw\df40\inswap" }
)

foreach ($target in $archiveTargets) {
    $archivePath = Join-Path $ProjectRoot $target.archive
    $extractedPath = Join-Path $ProjectRoot $target.extracted

    if (-not (Test-Path -LiteralPath $archivePath)) {
        $skipped.Add([pscustomobject]@{
            category = "archive"
            path     = $archivePath
            reason   = "archive_missing"
        }) | Out-Null
        continue
    }

    if (-not (Test-Path -LiteralPath $extractedPath)) {
        $skipped.Add([pscustomobject]@{
            category = "archive"
            path     = $archivePath
            reason   = "extracted_dir_missing"
        }) | Out-Null
        continue
    }

    $resolvedExtracted = Assert-UnderRoot -Path $extractedPath -Root $ProjectRoot
    $fileCount = Get-DirectoryFileCount -Path $resolvedExtracted
    if ($fileCount -le 0) {
        $skipped.Add([pscustomobject]@{
            category = "archive"
            path     = $archivePath
            reason   = "extracted_dir_empty"
        }) | Out-Null
        continue
    }

    $resolvedArchive = Assert-UnderRoot -Path $archivePath -Root $ProjectRoot
    $archiveItem = Get-Item -LiteralPath $resolvedArchive -Force
    $sizeBytes = $archiveItem.Length
    Remove-Item -LiteralPath $resolvedArchive -Force
    $totalBytesRemoved += $sizeBytes
    $removed.Add([pscustomobject]@{
        category        = "archive"
        path            = $resolvedArchive
        size_bytes      = $sizeBytes
        extracted_path  = $resolvedExtracted
        extracted_files = $fileCount
    }) | Out-Null
}

$timestamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$reportPath = Join-Path $ReportsDir ("conservative_cleanup_report_" + $timestamp + ".json")
$report = @{
    generated_at         = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")
    project_root         = $ProjectRoot
    total_removed_bytes  = $totalBytesRemoved
    total_removed_gb     = [math]::Round(($totalBytesRemoved / 1GB), 3)
    removed_count        = $removed.Count
    skipped_count        = $skipped.Count
    removed              = [object[]]$removed.ToArray()
    skipped              = [object[]]$skipped.ToArray()
}

$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $reportPath -Encoding UTF8

Write-Output ("Cleanup complete.")
Write-Output ("Removed files: " + $removed.Count)
Write-Output ("Recovered GB: " + [math]::Round(($totalBytesRemoved / 1GB), 3))
Write-Output ("Report: " + $reportPath)
