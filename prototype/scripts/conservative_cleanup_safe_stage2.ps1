Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ReportsDir = Join-Path $ProjectRoot "prototype\reports"
$DownloadsRoot = Join-Path $ProjectRoot "workspace\downloads\rgb_datasets"
$WorkspaceRoot = Join-Path $ProjectRoot "workspace"
$RawRoot = Join-Path $ProjectRoot "workspace\datasets\raw"
$DuplicateRoot = Join-Path $RawRoot "df40_real_support"

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
        throw "Refusing to operate outside allowed root. Path=$resolvedPath Root=$resolvedRoot"
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

function Get-PathBytes {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return [int64]0
    }
    $item = Get-Item -LiteralPath $Path -Force
    if ($item.PSIsContainer) {
        $measure = Get-ChildItem -LiteralPath $Path -Recurse -Force -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum
        if ($null -eq $measure -or $null -eq $measure.Sum) {
            return [int64]0
        }
        return [int64]$measure.Sum
    }
    return [int64]$item.Length
}

function Add-RemovedRecord {
    param(
        [System.Collections.Generic.List[object]]$List,
        [Parameter(Mandatory = $true)]
        [string]$Category,
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [int64]$SizeBytes,
        [hashtable]$Extra = @{}
    )
    $record = [ordered]@{
        category   = $Category
        path       = $Path
        size_bytes = $SizeBytes
    }
    foreach ($key in $Extra.Keys) {
        $record[$key] = $Extra[$key]
    }
    $List.Add([pscustomobject]$record) | Out-Null
}

$before = @{
    project_total_bytes   = Get-PathBytes -Path $ProjectRoot
    workspace_total_bytes = Get-PathBytes -Path $WorkspaceRoot
    downloads_total_bytes = Get-PathBytes -Path $DownloadsRoot
    duplicate_root_bytes  = Get-PathBytes -Path $DuplicateRoot
}

$removed = New-Object System.Collections.Generic.List[object]
$skipped = New-Object System.Collections.Generic.List[object]
$totalBytesRemoved = [int64]0

$archiveTargets = @(
    @{ pattern = "lists.zip";                              extracted = "workspace\datasets\raw\DeeperForensics-1.0\lists";                              category = "archive_lists" },
    @{ pattern = "manipulated_videos_part_*.zip";         extracted = "workspace\datasets\raw\DeeperForensics-1.0\manipulated_videos";                 category = "archive_deeperforensics_part" },
    @{ pattern = "DF40_faceswap_test.zip";                extracted = "workspace\datasets\raw\df40_test\faceswap";                                     category = "archive_df40_test" },
    @{ pattern = "DF40_fsgan_test.zip";                   extracted = "workspace\datasets\raw\df40_test\fsgan";                                        category = "archive_df40_test" },
    @{ pattern = "FaceForensics++_real_data_for_DF40.zip"; extracted = "workspace\datasets\raw\df40_real_support\FaceForensics++_real_data_for_DF40"; category = "archive_duplicate_real_support" },
    @{ pattern = "Celeb-DF-v2_real_data_for_DF40.zip";    extracted = "workspace\datasets\raw\df40_real_support\Celeb-DF-v2_real_data_for_DF40";      category = "archive_duplicate_real_support" }
)

$downloadsResolved = Assert-UnderRoot -Path $DownloadsRoot -Root $ProjectRoot
foreach ($target in $archiveTargets) {
    $files = Get-ChildItem -LiteralPath $downloadsResolved -Force -File -Filter $target.pattern -ErrorAction SilentlyContinue
    if (-not $files) {
        $skipped.Add([pscustomobject]@{
            category = $target.category
            path     = $target.pattern
            reason   = "archive_missing"
        }) | Out-Null
        continue
    }

    $extractedPath = Join-Path $ProjectRoot $target.extracted
    if (-not (Test-Path -LiteralPath $extractedPath)) {
        foreach ($file in $files) {
            $skipped.Add([pscustomobject]@{
                category = $target.category
                path     = $file.FullName
                reason   = "extracted_dir_missing"
            }) | Out-Null
        }
        continue
    }

    $resolvedExtracted = Assert-UnderRoot -Path $extractedPath -Root $ProjectRoot
    $fileCount = Get-DirectoryFileCount -Path $resolvedExtracted
    if ($fileCount -le 0) {
        foreach ($file in $files) {
            $skipped.Add([pscustomobject]@{
                category = $target.category
                path     = $file.FullName
                reason   = "extracted_dir_empty"
            }) | Out-Null
        }
        continue
    }

    foreach ($file in $files) {
        $resolvedFile = Assert-UnderRoot -Path $file.FullName -Root $downloadsResolved
        $sizeBytes = [int64]$file.Length
        Remove-Item -LiteralPath $resolvedFile -Force
        $totalBytesRemoved += $sizeBytes
        Add-RemovedRecord -List $removed -Category $target.category -Path $resolvedFile -SizeBytes $sizeBytes -Extra @{
            extracted_path  = $resolvedExtracted
            extracted_files = $fileCount
        }
    }
}

$duplicateTargets = @(
    "workspace\datasets\raw\df40_real_support\FaceForensics++_real_data_for_DF40",
    "workspace\datasets\raw\df40_real_support\Celeb-DF-v2_real_data_for_DF40"
)

foreach ($relativePath in $duplicateTargets) {
    $fullPath = Join-Path $ProjectRoot $relativePath
    if (-not (Test-Path -LiteralPath $fullPath)) {
        $skipped.Add([pscustomobject]@{
            category = "duplicate_real_support_dir"
            path     = $fullPath
            reason   = "dir_missing"
        }) | Out-Null
        continue
    }

    $resolvedDir = Assert-UnderRoot -Path $fullPath -Root $DuplicateRoot
    $fileCount = Get-DirectoryFileCount -Path $resolvedDir
    if ($fileCount -le 0) {
        $skipped.Add([pscustomobject]@{
            category = "duplicate_real_support_dir"
            path     = $resolvedDir
            reason   = "dir_empty"
        }) | Out-Null
        continue
    }

    $sizeBytes = Get-PathBytes -Path $resolvedDir
    Remove-Item -LiteralPath $resolvedDir -Recurse -Force
    $totalBytesRemoved += $sizeBytes
    Add-RemovedRecord -List $removed -Category "duplicate_real_support_dir" -Path $resolvedDir -SizeBytes $sizeBytes -Extra @{
        removed_files = $fileCount
    }
}

if (Test-Path -LiteralPath $DuplicateRoot) {
    $remaining = @(Get-ChildItem -LiteralPath $DuplicateRoot -Force -ErrorAction SilentlyContinue)
    if ($remaining.Count -eq 0) {
        $resolvedParent = Assert-UnderRoot -Path $DuplicateRoot -Root $ProjectRoot
        Remove-Item -LiteralPath $resolvedParent -Force
        Add-RemovedRecord -List $removed -Category "duplicate_real_support_parent" -Path $resolvedParent -SizeBytes 0
    }
}

$after = @{
    project_total_bytes   = Get-PathBytes -Path $ProjectRoot
    workspace_total_bytes = Get-PathBytes -Path $WorkspaceRoot
    downloads_total_bytes = Get-PathBytes -Path $DownloadsRoot
    duplicate_root_bytes  = Get-PathBytes -Path $DuplicateRoot
}

$timestamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$jsonPath = Join-Path $ReportsDir ("conservative_cleanup_stage2_report_" + $timestamp + ".json")
$mdPath = Join-Path $ReportsDir ("conservative_cleanup_stage2_execution_" + $timestamp + ".md")

$report = [ordered]@{
    generated_at        = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")
    project_root        = $ProjectRoot
    total_removed_bytes = $totalBytesRemoved
    total_removed_gb    = [math]::Round(($totalBytesRemoved / 1GB), 3)
    before              = $before
    after               = $after
    removed_count       = $removed.Count
    skipped_count       = $skipped.Count
    removed             = [object[]]$removed.ToArray()
    skipped             = [object[]]$skipped.ToArray()
}
$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $jsonPath -Encoding UTF8

$lines = @()
$lines += "# Conservative Cleanup Stage 2"
$lines += ""
$lines += "Executed on ``" + (Get-Date -Format "yyyy-MM-dd") + "``."
$lines += ""
$lines += "## Scope"
$lines += ""
$lines += "Only the safest second-stage cleanup categories were removed:"
$lines += ""
$lines += "1. DeeperForensics and DF40 support archives after verified extraction"
$lines += "2. The duplicated ``df40_real_support`` extracted mirrors that were audited as exact overlaps with local FF++ / Celeb-DF-v2 real subsets"
$lines += ""
$lines += "## Before vs After"
$lines += ""
$lines += "| Scope | Before (GB) | After (GB) | Delta (GB) |"
$lines += "| --- | ---: | ---: | ---: |"
$lines += ("| Project total | {0:N3} | {1:N3} | -{2:N3} |" -f ($before.project_total_bytes / 1GB), ($after.project_total_bytes / 1GB), (($before.project_total_bytes - $after.project_total_bytes) / 1GB))
$lines += ("| Workspace total | {0:N3} | {1:N3} | -{2:N3} |" -f ($before.workspace_total_bytes / 1GB), ($after.workspace_total_bytes / 1GB), (($before.workspace_total_bytes - $after.workspace_total_bytes) / 1GB))
$lines += ("| Downloads total | {0:N3} | {1:N3} | -{2:N3} |" -f ($before.downloads_total_bytes / 1GB), ($after.downloads_total_bytes / 1GB), (($before.downloads_total_bytes - $after.downloads_total_bytes) / 1GB))
$lines += ("| Duplicate real-support root | {0:N3} | {1:N3} | -{2:N3} |" -f ($before.duplicate_root_bytes / 1GB), ($after.duplicate_root_bytes / 1GB), (($before.duplicate_root_bytes - $after.duplicate_root_bytes) / 1GB))
$lines += ""
$lines += "## Totals"
$lines += ""
$lines += ("- Removed items: {0}" -f $removed.Count)
$lines += ("- Recovered space: {0:N3} GB" -f ($totalBytesRemoved / 1GB))
$lines += ("- JSON report: ``{0}``" -f $jsonPath)
$lines += ""
$lines += "## Removed Categories"
$lines += ""
$categoryGroups = $removed | Group-Object category | Sort-Object Name
$lines += "| Category | Count | Removed size (GB) |"
$lines += "| --- | ---: | ---: |"
foreach ($group in $categoryGroups) {
    $bytes = ($group.Group | Measure-Object size_bytes -Sum).Sum
    $lines += ("| ``{0}`` | {1} | {2:N3} |" -f $group.Name, $group.Count, ($bytes / 1GB))
}

$lines | Set-Content -LiteralPath $mdPath -Encoding UTF8

Write-Output "Cleanup stage 2 complete."
Write-Output ("Recovered GB: " + [math]::Round(($totalBytesRemoved / 1GB), 3))
Write-Output ("JSON report: " + $jsonPath)
Write-Output ("Markdown report: " + $mdPath)
