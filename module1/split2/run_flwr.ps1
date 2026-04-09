param(
    [int]$NumClients = 5,
    [int]$NumRounds = 20,
    [string]$Model = "dnn",
    [string]$UseSynthetic = "true",
    [string]$Attack = "none",
    [string]$Malicious = "1",
    [string]$BlockchainEnabled = "true",
    [string]$BlockchainBackend = "fabric"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$VenvScripts = Join-Path $RepoRoot "venv\Scripts"
$FlwrExe = Join-Path $VenvScripts "flwr.exe"

if (-not (Test-Path $FlwrExe)) {
    throw "flwr.exe not found in venv: $FlwrExe"
}

$env:Path = "$VenvScripts;$env:Path"
Set-Location $RepoRoot
$env:BATFL_PROJECT_ROOT = "$RepoRoot"

$FlwrConfigPath = Join-Path $env:USERPROFILE ".flwr\config.toml"
if (-not (Test-Path $FlwrConfigPath)) {
    throw "Flower config not found: $FlwrConfigPath"
}

$configText = [System.IO.File]::ReadAllText($FlwrConfigPath)
$sectionPattern = '(?ms)^\[superlink\.local\]\s*(?<body>(?:^(?!\[).*(?:\r?\n)?)+)?'
$sectionMatch = [regex]::Match($configText, $sectionPattern)

$updated = $configText
if ($sectionMatch.Success) {
    $sectionText = $sectionMatch.Value
    $localMatch = [regex]::Match($sectionText, 'options\.num-supernodes\s*=\s*(\d+)')
    if ($localMatch.Success) {
        $supernodes = [int]$localMatch.Groups[1].Value
        if ($supernodes -ne $NumClients) {
            $newSectionText = [regex]::Replace(
                $sectionText,
                'options\.num-supernodes\s*=\s*\d+',
                "options.num-supernodes = $NumClients",
                1
            )
            $updated = $configText.Substring(0, $sectionMatch.Index) + $newSectionText + $configText.Substring($sectionMatch.Index + $sectionMatch.Length)
            Write-Host "[run_flwr] Updated options.num-supernodes from $supernodes to $NumClients in $FlwrConfigPath" -ForegroundColor Yellow
        }
    } else {
        $insert = "options.num-supernodes = $NumClients`r`n"
        $newSectionText = $sectionText + (if ($sectionText.EndsWith("`n")) { "" } else { "`r`n" }) + $insert
        $updated = $configText.Substring(0, $sectionMatch.Index) + $newSectionText + $configText.Substring($sectionMatch.Index + $sectionMatch.Length)
        Write-Host "[run_flwr] Added options.num-supernodes = $NumClients under [superlink.local]" -ForegroundColor Yellow
    }
} else {
    $append = "`r`n[superlink.local]`r`noptions.num-supernodes = $NumClients`r`n"
    $updated = $configText + $append
    Write-Host "[run_flwr] Added [superlink.local] with options.num-supernodes = $NumClients" -ForegroundColor Yellow
}

# Flower can reject TOML with BOM. Always write UTF-8 without BOM.
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($FlwrConfigPath, $updated, $utf8NoBom)

$runConfig = @(
    "num_clients=$NumClients"
    "num_rounds=$NumRounds"
    "model='$Model'"
    "use_synthetic=$UseSynthetic"
    "attack='$Attack'"
    "malicious='$Malicious'"
    "blockchain_enabled=$BlockchainEnabled"
    "blockchain_backend='$BlockchainBackend'"
) -join " "

Write-Host "[run_flwr] Ensure your Flower local config has options.num-supernodes = $NumClients" -ForegroundColor Yellow
Write-Host "[run_flwr] File: $env:USERPROFILE\.flwr\config.toml" -ForegroundColor Yellow

& $FlwrExe run . --run-config $runConfig --stream
