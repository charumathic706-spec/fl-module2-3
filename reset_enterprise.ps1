param(
    [switch]$SetupAfterReset,
    [ValidateSet("raft", "bft")]
    [string]$Consensus = "bft"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path $PSScriptRoot
$EnterpriseDir = Join-Path $RepoRoot "module1\split3\hlf_enterprise"
$TeardownSh = Join-Path $EnterpriseDir "teardown.sh"
$SetupSh = Join-Path $EnterpriseDir "setup.sh"
$EnvFile = Join-Path $EnterpriseDir "fabric_connection.env"

$GeneratedPaths = @(
    (Join-Path $RepoRoot "logs_split2"),
    (Join-Path $RepoRoot "governance_output"),
    (Join-Path $RepoRoot "governance_output_eth"),
    (Join-Path $RepoRoot "governance_output_eth_check"),
    (Join-Path $RepoRoot "module1\logs_split2"),
    (Join-Path $RepoRoot "module1\split2\trust_training_log.json"),
    (Join-Path $RepoRoot "module1\split2\round_events.jsonl"),
    (Join-Path $RepoRoot "module1\split2\split2_training_curves.png"),
    (Join-Path $RepoRoot "module1\split2\.partition_cache_flwr_app.npz")
)

function Convert-ToWslPath([string]$PathText) {
    $full = (Resolve-Path $PathText).Path -replace "\\", "/"
    if ($full -match "^([A-Za-z]):/(.*)$") {
        $drive = $matches[1].ToLower()
        $rest = $matches[2]
        return "/mnt/$drive/$rest"
    }
    return $full
}

function Remove-GeneratedArtifact([string]$PathText) {
    if (Test-Path $PathText) {
        Remove-Item -Recurse -Force $PathText
        Write-Host "[Reset] Removed $PathText" -ForegroundColor DarkGray
    }
}

Write-Host "[Reset] Tearing down enterprise Fabric network..." -ForegroundColor Yellow
if (Test-Path $TeardownSh) {
    $wslTeardown = Convert-ToWslPath $TeardownSh
    & wsl bash -lc "chmod +x '$wslTeardown' ; '$wslTeardown'"
    if ($LASTEXITCODE -ne 0) {
        throw "Enterprise teardown failed during reset."
    }
}

Write-Host "[Reset] Clearing generated Split 2/3 artifacts..." -ForegroundColor Yellow
foreach ($path in $GeneratedPaths) {
    Remove-GeneratedArtifact $path
}
Remove-GeneratedArtifact $EnvFile

if ($SetupAfterReset) {
    if (-not (Test-Path $SetupSh)) {
        throw "Enterprise setup script not found: $SetupSh"
    }
    Write-Host "[Reset] Re-running enterprise setup with consensus=$Consensus..." -ForegroundColor Cyan
    $wslSetup = Convert-ToWslPath $SetupSh
    & wsl bash -lc "export BATFL_ENTERPRISE_CONSENSUS='$Consensus' ; chmod +x '$wslSetup' ; '$wslSetup'"
    if ($LASTEXITCODE -ne 0) {
        throw "Enterprise setup failed after reset."
    }
}

Write-Host "[Reset] Enterprise reset complete." -ForegroundColor Green
