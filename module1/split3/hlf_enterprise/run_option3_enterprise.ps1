param(
    [int]$NumClients = 2,
    [int]$NumRounds = 20,
    [ValidateSet("dnn", "logistic")]
    [string]$Model = "logistic",
    [ValidateSet("raft", "bft")]
    [string]$Consensus = "bft",
    [switch]$Reset,
    [switch]$ReplayTrustLog,
    [switch]$SkipFabricSetup,
    [switch]$SkipDashboard,
    [int]$DashboardPort = 5000
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$VenvScripts = Join-Path $RepoRoot "venv\Scripts"
$PythonExe = Join-Path $VenvScripts "python.exe"
$Split2Runner = Join-Path $RepoRoot "module1\split2\run_flwr.ps1"
$EnterpriseEnvFile = Join-Path $PSScriptRoot "fabric_connection.env"
$EnterpriseSetupSh = Join-Path $PSScriptRoot "setup.sh"
$EnterpriseHealthSh = Join-Path $PSScriptRoot "health_check.sh"
$EnterpriseTeardownSh = Join-Path $PSScriptRoot "teardown.sh"

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

if (-not (Test-Path $PythonExe)) {
    throw "python.exe not found in venv: $PythonExe"
}
if (-not (Test-Path $Split2Runner)) {
    throw "Split 2 runner not found: $Split2Runner"
}

$env:Path = "$VenvScripts;$env:Path"
Set-Location $RepoRoot

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
        Write-Host "[Enterprise] Removed $PathText" -ForegroundColor DarkGray
    }
}

function Invoke-FreshReset {
    Write-Host "[Enterprise] Reset requested: tearing down Fabric and clearing generated artifacts..." -ForegroundColor Yellow
    if (Test-Path $EnterpriseTeardownSh) {
        $wslTeardown = Convert-ToWslPath $EnterpriseTeardownSh
        & wsl bash -lc "chmod +x '$wslTeardown' ; '$wslTeardown'"
        if ($LASTEXITCODE -ne 0) {
            throw "Enterprise Fabric teardown failed during reset."
        }
    }

    foreach ($path in $GeneratedPaths) {
        Remove-GeneratedArtifact $path
    }

    if (Test-Path $EnterpriseEnvFile) {
        Remove-GeneratedArtifact $EnterpriseEnvFile
    }
}

if ($Reset) {
    $SkipFabricSetup = $false
    Invoke-FreshReset
}

if (-not $SkipFabricSetup) {
    $wslScript = Convert-ToWslPath $EnterpriseSetupSh
    Write-Host "[Enterprise] Setting up Hyperledger Fabric (isolated profile: CA + CouchDB + $Consensus)..." -ForegroundColor Cyan
    & wsl bash -lc "export BATFL_ENTERPRISE_CONSENSUS='$Consensus' ; chmod +x '$wslScript' ; '$wslScript'"
    if ($LASTEXITCODE -ne 0) {
        throw "Enterprise Fabric setup failed."
    }
}

if (-not (Test-Path $EnterpriseEnvFile)) {
    throw "Enterprise Fabric env file missing: $EnterpriseEnvFile"
}
if (-not (Test-Path $EnterpriseHealthSh)) {
    throw "Enterprise health-check script missing: $EnterpriseHealthSh"
}

$wslHealthScript = Convert-ToWslPath $EnterpriseHealthSh
Write-Host "[Enterprise] Running Fabric preflight health check..." -ForegroundColor Cyan
& wsl bash -lc "chmod +x '$wslHealthScript' ; '$wslHealthScript' '$Consensus'"
if ($LASTEXITCODE -ne 0) {
    throw "Enterprise Fabric health check failed."
}

$env:BATFL_FABRIC_ENV_FILE = $EnterpriseEnvFile
Write-Host "[Enterprise] Using Fabric env file: $($env:BATFL_FABRIC_ENV_FILE)" -ForegroundColor Cyan

if (-not $SkipDashboard) {
    Write-Host "[Enterprise] Starting dashboard at http://127.0.0.1:$DashboardPort" -ForegroundColor Cyan
    Start-Process -FilePath $PythonExe -ArgumentList @(
        "module1/dashboard_server.py",
        "--log", "logs_split2/trust_training_log.json",
        "--host", "127.0.0.1",
        "--port", "$DashboardPort",
        "--expected_rounds", "$NumRounds"
    ) | Out-Null
}

Write-Host "[Enterprise] Running Split 2 with Fabric governance..." -ForegroundColor Cyan
& powershell -ExecutionPolicy Bypass -File $Split2Runner `
    -NumClients $NumClients `
    -NumRounds $NumRounds `
    -Model $Model `
    -BlockchainEnabled "true" `
    -BlockchainBackend "fabric"
if ($LASTEXITCODE -ne 0) {
    throw "Split 2 run failed in enterprise mode."
}

$TrustLogPath = Join-Path $RepoRoot "logs_split2\trust_training_log.json"
if (-not (Test-Path $TrustLogPath)) {
    throw "Split 2 completed but trust log was not found: $TrustLogPath"
}

$roundCountRaw = & $PythonExe -c "import json,sys; p=sys.argv[1]; print(len(json.load(open(p, encoding='utf-8'))))" $TrustLogPath
if ($LASTEXITCODE -ne 0) {
    throw "Failed to read round count from trust log: $TrustLogPath"
}
[int]$RoundCount = ($roundCountRaw | Select-Object -Last 1)
if ($RoundCount -lt $NumRounds) {
    throw (
        "Split 2 produced only $RoundCount rounds, expected $NumRounds. " +
        "Stopping before Split 3. Check Flower/Ray logs for runtime interruption."
    )
}

Write-Host "[Enterprise] Running Split 3 Fabric audit..." -ForegroundColor Cyan
if ($ReplayTrustLog) {
    Write-Host "[Enterprise] Split 3 mode: REPLAY trust log commits (may conflict if rounds already on-chain)" -ForegroundColor Yellow
    & $PythonExe module1/split3/split3_main.py `
        --trust_log logs_split2/trust_training_log.json `
        --blockchain fabric `
        --output_dir governance_output
} else {
    Write-Host "[Enterprise] Split 3 mode: READ-ONLY blockchain attestation audit" -ForegroundColor Green
    & $PythonExe module1/split3/split3_main.py `
        --blockchain fabric `
        --audit_chain `
        --output_dir governance_output
}
if ($LASTEXITCODE -ne 0) {
    throw "Split 3 Fabric audit failed in enterprise mode."
}

Write-Host "`n[Enterprise] Completed successfully." -ForegroundColor Green
Write-Host "Reports: governance_output/governance_report.json and governance_output/hash_chain.json" -ForegroundColor Green
