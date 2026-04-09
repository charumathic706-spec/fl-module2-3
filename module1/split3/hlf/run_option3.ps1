param(
    [int]$NumClients = 2,
    [int]$NumRounds = 20,
    [ValidateSet("dnn", "logistic")]
    [string]$Model = "logistic",
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
$FabricEnvFile = Join-Path $PSScriptRoot "fabric_connection.env"

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

if (-not $SkipFabricSetup) {
    $wslScriptDir = Convert-ToWslPath $PSScriptRoot
    Write-Host "[Option3] Setting up Hyperledger Fabric network (this can take several minutes)..." -ForegroundColor Cyan
    & wsl bash -lc "cd '$wslScriptDir' ; chmod +x setup.sh teardown.sh ; set -o pipefail ; ./setup.sh 2>&1 | tee setup_run.log"
    if ($LASTEXITCODE -ne 0) {
        throw "Fabric setup failed. Check module1/split3/hlf/setup.sh output."
    }
}

if (-not (Test-Path $FabricEnvFile)) {
    if ($SkipFabricSetup) {
        throw "Fabric connection file missing: $FabricEnvFile. Run without -SkipFabricSetup at least once."
    }
    throw "Fabric setup did not produce $FabricEnvFile. Check setup.sh logs and rerun."
}

if (-not $SkipDashboard) {
    Write-Host "[Option3] Starting dashboard at http://127.0.0.1:$DashboardPort" -ForegroundColor Cyan
    Start-Process -FilePath $PythonExe -ArgumentList @(
        "-m", "module1.dashboard_server",
        "--log", "logs_split2/trust_training_log.json",
        "--host", "127.0.0.1",
        "--port", "$DashboardPort",
        "--expected_rounds", "$NumRounds"
    ) | Out-Null
}

Write-Host "[Option3] Running Split 2 with live Fabric governance..." -ForegroundColor Cyan
& powershell -ExecutionPolicy Bypass -File $Split2Runner `
    -NumClients $NumClients `
    -NumRounds $NumRounds `
    -Model $Model `
    -BlockchainEnabled "true" `
    -BlockchainBackend "fabric"
if ($LASTEXITCODE -ne 0) {
    throw "Split 2 run failed."
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

Write-Host "[Option3] Running Split 3 Fabric audit..." -ForegroundColor Cyan
if ($ReplayTrustLog) {
    Write-Host "[Option3] Split 3 mode: REPLAY trust log commits (may conflict if rounds already on-chain)" -ForegroundColor Yellow
    & $PythonExe module1/split3/split3_main.py `
        --trust_log logs_split2/trust_training_log.json `
        --blockchain fabric `
        --output_dir governance_output
} else {
    Write-Host "[Option3] Split 3 mode: READ-ONLY blockchain attestation audit" -ForegroundColor Green
    & $PythonExe module1/split3/split3_main.py `
        --blockchain fabric `
        --audit_chain `
        --output_dir governance_output
}
if ($LASTEXITCODE -ne 0) {
    throw "Split 3 Fabric audit failed."
}

Write-Host "`n[Option3] Completed successfully." -ForegroundColor Green
Write-Host "Reports: governance_output/governance_report.json and governance_output/hash_chain.json" -ForegroundColor Green
