param(
    [int]$NumClients = 5,
    [int]$NumRounds = 20,
    [string]$Model = "dnn",
    [ValidateSet("true", "false")]
    [string]$UseSynthetic = "true",
    [string]$Attack = "none",
    [string]$Malicious = "1",
    [ValidateSet("true", "false")]
    [string]$BlockchainEnabled = "false",
    [string]$BlockchainBackend = "simulation"
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

# Keep local Flower simulation topology aligned with requested NumClients.
$flwrConfigPath = Join-Path $env:USERPROFILE ".flwr\config.toml"
if (Test-Path $flwrConfigPath) {
    $cfg = Get-Content $flwrConfigPath -Raw
    if ($cfg -match "options\.num-supernodes\s*=") {
        $cfg = [regex]::Replace($cfg, "options\.num-supernodes\s*=\s*\d+", "options.num-supernodes = $NumClients")
    } elseif ($cfg -match "\[superlink\.local\]") {
        $cfg = [regex]::Replace($cfg, "\[superlink\.local\]\s*", "[superlink.local]`r`noptions.num-supernodes = $NumClients`r`n", 1)
    }
    Set-Content -Path $flwrConfigPath -Value $cfg -Encoding Ascii
}

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

& $FlwrExe run . --run-config $runConfig --stream
