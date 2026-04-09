param(
    [int]$NumClients = 2,
    [int]$NumRounds = 20,
    [ValidateSet("dnn", "logistic")]
    [string]$Model = "logistic",
    [ValidateSet("raft", "bft")]
    [string]$Consensus = "bft",
    [switch]$Reset,
    [switch]$SkipFabricSetup,
    [switch]$SkipDashboard,
    [int]$DashboardPort = 5000
)

$scriptPath = Join-Path $PSScriptRoot "module1\split3\hlf_enterprise\run_option3_enterprise.ps1"
$forwardArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "$scriptPath",
    "-NumClients", "$NumClients",
    "-NumRounds", "$NumRounds",
    "-Model", "$Model",
    "-Consensus", "$Consensus",
    "-DashboardPort", "$DashboardPort"
)
if ($Reset) { $forwardArgs += "-Reset" }
if ($SkipFabricSetup) { $forwardArgs += "-SkipFabricSetup" }
if ($SkipDashboard) { $forwardArgs += "-SkipDashboard" }

$proc = Start-Process -FilePath "powershell" -ArgumentList $forwardArgs -Wait -NoNewWindow -PassThru
if ($proc.ExitCode -ne 0) { exit $proc.ExitCode }
