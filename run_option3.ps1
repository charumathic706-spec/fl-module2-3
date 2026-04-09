param(
    [int]$NumClients = 2,
    [int]$NumRounds = 20,
    [ValidateSet("dnn", "logistic")]
    [string]$Model = "logistic",
    [switch]$SkipFabricSetup,
    [switch]$SkipDashboard,
    [int]$DashboardPort = 5000
)

$scriptPath = Join-Path $PSScriptRoot "module1\split3\hlf\run_option3.ps1"
$forwardArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "$scriptPath",
    "-NumClients", "$NumClients",
    "-NumRounds", "$NumRounds",
    "-Model", "$Model",
    "-DashboardPort", "$DashboardPort"
)
if ($SkipFabricSetup) { $forwardArgs += "-SkipFabricSetup" }
if ($SkipDashboard) { $forwardArgs += "-SkipDashboard" }

$proc = Start-Process -FilePath "powershell" -ArgumentList $forwardArgs -Wait -NoNewWindow -PassThru
if ($proc.ExitCode -ne 0) { exit $proc.ExitCode }
