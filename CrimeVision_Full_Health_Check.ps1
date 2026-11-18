<#
.SYNOPSIS
  Non-destructive diagnostic health check for the CrimeVision AI repository.

.DESCRIPTION
  Collects project structure, Git state, frontend tests/build and available lint,
  Python dependency/syntax/test results, Docker/Compose state and recent logs,
  local HTTP/TCP checks, model-artifact/configuration presence, and Windows
  resource information into one timestamped text report, then opens it in
  Notepad.

  This script DOES NOT install packages, rebuild Docker images, restart/stop
  containers, modify .env files, upload files, or submit an evidence-analysis
  request. The Next.js production build refreshes frontend-web/.next, and test
  tools may create their normal temporary/cache output. It does not prove
  model accuracy or forensic validity.

.USAGE
  Run from the repository root:
    powershell -ExecutionPolicy Bypass -File .\CrimeVision_Full_Health_Check.ps1

  Or specify the repository path:
    powershell -ExecutionPolicy Bypass -File .\CrimeVision_Full_Health_Check.ps1 -ProjectRoot "C:\path\to\CrimeVision-AI"

  Add -RunSecurityAudits to run npm audit and pip-audit (network access may be
  required; audit findings are reported as warnings, not test failures).
#>

[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [switch]$RunSecurityAudits
)

$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
$script:PassCount = 0
$script:WarnCount = 0
$script:FailCount = 0
$script:InfoCount = 0
$script:ReportPath = $null

function Write-Section {
    param([Parameter(Mandatory = $true)][string]$Name)
    Write-Host "`r`n"
    Write-Host ('=' * 88)
    Write-Host (" {0}" -f $Name)
    Write-Host ('=' * 88)
}

function Write-Check {
    param(
        [ValidateSet('PASS','WARN','FAIL','INFO')][string]$Status,
        [Parameter(Mandatory = $true)][string]$Message
    )
    switch ($Status) {
        'PASS' { $script:PassCount++ ; Write-Host "[PASS] $Message" }
        'WARN' { $script:WarnCount++ ; Write-Host "[WARN] $Message" }
        'FAIL' { $script:FailCount++ ; Write-Host "[FAIL] $Message" }
        'INFO' { $script:InfoCount++ ; Write-Host "[INFO] $Message" }
    }
}

function Protect-Text {
    param([AllowNull()][string]$Text)
    if ($null -eq $Text) { return '' }
    $safe = $Text
    # Redact common key/value secret patterns in command output and logs.
    $safe = [regex]::Replace($safe, '(?i)(password|passwd|token|secret|api[_-]?key|authorization|client_secret)(\s*[=:]\s*)("[^"]*"|''[^'']*''|[^\s,;]+)', '$1$2<REDACTED>')
    # Redact credentials embedded in URL authority sections.
    $safe = [regex]::Replace($safe, '(?i)(://[^:/\s]+:)[^@/\s]+(@)', '$1<REDACTED>$2')
    return $safe
}

function Invoke-Section {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )
    Write-Section $Name
    try { & $Action }
    catch { Write-Check 'FAIL' ("Section error: " + (Protect-Text $_.Exception.Message)) }
}

function Invoke-NativeCheck {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Command,
        [string[]]$Arguments = @(),
        [string]$Directory = $script:Root,
        [ValidateSet('PASS','WARN','FAIL')][string]$NonZeroStatus = 'FAIL'
    )
    Invoke-Section $Name {
        $commandInfo = Get-Command $Command -ErrorAction SilentlyContinue
        if (-not $commandInfo) {
            Write-Check 'WARN' "$Command is not installed or is not available in PATH."
            return
        }
        $oldLocation = Get-Location
        try {
            if ($Directory -and (Test-Path -LiteralPath $Directory)) {
                Set-Location -LiteralPath $Directory
            }
            Write-Host (Protect-Text ("COMMAND: {0} {1}" -f $Command, ($Arguments -join ' ')))
            $nativeOutput = @(& $Command @Arguments 2>&1)
            $exitCode = $LASTEXITCODE
            foreach ($item in $nativeOutput) {
                Write-Host (Protect-Text ([string]$item))
            }
            if ($null -eq $exitCode) { $exitCode = 0 }
            Write-Host "EXIT CODE: $exitCode"
            if ($exitCode -eq 0) { Write-Check 'PASS' "$Name completed successfully." }
            else { Write-Check $NonZeroStatus "$Name returned exit code $exitCode." }
        }
        catch {
            Write-Check 'FAIL' ("Could not run $Name : " + (Protect-Text $_.Exception.Message))
        }
        finally { Set-Location $oldLocation }
    }
}

function Get-EnvValue {
    param([string]$Name, [string]$FilePath)
    # Environment variables passed to this PowerShell process have precedence.
    $processValue = [Environment]::GetEnvironmentVariable($Name, 'Process')
    if (-not [string]::IsNullOrWhiteSpace($processValue)) { return $processValue.Trim().Trim('"').Trim("'") }
    if (-not (Test-Path -LiteralPath $FilePath)) { return '' }
    foreach ($line in (Get-Content -LiteralPath $FilePath -ErrorAction SilentlyContinue)) {
        if ($line -match ('^\s*' + [regex]::Escape($Name) + '\s*=\s*(.*?)\s*$')) {
            return $Matches[1].Trim().Trim('"').Trim("'")
        }
    }
    return ''
}

function Test-TcpPort {
    param([string]$HostName = '127.0.0.1', [int]$Port)
    try {
        $result = Test-NetConnection -ComputerName $HostName -Port $Port -WarningAction SilentlyContinue
        if ($result.TcpTestSucceeded) { Write-Check 'PASS' "$HostName`:$Port accepts TCP connections." }
        else { Write-Check 'WARN' "$HostName`:$Port is not accepting TCP connections." }
    }
    catch { Write-Check 'WARN' "TCP test for $HostName`:$Port could not run: $($_.Exception.Message)" }
}

function Test-HttpEndpoint {
    param([string]$Url, [int]$TimeoutSeconds = 8)
    Write-Host "`nURL: $Url"
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSeconds -ErrorAction Stop
        $statusCode = [int]$response.StatusCode
        Write-Host "HTTP STATUS: $statusCode"
        Write-Host "CONTENT TYPE: $($response.Headers['Content-Type'])"
        if ($response.Content) {
            $preview = [string]$response.Content
            if ($preview.Length -gt 1200) { $preview = $preview.Substring(0,1200) + '...[preview truncated]' }
            Write-Host ('BODY PREVIEW: ' + (Protect-Text $preview))
        }
        if ($statusCode -ge 200 -and $statusCode -lt 400) { Write-Check 'PASS' "$Url responded with HTTP $statusCode." }
        else { Write-Check 'WARN' "$Url responded with HTTP $statusCode." }
    }
    catch {
        $statusText = 'NO RESPONSE'
        try { if ($_.Exception.Response) { $statusText = 'HTTP ' + [int]$_.Exception.Response.StatusCode } } catch {}
        Write-Host "RESULT: $statusText"
        Write-Host ('DETAIL: ' + (Protect-Text $_.Exception.Message))
        Write-Check 'WARN' "$Url did not return a successful HTTP response ($statusText)."
    }
}

# Discover repository root from the supplied folder or its parents.
$startPath = (Resolve-Path -LiteralPath $ProjectRoot -ErrorAction SilentlyContinue).Path
if (-not $startPath) { $startPath = (Get-Location).Path }
$probe = Get-Item -LiteralPath $startPath -ErrorAction SilentlyContinue
if ($probe -and -not $probe.PSIsContainer) { $probe = $probe.Directory }
$foundRoot = $null
while ($probe) {
    $candidate = $probe.FullName
    if ((Test-Path (Join-Path $candidate 'docker-compose.yml')) -and
        (Test-Path (Join-Path $candidate 'research-engine\main.py')) -and
        (Test-Path (Join-Path $candidate 'frontend-web\package.json'))) {
        $foundRoot = $candidate
        break
    }
    $probe = $probe.Parent
}
if ($foundRoot) { $script:Root = $foundRoot }
else { $script:Root = $startPath }

$timestamp = Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'
$script:ReportPath = Join-Path $script:Root ("CrimeVision_Full_Health_Report_{0}.txt" -f $timestamp)

try {
    Start-Transcript -Path $script:ReportPath -Force | Out-Null

    Write-Host 'CRIMEVISION AI - COMPREHENSIVE PROJECT HEALTH REPORT'
    Write-Host ('Generated: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'))
    Write-Host ('Repository root: ' + $script:Root)
    Write-Host ('Report file: ' + $script:ReportPath)
    Write-Host 'Safety: no package installation, Docker image rebuild, container restart, or evidence submission. The frontend build refreshes .next.'

    # 1. Project layout / important paths
    Invoke-Section '1. PROJECT STRUCTURE AND EXPECTED FILES' {
        if (Test-Path (Join-Path $script:Root 'docker-compose.yml')) { Write-Check 'PASS' 'docker-compose.yml found.' } else { Write-Check 'FAIL' 'docker-compose.yml missing.' }
        $expected = @(
            '.env.example', '.env', 'docker-compose.yml', '.github\workflows\ci.yml',
            'frontend-web\package.json', 'frontend-web\package-lock.json', 'frontend-web\Dockerfile',
            'frontend-web\src', 'frontend-web\tests', 'frontend-web\vitest.config.js',
            'research-engine\main.py', 'research-engine\Dockerfile', 'research-engine\requirements.txt',
            'research-engine\requirements.in', 'research-engine\src\inference.py',
            'research-engine\src\conformal.py', 'research-engine\src\evidential_fusion.py',
            'research-engine\tests', 'research-engine\train_evidential_head.py',
            'research-engine\calibrate_conformal.py', 'research-engine\reports\group_v3\ablation_artifacts\all_three.json'
        )
        foreach ($relative in $expected) {
            if (Test-Path -LiteralPath (Join-Path $script:Root $relative)) { Write-Check 'PASS' "$relative exists." }
            else { Write-Check 'WARN' "$relative not found." }
        }
        if (-not (Test-Path (Join-Path $script:Root 'backend'))) { Write-Check 'INFO' 'No separate backend/ directory; the FastAPI backend is research-engine/ in this repository.' }
        if (-not (Test-Path (Join-Path $script:Root 'frontend'))) { Write-Check 'INFO' 'No frontend/ directory; the Next.js app is frontend-web/ in this repository.' }
        Write-Host "`nTop-level entries:"
        Get-ChildItem -LiteralPath $script:Root -Force -ErrorAction SilentlyContinue |
            Select-Object Mode, Name, Length, LastWriteTime |
            Format-Table -AutoSize | Out-String | Write-Host
    }

    # 2. Runtime / host environment
    Invoke-Section '2. WINDOWS / POWERSHELL / RUNTIME VERSIONS' {
        Write-Host "Windows: $([Environment]::OSVersion.VersionString)"
        Write-Host "PowerShell: $($PSVersionTable.PSVersion) ($($PSVersionTable.PSEdition))"
        Write-Host "Current directory: $((Get-Location).Path)"
        foreach ($cmd in @('git','docker','node','npm','python','py','wsl')) {
            $found = Get-Command $cmd -ErrorAction SilentlyContinue
            if ($found) { Write-Check 'PASS' "$cmd found at $($found.Source)" }
            else { Write-Check 'WARN' "$cmd not found in PATH." }
        }
        if (Get-Command node -ErrorAction SilentlyContinue) { Invoke-NativeCheck 'NODE VERSION' 'node' @('--version') $script:Root }
        if (Get-Command npm -ErrorAction SilentlyContinue) { Invoke-NativeCheck 'NPM VERSION' 'npm' @('--version') $script:Root }
        if (Get-Command python -ErrorAction SilentlyContinue) { Invoke-NativeCheck 'PYTHON VERSION' 'python' @('--version') $script:Root }
        if (Get-Command py -ErrorAction SilentlyContinue) { Invoke-NativeCheck 'PY LAUNCHER VERSION' 'py' @('--version') $script:Root }
    }

    # 3. Git health
    Invoke-NativeCheck '3A. GIT STATUS' 'git' @('status','--short','--branch') $script:Root
    Invoke-NativeCheck '3B. RECENT GIT COMMITS' 'git' @('log','-5','--oneline','--decorate') $script:Root
    Invoke-NativeCheck '3C. WHITESPACE / PATCH VALIDATION' 'git' @('diff','--check') $script:Root 'WARN'
    Invoke-Section '3D. GIT CHANGE SUMMARY AND SECRET-FILE TRACKING' {
        if (Get-Command git -ErrorAction SilentlyContinue) {
            $stat = @(& git -C $script:Root diff --stat 2>&1)
            foreach ($line in $stat) { Write-Host (Protect-Text ([string]$line)) }
            $trackedSensitive = @(& git -C $script:Root ls-files '.env' 'frontend-web/.env.local' 'research-engine/.env' 2>&1)
            if ($trackedSensitive.Count -gt 0 -and -not [string]::IsNullOrWhiteSpace([string]$trackedSensitive[0])) {
                Write-Check 'WARN' ('Potentially sensitive config file(s) are tracked by Git: ' + ($trackedSensitive -join ', '))
            } else { Write-Check 'PASS' 'No root .env / frontend-web/.env.local / research-engine/.env file is listed as a tracked Git file.' }
            $ignored = @(& git -C $script:Root check-ignore -v '.env' 'frontend-web/.env.local' 'research-engine/.env' 2>&1)
            if ($ignored.Count -gt 0) { Write-Host 'Ignore rules (paths only):'; $ignored | ForEach-Object { Write-Host (Protect-Text ([string]$_)) } }
        } else { Write-Check 'WARN' 'Git is unavailable; Git integrity checks skipped.' }
    }

    # 4. Environment configuration: never print .env contents or secrets.
    Invoke-Section '4. ENVIRONMENT CONFIGURATION (SECRET VALUES ARE NOT DISPLAYED)' {
        $rootEnv = Join-Path $script:Root '.env'
        $exampleEnv = Join-Path $script:Root '.env.example'
        if (Test-Path $rootEnv) { Write-Check 'PASS' 'Root .env exists. Its contents are intentionally not printed.' }
        else { Write-Check 'WARN' 'Root .env does not exist. Docker Compose may require values supplied externally.' }
        if (Test-Path $exampleEnv) { Write-Check 'PASS' '.env.example exists.' } else { Write-Check 'WARN' '.env.example missing.' }
        $decisionKeys = @('EVIDENTIAL_HEAD_PATH','EVIDENTIAL_CALIBRATION_PATH','CONFORMAL_ARTIFACT_PATH')
        foreach ($key in $decisionKeys) {
            $value = Get-EnvValue -Name $key -FilePath $rootEnv
            if ([string]::IsNullOrWhiteSpace($value)) {
                Write-Check 'WARN' "$key is EMPTY in current process/root .env. Empty decision-artifact configuration can cause INDETERMINATE-only behaviour."
            } else {
                Write-Host "$key = $value"
                if ([System.IO.Path]::IsPathRooted($value)) { $candidatePath = $value }
                else { $candidatePath = Join-Path $script:Root $value }
                if (Test-Path -LiteralPath $candidatePath) { Write-Check 'PASS' "$key points to an existing host-side file/path." }
                else { Write-Check 'WARN' "$key is configured, but this path is not found on the host. It may be a container-only path or named-volume path." }
            }
        }
        foreach ($key in @('DEFAULT_DEVICE','CLIP_MODEL_NAME','CLIP_MODEL_REVISION','TRUST_MODE')) {
            $value = Get-EnvValue -Name $key -FilePath $rootEnv
            if ($value) { Write-Host "$key = $value" } else { Write-Host "$key = <unset; application default may apply>" }
        }
        foreach ($relative in @('frontend-web\.env.local','research-engine\.env','research-engine\.env.example')) {
            if (Test-Path (Join-Path $script:Root $relative)) { Write-Check 'INFO' "$relative exists; contents not displayed." }
        }
    }

    # 5. Frontend package/testing/build checks
    $frontendDir = Join-Path $script:Root 'frontend-web'
    if (Test-Path (Join-Path $frontendDir 'package.json')) {
        Invoke-Section '5A. FRONTEND PACKAGE METADATA' {
            $package = Get-Content (Join-Path $frontendDir 'package.json') -Raw | ConvertFrom-Json
            Write-Host "Package: $($package.name)"
            Write-Host "Version: $($package.version)"
            Write-Host 'Scripts:'
            if ($package.scripts) { $package.scripts.PSObject.Properties | ForEach-Object { Write-Host ("  {0}: {1}" -f $_.Name,$_.Value) } }
            if (Test-Path (Join-Path $frontendDir 'node_modules')) { Write-Check 'PASS' 'frontend-web/node_modules exists.' }
            else { Write-Check 'WARN' 'frontend-web/node_modules missing. No package installation will be performed by this script.' }
        }
        Invoke-NativeCheck '5B. FRONTEND DEPENDENCY TREE' 'npm' @('ls','--depth=0') $frontendDir 'WARN'
        Invoke-NativeCheck '5C. FRONTEND UNIT TESTS (VITEST)' 'npm' @('test','--','--reporter=verbose') $frontendDir
        Invoke-NativeCheck '5D. FRONTEND PRODUCTION BUILD' 'npm' @('run','build') $frontendDir
        $eslintConfig = @('.eslintrc','.eslintrc.js','.eslintrc.cjs','.eslintrc.json','eslint.config.js','eslint.config.mjs') | Where-Object { Test-Path (Join-Path $frontendDir $_) }
        if ($eslintConfig -or (Test-Path (Join-Path $frontendDir 'node_modules\.bin\eslint.cmd'))) {
            Invoke-NativeCheck '5E. FRONTEND LINT SCRIPT' 'npm' @('run','lint') $frontendDir 'WARN'
        } else {
            Invoke-Section '5E. FRONTEND LINT' { Write-Check 'INFO' 'No ESLint configuration/binary detected in the project snapshot; skipped to avoid Next.js interactive setup prompts.' }
        }
        if ($RunSecurityAudits) {
            Invoke-NativeCheck '5F. NPM SECURITY AUDIT (ALL DEPENDENCIES)' 'npm' @('audit') $frontendDir 'WARN'
        } else { Invoke-Section '5F. NPM SECURITY AUDIT' { Write-Check 'INFO' 'Skipped by default; rerun with -RunSecurityAudits to query npm advisories (network required).' } }
    } else {
        Invoke-Section '5. FRONTEND' { Write-Check 'FAIL' 'frontend-web/package.json not found; frontend tests/build skipped.' }
    }

    # 6. Research-engine Python checks
    $researchDir = Join-Path $script:Root 'research-engine'
    $pythonExe = $null
    foreach ($candidate in @(
        (Join-Path $researchDir '.venv\Scripts\python.exe'),
        (Join-Path $researchDir 'venv\Scripts\python.exe'),
        (Join-Path $script:Root '.venv\Scripts\python.exe'),
        (Join-Path $script:Root 'venv\Scripts\python.exe')
    )) {
        if (Test-Path $candidate) { $pythonExe = $candidate; break }
    }
    if (-not $pythonExe) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCommand) { $pythonExe = 'python' }
        elseif (Get-Command py -ErrorAction SilentlyContinue) { $pythonExe = 'py' }
    }
    if (Test-Path (Join-Path $researchDir 'main.py')) {
        Invoke-Section '6A. RESEARCH ENGINE PYTHON INTERPRETER' {
            if ($pythonExe) { Write-Host "Selected Python: $pythonExe"; Write-Check 'PASS' 'Python interpreter found.' }
            else { Write-Check 'FAIL' 'Python interpreter not found; Python tests skipped.' }
        }
        if ($pythonExe) {
            Invoke-NativeCheck '6B. RESEARCH ENGINE PYTHON VERSION' $pythonExe @('--version') $researchDir
            Invoke-NativeCheck '6C. RESEARCH ENGINE PIP DEPENDENCY CONSISTENCY' $pythonExe @('-m','pip','check') $researchDir 'WARN'
            Invoke-Section '6D. RESEARCH ENGINE SOURCE SYNTAX CHECK (AST; NO BYTECODE FILES WRITTEN)' {
                $syntaxCode = [string]::Join([Environment]::NewLine, @(
                    'import ast, pathlib, sys',
                    'skip={".venv","venv","__pycache__","node_modules",".pytest_cache",".git"}',
                    'files=[p for p in pathlib.Path(".").rglob("*.py") if not any(x in skip for x in p.parts)]',
                    'errors=[]',
                    'for p in files:',
                    ' try: ast.parse(p.read_text(encoding="utf-8"), filename=str(p))',
                    ' except Exception as e: errors.append((str(p), str(e)))',
                    'print(f"Parsed {len(files)} Python files")',
                    'for p,e in errors: print(f"SYNTAX ERROR: {p}: {e}")',
                    'sys.exit(1 if errors else 0)'
                ))
                Invoke-NativeCheck 'PYTHON AST SYNTAX CHECK' $pythonExe @('-c',$syntaxCode) $researchDir
            }
            Invoke-NativeCheck '6E. RESEARCH ENGINE FULL PYTEST SUITE' $pythonExe @('-m','pytest','-v','--tb=short','--durations=15') $researchDir
            Invoke-Section '6F. KEY PYTHON LIBRARY IMPORT / VERSION PROBE' {
                $probeCode = [string]::Join([Environment]::NewLine, @(
                    'import importlib, importlib.metadata as m',
                    "names=['torch','torchvision','transformers','numpy','scipy','cv2','Pillow','fastapi','uvicorn','python-multipart','neo4j','python-dotenv']",
                    "aliases={'cv2':'cv2','Pillow':'PIL','python-multipart':'multipart','python-dotenv':'dotenv'}",
                    'for n in names:',
                    ' try:',
                    '  mod=importlib.import_module(aliases.get(n,n))',
                    "  v=getattr(mod,'__version__',None) or m.version(n)",
                    "  print(f'{n}: OK ({v})')",
                    ' except Exception as e:',
                    "  print(f'{n}: IMPORT ERROR ({type(e).__name__}: {e})')"
                ))
                Invoke-NativeCheck 'PYTHON IMPORT PROBE' $pythonExe @('-c',$probeCode) $researchDir 'WARN'
            }
            if ($RunSecurityAudits) {
                if (Get-Command pip-audit -ErrorAction SilentlyContinue) {
                    Invoke-NativeCheck '6G. PIP-AUDIT' 'pip-audit' @('-r','requirements.in') $researchDir 'WARN'
                } else { Invoke-Section '6G. PIP-AUDIT' { Write-Check 'INFO' 'pip-audit is not installed; no tool was installed automatically.' } }
            } else { Invoke-Section '6G. PIP SECURITY AUDIT' { Write-Check 'INFO' 'Skipped by default; rerun with -RunSecurityAudits if pip-audit is installed.' } }
        }
    } else { Invoke-Section '6. RESEARCH ENGINE' { Write-Check 'FAIL' 'research-engine/main.py not found.' } }

    # 7. Research scripts/artifacts/routes inventory
    Invoke-Section '7. RESEARCH SCRIPTS, TESTS, ARTIFACTS AND API ROUTES' {
        foreach ($folder in @('research-engine\tests','research-engine\src','research-engine\models','research-engine\reports\group_v3\ablation_artifacts','research-engine\datasets')) {
            $path = Join-Path $script:Root $folder
            if (Test-Path $path) {
                $items = @(Get-ChildItem -LiteralPath $path -Recurse -File -ErrorAction SilentlyContinue)
                $totalBytes = ($items | Measure-Object -Property Length -Sum).Sum
                if ($null -eq $totalBytes) { $totalBytes = 0 }
                Write-Host ("{0}: {1} files, {2:N2} MB" -f $folder,$items.Count,($totalBytes/1MB))
                $items | Sort-Object FullName | Select-Object FullName,Length,LastWriteTime | Format-Table -AutoSize | Out-String | Write-Host
            } else { Write-Check 'INFO' "$folder not present." }
        }
        $mainPath = Join-Path $script:Root 'research-engine\main.py'
        if (Test-Path $mainPath) {
            Write-Host "`nFastAPI route decorators found in main.py:"
            Select-String -Path $mainPath -Pattern '@app\.(get|post|put|delete|patch)' | ForEach-Object { Write-Host ("Line {0}: {1}" -f $_.LineNumber,$_.Line.Trim()) }
        }
        $artifactPath = Join-Path $script:Root 'research-engine\reports\group_v3\ablation_artifacts\all_three.json'
        if (Test-Path $artifactPath) {
            try {
                $artifact = Get-Content -LiteralPath $artifactPath -Raw | ConvertFrom-Json
                Write-Check 'PASS' 'all_three.json exists and parses as JSON.'
                Write-Host ('Artifact top-level keys: ' + (($artifact.PSObject.Properties.Name) -join ', '))
            } catch { Write-Check 'FAIL' ('all_three.json exists but JSON parsing failed: ' + $_.Exception.Message) }
        }
        $hfCacheCandidates = @()
        if ($env:HF_HOME) { $hfCacheCandidates += $env:HF_HOME }
        if ($env:HUGGINGFACE_HUB_CACHE) { $hfCacheCandidates += $env:HUGGINGFACE_HUB_CACHE }
        if ($env:USERPROFILE) { $hfCacheCandidates += (Join-Path $env:USERPROFILE '.cache\huggingface\hub') }
        $hfCacheCandidates = @($hfCacheCandidates | Select-Object -Unique)
        foreach ($cachePath in $hfCacheCandidates) {
            if (Test-Path $cachePath) {
                $clipCachePath = Join-Path $cachePath 'models--openai--clip-vit-base-patch32'
                if (Test-Path $clipCachePath) { Write-Check 'PASS' "CLIP model cache directory found: $clipCachePath" }
                else { Write-Check 'INFO' "Hugging Face cache exists, but CLIP snapshot folder was not found at $clipCachePath" }
            } else { Write-Check 'INFO' "Hugging Face cache path not found: $cachePath" }
        }
    }

    # 8. Docker health / Compose state
    Invoke-NativeCheck '8A. DOCKER CLIENT VERSION' 'docker' @('--version') $script:Root 'WARN'
    Invoke-NativeCheck '8B. DOCKER COMPOSE VERSION' 'docker' @('compose','version') $script:Root 'WARN'
    Invoke-NativeCheck '8C. DOCKER SERVER INFO (SUMMARY)' 'docker' @('info','--format','Server={{.ServerVersion}} OS={{.OSType}} Arch={{.Architecture}} Containers={{.Containers}} Running={{.ContainersRunning}} Images={{.Images}}') $script:Root 'WARN'
    Invoke-NativeCheck '8D. COMPOSE CONFIG VALIDATION (QUIET; DOES NOT PRINT SECRETS)' 'docker' @('compose','config','--quiet') $script:Root 'WARN'
    Invoke-NativeCheck '8E. COMPOSE SERVICE LIST' 'docker' @('compose','config','--services') $script:Root 'WARN'
    Invoke-NativeCheck '8F. COMPOSE CONTAINER STATUS' 'docker' @('compose','ps','-a') $script:Root 'WARN'
    Invoke-NativeCheck '8G. COMPOSE IMAGES' 'docker' @('compose','images') $script:Root 'WARN'
    Invoke-NativeCheck '8H. DOCKER DISK USAGE' 'docker' @('system','df') $script:Root 'WARN'
    Invoke-Section '8I. DOCKER CONTAINER HEALTH DETAILS (NO ENVIRONMENT VARIABLES)' {
        if (Get-Command docker -ErrorAction SilentlyContinue) {
            foreach ($containerName in @('crimevision-neo4j','crimevision-research-engine','crimevision-frontend')) {
                $format = '{{.Name}} | status={{.State.Status}} | restarts={{.RestartCount}} | health={{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}} | ports={{json .NetworkSettings.Ports}}'
                $result = @(& docker inspect --format $format $containerName 2>&1)
                if ($LASTEXITCODE -eq 0) { $result | ForEach-Object { Write-Host (Protect-Text ([string]$_)) } }
                else { Write-Check 'WARN' "$containerName is not available for inspect (container may not exist)." }
            }
        } else { Write-Check 'WARN' 'Docker CLI unavailable.' }
    }
    Invoke-NativeCheck '8J. DOCKER NETWORK LIST' 'docker' @('network','ls') $script:Root 'WARN'
    Invoke-NativeCheck '8K. DOCKER VOLUME LIST' 'docker' @('volume','ls') $script:Root 'WARN'

    # 9. Container runtime artifact env check: reports only three decision paths.
    Invoke-Section '9. EFFECTIVE DECISION-ARTIFACT SETTINGS INSIDE RUNNING RESEARCH CONTAINER' {
        if (Get-Command docker -ErrorAction SilentlyContinue) {
            $containerStatus = @(& docker inspect --format '{{.State.Status}}' 'crimevision-research-engine' 2>$null)
            if ($LASTEXITCODE -eq 0 -and $containerStatus -contains 'running') {
                $containerProbe = [string]::Join([Environment]::NewLine, @(
                    'import os,pathlib',
                    "keys=['EVIDENTIAL_HEAD_PATH','EVIDENTIAL_CALIBRATION_PATH','CONFORMAL_ARTIFACT_PATH']",
                    'for k in keys:',
                    " v=os.getenv(k,'')",
                    " print(k+'='+('EMPTY' if not v else v))",
                    " print('  exists_in_container='+str(pathlib.Path(v).exists()) if v else '  exists_in_container=NO_PATH_CONFIGURED')"
                ))
                Invoke-NativeCheck 'CONTAINER DECISION-ARTIFACT ENV / FILE CHECK' 'docker' @('exec','crimevision-research-engine','python','-c',$containerProbe) $script:Root 'WARN'
                $hfContainerProbe = [string]::Join([Environment]::NewLine, @(
                    'import os,pathlib',
                    "home=pathlib.Path.home()",
                    "candidates=[os.getenv('HF_HOME'),os.getenv('HUGGINGFACE_HUB_CACHE'),str(home/'.cache'/'huggingface'/'hub')]",
                    'for raw in candidates:',
                    ' if raw:',
                    '  p=pathlib.Path(raw)',
                    "  print(str(p)+' exists='+str(p.exists())+' clip_cached='+str((p/'models--openai--clip-vit-base-patch32').exists()))"
                ))
                Invoke-NativeCheck 'CONTAINER HUGGING FACE CLIP CACHE CHECK' 'docker' @('exec','crimevision-research-engine','python','-c',$hfContainerProbe) $script:Root 'WARN'
            } else { Write-Check 'WARN' 'Research-engine container is not running; effective container environment cannot be checked.' }
        } else { Write-Check 'WARN' 'Docker CLI unavailable.' }
    }

    # 10. Logs. Keep tail bounded; redact common secret patterns.
    Invoke-NativeCheck '10A. ALL COMPOSE SERVICE LOGS (LAST 300 LINES; REDACTION APPLIED)' 'docker' @('compose','logs','--no-color','--tail=300') $script:Root 'WARN'
    Invoke-NativeCheck '10B. RESEARCH-ENGINE LOGS (LAST 500 LINES; REDACTION APPLIED)' 'docker' @('compose','logs','--no-color','--tail=500','research-engine') $script:Root 'WARN'
    Invoke-NativeCheck '10C. FRONTEND LOGS (LAST 250 LINES; REDACTION APPLIED)' 'docker' @('compose','logs','--no-color','--tail=250','frontend-web') $script:Root 'WARN'
    Invoke-NativeCheck '10D. NEO4J LOGS (LAST 250 LINES; REDACTION APPLIED)' 'docker' @('compose','logs','--no-color','--tail=250','neo4j') $script:Root 'WARN'

    # 11. Host resource and process checks
    Invoke-Section '11. WINDOWS RESOURCE / DOCKER DESKTOP PROCESS CHECK' {
        try {
            $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
            $totalGB = [math]::Round($os.TotalVisibleMemorySize/1MB,2)
            $freeGB = [math]::Round($os.FreePhysicalMemory/1MB,2)
            Write-Host "Physical memory total: $totalGB GB"
            Write-Host "Physical memory free:  $freeGB GB"
            if ($freeGB -lt 2) { Write-Check 'WARN' 'Less than 2 GB physical memory is currently free.' } else { Write-Check 'PASS' 'Physical memory is not critically low at the time of this report.' }
        } catch { Write-Check 'WARN' 'Could not read Windows memory statistics.' }
        try {
            $systemDrive = $env:SystemDrive.TrimEnd(':')
            $volume = Get-Volume -DriveLetter $systemDrive -ErrorAction Stop
            $freeDiskGB = [math]::Round($volume.SizeRemaining/1GB,2)
            $totalDiskGB = [math]::Round($volume.Size/1GB,2)
            Write-Host "System drive $systemDrive`: free $freeDiskGB GB / total $totalDiskGB GB"
            if ($freeDiskGB -lt 10) { Write-Check 'WARN' 'Less than 10 GB free on the system drive.' } else { Write-Check 'PASS' 'System drive free space is at least 10 GB.' }
        } catch { Write-Check 'WARN' 'Could not read system-drive free space.' }
        Get-Process -ErrorAction SilentlyContinue |
            Where-Object { $_.ProcessName -match 'docker|com\.docker|vmmem|wsl' } |
            Select-Object ProcessName,Id,CPU,@{Name='Memory_MB';Expression={[math]::Round($_.WorkingSet64/1MB,0)}} |
            Format-Table -AutoSize | Out-String | Write-Host
    }

    # 12. Live ports and HTTP endpoints. Does not post/upload any evidence.
    Invoke-Section '12A. LOCAL TCP PORT CHECKS' {
        Test-TcpPort -Port 3000
        Test-TcpPort -Port 8001
        Test-TcpPort -Port 7474
        Test-TcpPort -Port 7687
    }
    Invoke-Section '12B. LOCAL HTTP ENDPOINT CHECKS (READ-ONLY GET REQUESTS)' {
        foreach ($url in @(
            'http://127.0.0.1:3000/',
            'http://127.0.0.1:3000/login',
            'http://127.0.0.1:8001/health',
            'http://127.0.0.1:8001/ready',
            'http://127.0.0.1:8001/docs',
            'http://127.0.0.1:8001/openapi.json'
        )) { Test-HttpEndpoint -Url $url }
        Write-Check 'INFO' 'Port 8000 is not included: docker-compose.yml defines research-engine on host port 8001.'
    }

    # 13. CI configuration and test coverage inventory
    Invoke-Section '13. CI WORKFLOW / TEST COVERAGE INVENTORY' {
        $ciFile = Join-Path $script:Root '.github\workflows\ci.yml'
        if (Test-Path $ciFile) {
            Write-Host 'CI workflow commands and setup lines:'
            Select-String -Path $ciFile -Pattern 'python-version|node-version|pip install|pytest|compileall|npm ci|npm test|npm run build|working-directory' |
                ForEach-Object { Write-Host (Protect-Text $_.Line.Trim()) }
            Write-Check 'PASS' 'CI workflow file found.'
        } else { Write-Check 'WARN' 'No .github/workflows/ci.yml found.' }
        $frontendTestFiles = @(Get-ChildItem (Join-Path $script:Root 'frontend-web\tests') -Filter '*.test.*' -File -ErrorAction SilentlyContinue)
        $pythonTestFiles = @(Get-ChildItem (Join-Path $script:Root 'research-engine\tests') -Filter 'test_*.py' -File -ErrorAction SilentlyContinue)
        Write-Host "Frontend test files: $($frontendTestFiles.Count)"
        Write-Host "Research-engine test files: $($pythonTestFiles.Count)"
        Write-Check 'INFO' 'Unit tests do not establish model accuracy, calibrated coverage on arbitrary uploads, camera generalisation, or forensic validity.'
    }

    # 14. Clear statement of what this script intentionally does not test.
    Invoke-Section '14. TESTS NOT AUTOMATED BY THIS HEALTH CHECK' {
        Write-Check 'INFO' 'No live evidence-upload POST was sent. That would create application/evidence state and requires a user-selected test video and statement.'
        Write-Check 'INFO' 'No full CLIP model download or fresh model inference was forced. It may require network access, cache space, and substantial CPU/GPU time.'
        Write-Check 'INFO' 'No Docker image rebuild was started. The research-engine dependency set can pull very large PyTorch/CUDA wheels; rebuild should be a deliberate separate step.'
        Write-Check 'INFO' 'No browser-driven end-to-end test was run unless such a test is explicitly configured in the project; Vitest is the frontend unit/component test suite.'
        Write-Check 'INFO' 'No database data was deleted, no volumes were pruned, and no containers were restarted.'
        Write-Check 'INFO' 'Health endpoints and green tests confirm service/test execution, not the correctness of a specific MATCH/CONTRADICTION decision.'
    }

    # 15. Summary
    Invoke-Section '15. FINAL SUMMARY' {
        Write-Host ("PASS: {0}" -f $script:PassCount)
        Write-Host ("WARN: {0}" -f $script:WarnCount)
        Write-Host ("FAIL: {0}" -f $script:FailCount)
        Write-Host ("INFO: {0}" -f $script:InfoCount)
        Write-Host ('Finished: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'))
        Write-Host ('Report: ' + $script:ReportPath)
        if ($script:FailCount -eq 0 -and $script:WarnCount -eq 0) { Write-Check 'PASS' 'No diagnostic warnings/failures were recorded by this run.' }
        elseif ($script:FailCount -gt 0) { Write-Check 'WARN' 'Review every [FAIL] entry. Some failures may be expected when a service/dependency is not running.' }
        else { Write-Check 'WARN' 'Review every [WARN] entry; warnings often indicate missing optional services or unconfigured decision artifacts.' }
    }
}
catch {
    Write-Host ('FATAL REPORT ERROR: ' + (Protect-Text $_.Exception.Message))
}
finally {
    try { Stop-Transcript | Out-Null } catch {}
}

Write-Host "`r`n============================================================"
Write-Host 'CRIMEVISION HEALTH CHECK FINISHED'
Write-Host "Report saved to: $script:ReportPath"
Write-Host 'Opening report in Notepad...'
if ($script:ReportPath -and (Test-Path -LiteralPath $script:ReportPath)) {
    try { Start-Process notepad.exe -ArgumentList @('"' + $script:ReportPath + '"') }
    catch { Write-Host "Could not launch Notepad automatically. Open this file manually: $script:ReportPath" }
}
