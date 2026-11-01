# Build a portable Windows package: embedded Python + dependencies + model.
# Output: dist\VoiceSorter\  and  dist\VoiceSorter-1.1.0.zip

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Dist = Join-Path $Root "dist\VoiceSorter"
$PyVer = "3.12.8"
$PyZip = Join-Path $Root "dist\python-embed.zip"
$PyMirrors = @(
  "https://mirrors.huaweicloud.com/python/$PyVer/python-$PyVer-embed-amd64.zip",
  "https://registry.npmmirror.com/-/binary/python/$PyVer/python-$PyVer-embed-amd64.zip",
  "https://www.python.org/ftp/python/$PyVer/python-$PyVer-embed-amd64.zip"
)
$PipMirrors = @(
  "https://mirrors.aliyun.com/pypi/get-pip.py",
  "https://bootstrap.pypa.io/get-pip.py"
)

Write-Host "== preparing $Dist"
if (Test-Path $Dist) { Remove-Item $Dist -Recurse -Force }
New-Item -ItemType Directory -Path $Dist -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Root "dist") -Force | Out-Null

if (-not (Test-Path $PyZip) -or (Get-Item $PyZip).Length -lt 1MB) {
  $ok = $false
  foreach ($u in $PyMirrors) {
    try {
      Write-Host "== downloading Python from $u"
      Invoke-WebRequest -Uri $u -OutFile $PyZip -UseBasicParsing -TimeoutSec 300
      if ((Get-Item $PyZip).Length -gt 1MB) { $ok = $true; break }
    } catch { Write-Host "   failed: $($_.Exception.Message)" }
  }
  if (-not $ok) { throw "could not download Python" }
}
Write-Host "== extracting python"
Expand-Archive -Path $PyZip -DestinationPath (Join-Path $Dist "python") -Force

$pth = Get-ChildItem (Join-Path $Dist "python") -Filter "python*._pth" | Select-Object -First 1
@"
python312.zip
.
Lib\site-packages
..
import site
"@ | Set-Content -Path $pth.FullName -Encoding ASCII
New-Item -ItemType Directory -Path (Join-Path $Dist "python\Lib\site-packages") -Force | Out-Null

Write-Host "== staging wheels (offline install)"
$Wheels = Join-Path $Root "dist\wheels"
New-Item -ItemType Directory -Path $Wheels -Force | Out-Null
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) { $VenvPy = "python" }
& $VenvPy -m pip download -r (Join-Path $Root "requirements.txt") -d $Wheels `
  --only-binary=:all: -i https://pypi.tuna.tsinghua.edu.cn/simple
& $VenvPy -m pip download pip -d $Wheels --only-binary=:all: `
  -i https://pypi.tuna.tsinghua.edu.cn/simple

$SitePackages = Join-Path $Dist "python\Lib\site-packages"
New-Item -ItemType Directory -Path $SitePackages -Force | Out-Null
$pipWheel = Get-ChildItem $Wheels -Filter "pip-*.whl" | Select-Object -First 1
if ($pipWheel) {
  Copy-Item $pipWheel.FullName (Join-Path $Dist "pip.zip")
  Expand-Archive -Path (Join-Path $Dist "pip.zip") -DestinationPath $SitePackages -Force
  Remove-Item (Join-Path $Dist "pip.zip")
}

Write-Host "== installing dependencies into the package"
& (Join-Path $Dist "python\python.exe") -m pip install --no-index --find-links $Wheels `
  -r (Join-Path $Root "requirements.txt") --no-warn-script-location

Write-Host "== copying application"
Copy-Item (Join-Path $Root "voicesorter") (Join-Path $Dist "voicesorter") -Recurse
Get-ChildItem (Join-Path $Dist "voicesorter") -Recurse -Directory -Filter "__pycache__" |
  Remove-Item -Recurse -Force
Copy-Item (Join-Path $Root "launcher\start.bat") (Join-Path $Dist "启动.bat")

$modelsSrc = Join-Path $Root "models"
if (Test-Path $modelsSrc) {
  New-Item -ItemType Directory -Path (Join-Path $Dist "models") -Force | Out-Null
  $model = Join-Path $modelsSrc "campplus_zh_16k.onnx"
  if (Test-Path $model) { Copy-Item $model (Join-Path $Dist "models") -Force }
}

Write-Host "== building launcher exe"
$csc = "C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if (Test-Path $csc) {
  $cs = Join-Path $Root "dist\launcher.cs"
  @'
using System;
using System.Diagnostics;
using System.IO;
class Launcher {
  static void Main(string[] args) {
    string dir = AppDomain.CurrentDomain.BaseDirectory;
    var psi = new ProcessStartInfo(Path.Combine(dir, "python", "python.exe"), "-m voicesorter serve");
    psi.WorkingDirectory = dir;
    psi.UseShellExecute = true;
    try { Process.Start(psi); } catch (Exception e) {
      Console.WriteLine("启动失败: " + e.Message);
      Console.ReadKey();
    }
  }
}
'@ | Set-Content -Path $cs -Encoding UTF8
  $cscArgs = @("/nologo", "/target:winexe", "/out:$Dist\VoiceSorter.exe", $cs)
  & $csc @cscArgs
} else {
  Write-Host "csc not found, keeping 启动.bat only"
}

Write-Host "== zipping"
$zip = Join-Path $Root "dist\VoiceSorter-1.1.0.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path "$Dist\*" -DestinationPath $zip

$size = (Get-ChildItem $zip).Length / 1MB
Write-Host ("== done: {0} ({1:N1} MB)" -f $zip, $size)
