[CmdletBinding()]
param (
    [Parameter()]
    [ValidateSet("debug", "release")]
    $mode = "debug",

    [Parameter()]
    [switch]
    $updateBackend = $false
)

$root = Split-Path -Parent $PSCommandPath

try {
    if ($updateBackend) {
        Set-Location "$($root)/backend"

        .venv/Scripts/activate.ps1
        
        cxfreeze build

        deactivate
        
        $prefix = rustc -Vv | Select-String "host:" | ForEach-Object { $_.Line.split(" ")[1] }

        Rename-Item "build/main/main.exe" "FJSP-AGV_backend-$($prefix).exe"
    }

    Set-Location "$($root)/frontend"

    if (-Not (Test-Path node_modules)) {
        npm install
    }

    if ($mode -eq "debug") {
        npm run tauri dev
    }
    else {
        npm run tauri build
    }
}
finally {
    Set-Location $root
}
