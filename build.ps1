[CmdletBinding()]
param (
    [Parameter()]
    [ValidateSet("module", "frontend", "all")]
    $target = "all",

    [Parameter()]
    [ValidateSet("debug", "release")]
    $mode = "debug",

    [Parameter()]
    [switch]
    $install = $false,

    [Parameter()]
    [switch]
    $updateBackend = $false
)

$root = Split-Path -Parent $PSCommandPath

function BuildBind {
    Set-Location "$($root)/binds"

    if (-Not (Test-Path pybind11/build)) {
        New-Item pybind11/build -ItemType "directory"
        Set-Location pybind11/build
        cmake ..
        cmake --build . --config Release --target check

        Set-Location "$($root)/binds"
    }


    if (-Not (Test-Path build)) {
        New-Item build -ItemType "directory"
    }

    Set-Location build

    cmake ..

    $config = (Get-Culture).TextInfo.ToTitleCase($mode)

    cmake --build . --config $config

    Set-Location "$($root)/binds/module/FJSP_env"

    Move-Item "$($root)/binds/build/$($config)/FJSP_env.cp312-win_amd64.pyd" FJSP_env.cp312-win_amd64.pyd -Force

    $stubgenCmd = "stubgen "
    foreach ($file in (Get-ChildItem -Filter *.pyd)) {
        $stubgenCmd = $stubgenCmd + "-m $($file.Name.Split('.')[0]) "
    }
    $stubgenCmd = $stubgenCmd + "-o ."

    Invoke-Expression $stubgenCmd

    if ($install) {
        Set-Location "$($root)/binds/module"
        pip install -e .

        Set-Location "$($root)/model"
        pip install -e .
    }
}

function BuildFrontend {
    if ($updateBackend) {
        Set-Location "$($root)/backend"

        .venv/Scripts/activate.ps1
        
        cxfreeze build

        deactivate
        
        $prefix = rustc -Vv | Select-String "host:" | ForEach-Object { $_.Line.split(" ")[1] }

        Rename-Item "build/main/main.exe" "FJSP-AGV_backend-$($prefix).exe"

        # Set-Location $root

        # Move-Item "backend/dist/main/main.exe" "frontend/src-tauri/binaries/FJSP-AGV_backend-$($prefix).exe" -Force

        # if (Test-Path "frontend/src-tauri/target/$($mode)/_internal") {
        #     Remove-Item "frontend/src-tauri/target/$($mode)/_internal" -Recurse -Force
        # }
        
        # $moved = $false
        # do {
        #     try {
        #         Move-Item "backend/dist/main/_internal" "frontend/src-tauri/target/$($mode)" -Force -ErrorAction Stop
        #         $moved = $true
        #     }
        #     catch {
        #         Start-Sleep -Seconds 2
        #     }
        # } while (-Not $moved)
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


try {
    switch ($target) {
        "module" { BuildBind }
        "frontend" { BuildFrontend }
        "all" { BuildBind; BuildFrontend }
    }
}
finally {
    Set-Location $root
}
