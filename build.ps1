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
    $install = $false
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
}

function TryInstallModule {
    if($install){
        Set-Location "$($root)/binds/module"
        pip install -e .

        Set-Location "$($root)/model"
        pip install -e .
    }
}

function BuildBackend {
    Set-Location "$($root)/backend"

    pyinstaller main.py --noconfirm

    Set-Location $root

    $prefix = rustc -Vv | Select-String "host:" | ForEach-Object { $_.Line.split(" ")[1] }

    Move-Item "backend/dist/main/main.exe" "frontend/src-tauri/binaries/backend-$($prefix).exe" -Force
}

function BuildFrontend {
    Push-Location "$($root)/frontend"

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
        "module" { BuildBind; TryInstallModule }
        "frontend" { BuildBackend; BuildFrontend }
        "all" { BuildBind; TryInstallModule; BuildBackend; BuildFrontend }
    }
}
finally {
    Set-Location $root
}
