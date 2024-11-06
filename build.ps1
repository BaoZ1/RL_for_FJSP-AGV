[CmdletBinding()]
param (
    [Parameter()]
    [ValidateSet("bind", "backend", "frontend", "all")]
    $target="all",

    [Parameter()]
    [ValidateSet("Debug", "Release")]
    $mode="Debug"
)

$root = Split-Path -Parent $PSCommandPath

function BuildBind {
    Set-Location "$($root)/binds/pybind11"

    if (-Not (Test-Path build)) {
        New-Item build -ItemType "directory"
        Set-Location build
        cmake ..
        cmake --build . --config Release --target check
    }

    Set-Location "$($root)/binds"

    if (-Not (Test-Path build)) {
        New-Item build -ItemType "directory"
    }

    Set-Location build

    cmake ..

    $config = (Get-Culture).TextInfo.ToTitleCase($mode)

    cmake --build . --config $config

    Set-Location "$($root)/model"

    Move-Item "$($root)/binds/build/$($config)/graph.cp312-win_amd64.pyd" graph.cp312-win_amd64.pyd -Force

    $stubgenCmd = "stubgen "
    foreach ($file in (Get-ChildItem -Filter *.pyd)) {
        $stubgenCmd = $stubgenCmd + "-m $($file.Name.Split('.')[0]) "
    }
    $stubgenCmd = $stubgenCmd + "-o ."

    Invoke-Expression $stubgenCmd
}

function BuildBackend {
    Set-Location "$($root)/backend"

    pyinstaller main.py --noconfirm

    Set-Location $root

    $prefix = rustc -Vv | Select-String "host:" | ForEach-Object {$_.Line.split(" ")[1]}

    Move-Item "backend/dist/main/main.exe" "frontend/src-tauri/binaries/backend-$($prefix).exe" -Force
}

function BuildFrontend {
    Push-Location "$($root)/frontend"

    if (-Not (Test-Path node_modules)) {
        npm install
    }

    npm run tauri build
}

try {
    switch ($target) {
        "bind" { BuildBind }
        "backend" { BuildBackend }
        "frontend" { BuildFrontend }
        "all" { BuildBind; BuildBackend; BuildFrontend }
    }
}
finally {
    Set-Location $root
}
