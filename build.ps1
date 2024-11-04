Push-Location (Split-Path -Parent $PSCommandPath)

Push-Location model

Get-ChildItem -Filter *.pyd | Remove-Item
Get-ChildItem -Filter *.pyi | Remove-Item

Pop-Location

Push-Location binds

if (-Not (Test-Path pybind11\build)) {
    Push-Location pybind11
    mkdir build
    Push-Location build
    cmake ..
    cmake --build . --config Release --target check
    Pop-Location
    Pop-Location
}

if (Test-Path build) {
    Remove-Item build\* -Recurse -Force
}
else {
    mkdir build
}

Push-Location build

cmake ..

$config = "Release"

if ($args.Count -ne 0) {
    $config = $args[0]
}
cmake --build . --config $config

Pop-Location
Pop-Location

Move-Item "binds/build/$($config)/graph.cp312-win_amd64.pyd" "model/graph.cp312-win_amd64.pyd"

Push-Location model

$stubgenCmd = "stubgen "
foreach ($file in (Get-ChildItem -Filter *.pyd)) {
    $stubgenCmd = $stubgenCmd + "-m $($file.Name.Split('.')[0]) "
}
$stubgenCmd = $stubgenCmd + "-o ."

Invoke-Expression $stubgenCmd

Pop-Location