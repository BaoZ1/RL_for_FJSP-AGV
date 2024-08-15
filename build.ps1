Push-Location (Split-Path -Parent $PSCommandPath)

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

Get-ChildItem -Filter *.pyd | Remove-Item
Get-ChildItem -Filter *.pyi | Remove-Item

if (Test-Path build) {
    Remove-Item build\* -Recurse -Force
}
else {
    mkdir build
}

Push-Location build

cmake ..

if ($args.Count -eq 0) {
    cmake --build . --config Release
}
else {
    cmake --build . --config $args[0]
}

Pop-Location

$stubgenCmd = "stubgen "
foreach ($file in (Get-ChildItem -Filter *.pyd)) {
    $stubgenCmd = $stubgenCmd + "-m $($file.Name.Split('.')[0]) "
}
$stubgenCmd = $stubgenCmd + "-o ."

Invoke-Expression $stubgenCmd

Pop-Location