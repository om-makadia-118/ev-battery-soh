$ErrorActionPreference = 'Stop'

$paths = @(
    'data/raw/nasa/B0005.mat',
    'data/raw/oxford/Oxford_Battery_Degradation_Dataset_1.mat',
    'data/raw/xjtu/XJTU_3C_battery-1.pkl'
)

foreach ($path in $paths) {
    if (Test-Path $path) {
        Write-Host "`n===== $path =====" -ForegroundColor Cyan
        python -m battery_soh inspect --path $path
    } else {
        Write-Host "NOT FOUND: $path" -ForegroundColor Yellow
    }
}
