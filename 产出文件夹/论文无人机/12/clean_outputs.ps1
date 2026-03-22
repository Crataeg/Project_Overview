$ErrorActionPreference = "Stop"

$paths = @(
    "output\measured_corner_compare_smoke",
    "output\__tmp_kpi",
    "output\tmp_BLER_B.csv",
    "output\tmp_bler_b2.csv",
    "output\__tmp_ga.npy",
    "output\__tmp_gan.npy",
    "__pycache__"
)

foreach ($path in $paths) {
    if (Test-Path $path) {
        Remove-Item $path -Recurse -Force
        Write-Host "Removed $path"
    }
}
