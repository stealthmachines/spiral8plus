# Rename all human_ecoli*.py files to human_eco*.py
# Also update internal references from 'ecoli' to 'eco'

$targetDir = "C:\Users\Owner\Documents\Josef's Code 2025\Combined Works\ncbi_dataset_wuhan\advanced-spiral-8"
Set-Location $targetDir

# Get all human_ecoli*.py files
$files = Get-ChildItem -Filter "human_ecoli*.py"

Write-Host "Found" $files.Count "files to rename"

foreach ($file in $files) {
    $oldName = $file.Name
    $newName = $oldName -replace "human_ecoli", "human_eco"

    Write-Host "Processing: $oldName -> $newName"

    # Read file content
    $content = Get-Content $file.FullName -Raw

    # Replace 'ecoli' with 'eco' (case-sensitive and case-insensitive variants)
    # Be careful not to replace in URLs or where it's part of another word
    $content = $content -replace "human_ecoli", "human_eco"
    $content = $content -replace "ecoli_", "eco_"
    $content = $content -replace "_ecoli", "_eco"
    $content = $content -replace "E\.coli", "E.co"
    $content = $content -replace "E\. coli", "E. co"

    # Rename the file first
    Rename-Item -Path $file.FullName -NewName $newName -Force

    # Write updated content to the renamed file
    Set-Content -Path (Join-Path $targetDir $newName) -Value $content -NoNewline

    Write-Host "Renamed and updated content"
}

Write-Host ""
Write-Host "Complete! Renamed" $files.Count "files from human_ecoli to human_eco"