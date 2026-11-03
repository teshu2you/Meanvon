Get-ChildItem 'i:\meanvon\tmp' -File | Group-Object { ($_.BaseName -replace '[0-9]', 'N').Substring(0, [Math]::Min(10, ($_.BaseName -replace '[0-9]', 'N').Length)) } | Sort-Object Count -Descending | Select-Object -First 5 Name, Count | Format-Table
Write-Output "oldest-3:"
Get-ChildItem 'i:\meanvon\tmp' -File | Sort-Object LastWriteTime | Select-Object -First 3 Name, LastWriteTime | Format-Table
