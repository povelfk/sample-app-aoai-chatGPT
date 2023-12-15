Get-Content secrets.txt | ForEach-Object {
    az webapp config appsettings set --name digge --resource-group digge --settings $_
  }
  