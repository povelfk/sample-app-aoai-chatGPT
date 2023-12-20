# Read the .env file and convert it to a hashtable
$envVariables = @{}
Get-Content .env | Where-Object { $_ -match '^(.+?)=(.*)$' } | ForEach-Object {
    $envVariables[$Matches[1]] = $Matches[2]
}

# Build the settings string
$settings = $envVariables.GetEnumerator() | ForEach-Object {
    '{0}={1}' -f $_.Key, $_.Value
}

# Set the application settings
az webapp config appsettings set `
    --subscription ME-MngEnvMCAP854902-povelf-2 `
    --resource-group rg-digge-sweden `
    --name digge-test-sweden `
    --settings $settings