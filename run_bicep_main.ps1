# Connect-AzAccount
$subscriptionId = "ea41ce80-2e64-4e78-b06b-38c79e6f0d53"
Set-AzContext -SubscriptionId $subscriptionId

Set-Location -Path "infra/main.bicep"

$deploymentName = "myBicepStgDeployment"
$BicepFile = "main.bicep"
New-AzResourceGroupDeployment `
    -Name $deploymentName `
    -ResourceGroupName $ResourceGroupName `
    -TemplateFile $BicepFile

New-AzResourceGroupDeployment `
    -TemplateFile .\main.bicep `
    -ResourceGroupName <resourceGroupName> `
    -environmentName <environmentName> `
    -location <location> `
    -appServicePlanName <appServicePlanName> `
    -backendServiceName <backendServiceName> `
    -searchServiceName <searchServiceName> `
    -searchServiceResourceGroupName <searchServiceResourceGroupName> `
    -searchServiceResourceGroupLocation <searchServiceResourceGroupLocation> `
    -searchServiceSkuName <searchServiceSkuName> `
    -searchIndexName <searchIndexName> `
    -searchUseSemanticSearch <searchUseSemanticSearch> `
    -searchSemanticSearchConfig <searchSemanticSearchConfig> `
    -searchTopK <searchTopK> `
    -searchEnableInDomain <searchEnableInDomain> `
    -searchContentColumns <searchContentColumns> `
    -searchFilenameColumn <searchFilenameColumn> `
    -searchTitleColumn <searchTitleColumn> `
    -searchUrlColumn <searchUrlColumn> `
    -openAiResourceName <openAiResourceName> `
    -openAiResourceGroupName <openAiResourceGroupName> `
    -openAiResourceGroupLocation <openAiResourceGroupLocation> `
    -openAiSkuName <openAiSkuName> `
    -openAIModel <openAIModel> `
    -openAIModelName <openAIModelName> `
    -openAITemperature <openAITemperature> `
    -openAITopP <openAITopP> `
    -openAIMaxTokens <openAIMaxTokens> `
    -openAIStopSequence <openAIStopSequence> `
    -openAISystemMessage <openAISystemMessage> `
    -openAIApiVersion <openAIApiVersion> `
    -openAIStream <openAIStream> `
    -embeddingDeploymentName <embeddingDeploymentName> `
    -embeddingModelName <embeddingModelName> `
    -formRecognizerServiceName <formRecognizerServiceName> `
    -formRecognizerResourceGroupName <formRecognizerResourceGroupName> `
    -formRecognizerResourceGroupLocation <formRecognizerResourceGroupLocation> `
    -formRecognizerSkuName <formRecognizerSkuName> `
    -authClientId <authClientId> `
    -authClientSecret <authClientSecret> `
    -cosmosAccountName <cosmosAccountName> `
    -principalId <principalId>
