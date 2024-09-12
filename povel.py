import os
from openai import AzureOpenAI

client = AzureOpenAI(
  azure_endpoint="https://aoai-diggechatbot.openai.azure.com/", 
  api_key="257ad62044b14f2a9619490be225846d",
  api_version="2024-02-01"
)

response = client.chat.completions.create(
    model="gpt-35-turbo-1106", # model = "deployment_name".
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
        {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
        {"role": "user", "content": "Do other Azure AI services support this too?"}
    ]
)

# print(response.choices[0].message.content)
print(response)