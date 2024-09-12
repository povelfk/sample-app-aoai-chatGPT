import json
import os
import logging
import requests
import openai
import copy
from azure.identity import DefaultAzureCredential
from base64 import b64encode
from flask import Flask, Response, request, jsonify, send_from_directory, session
from dotenv import load_dotenv

from backend.auth.auth_utils import get_authenticated_user_details
from backend.history.cosmosdbservice import CosmosConversationClient

from azure.monitor.opentelemetry import configure_azure_monitor

# Importing from common.helper
from common.helper import (
    # is_json,
    check_missing_parameters,
    ask_follow_up_question,
    ask_confirmation_question,
    classify,
    call_function,
    count_tokens,
    generate_static_message
)

load_dotenv()

app = Flask(__name__, static_folder="static")
app.secret_key = os.environ.get("FLASK_KEY")

# # Configure OpenTelemetry to use Azure Monitor with the specified connection string.
# configure_azure_monitor(
#     connection_string=os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"),
#     disable_tracing=True,
#     disable_logging=False,
#     disable_metrics=False,
#     instrumentation_options={
#         "azure_sdk": {"enabled": True},
#         "flask": {"enabled": True},
#         "django": {"enabled": False},
#     }
# )

class ConversationHandler:
    def __init__(self):
        self.MAX_TOKENS = 50000 #16384
        self.function_requires_confirmation = {
            '4': True,
            '5': True,
            '6': True,
            '7': True
        } # functions that requires confirmation

    def conversation_internal(self, request_body):

        # initialize session variables, if not already present
        if 'user_confirmation' not in session:
            session['user_confirmation'] = False

        CURRENT_TOKEN_COUNT = count_tokens(AZURE_OPENAI_SYSTEM_MESSAGE_CLASSIFIER, request_body)
        print(f"CURRENT_TOKEN_COUNT: {CURRENT_TOKEN_COUNT}")
        if CURRENT_TOKEN_COUNT > self.MAX_TOKENS:
            return generate_static_message("Du har nått maxgränsen av antal ord, vänligen starta en ny konversation.", request_body)
        
        # classify the user input
        classification_result = classify(request_body)
        if isinstance(classification_result, str):  # Check if the result is a string (error message)
            return generate_static_message(classification_result, request_body)
        else:
            function_to_call, parameters, is_function, classification_message = classification_result
            if is_function:
                missing_params = check_missing_parameters(parameters)
                if missing_params:
                    follow_up_question = ask_follow_up_question(missing_params, function_to_call, classification_message, request_body)
                    return follow_up_question
                else:
                    function_augmented_prompt = None  # Initialize variable outside the if-else scope for clarity

                    if self.function_requires_confirmation.get(function_to_call, False):
                        request_body = request.get_json()
                        latest_user_message = request_body["messages"][-1]["content"].lower()

                        if latest_user_message == "nej":
                            return generate_static_message("Ok, jag avbryter processen.", request_body)
                        elif latest_user_message != "ja":
                            return ask_confirmation_question(parameters, classification_message, function_to_call, request_body)
                        
                        session['user_confirmation'] = True
                        function_augmented_prompt = call_function(function_to_call, parameters)
                        session['user_confirmation'] = False
                    else:
                        function_augmented_prompt = call_function(function_to_call, parameters)
                    return conversation_without_data(request_body, function_augmented_prompt)
            else:
                try:
                    use_data = should_use_data()
                    if use_data:
                        return conversation_with_data(request_body)
                    else:
                        return conversation_without_data(request_body)
                except Exception as e:
                    logging.exception("Exception in /conversation")
                    return jsonify({"error": str(e)}), 500

handler = ConversationHandler()

# Static Files
@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/favicon.ico")
def favicon():
    return app.send_static_file('favicon.ico')

@app.route("/assets/<path:path>")
def assets(path):
    return send_from_directory("static/assets", path)

# Debug settings
DEBUG = os.environ.get("DEBUG", "false")
DEBUG_LOGGING = DEBUG.lower() == "true"
if DEBUG_LOGGING:
    logging.basicConfig(level=logging.DEBUG)

# On Your Data Settings
DATASOURCE_TYPE = os.environ.get("DATASOURCE_TYPE", "azure_search") # CHANGE AzureCognitiveSearch
SEARCH_TOP_K = os.environ.get("SEARCH_TOP_K", 5)
SEARCH_STRICTNESS = os.environ.get("SEARCH_STRICTNESS", 3)
SEARCH_ENABLE_IN_DOMAIN = os.environ.get("SEARCH_ENABLE_IN_DOMAIN", "true")

# ACS Integration Settings
AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX")
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
AZURE_SEARCH_USE_SEMANTIC_SEARCH = os.environ.get("AZURE_SEARCH_USE_SEMANTIC_SEARCH", "true")
AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG = os.environ.get("AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG", "default")
AZURE_SEARCH_TOP_K = os.environ.get("AZURE_SEARCH_TOP_K", SEARCH_TOP_K)
AZURE_SEARCH_ENABLE_IN_DOMAIN = os.environ.get("AZURE_SEARCH_ENABLE_IN_DOMAIN", SEARCH_ENABLE_IN_DOMAIN)
AZURE_SEARCH_CONTENT_COLUMNS = os.environ.get("AZURE_SEARCH_CONTENT_COLUMNS")
AZURE_SEARCH_FILENAME_COLUMN = os.environ.get("AZURE_SEARCH_FILENAME_COLUMN")
AZURE_SEARCH_TITLE_COLUMN = os.environ.get("AZURE_SEARCH_TITLE_COLUMN")
AZURE_SEARCH_URL_COLUMN = os.environ.get("AZURE_SEARCH_URL_COLUMN")
AZURE_SEARCH_VECTOR_COLUMNS = os.environ.get("AZURE_SEARCH_VECTOR_COLUMNS")
AZURE_SEARCH_QUERY_TYPE = os.environ.get("AZURE_SEARCH_QUERY_TYPE")
AZURE_SEARCH_PERMITTED_GROUPS_COLUMN = os.environ.get("AZURE_SEARCH_PERMITTED_GROUPS_COLUMN")
AZURE_SEARCH_STRICTNESS = os.environ.get("AZURE_SEARCH_STRICTNESS", SEARCH_STRICTNESS)

# AOAI Integration Settings
AZURE_OPENAI_RESOURCE = os.environ.get("AZURE_OPENAI_RESOURCE")
AZURE_OPENAI_MODEL = os.environ.get("AZURE_OPENAI_MODEL")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_TEMPERATURE = os.environ.get("AZURE_OPENAI_TEMPERATURE", 0)
AZURE_OPENAI_TOP_P = os.environ.get("AZURE_OPENAI_TOP_P", 1.0)
AZURE_OPENAI_MAX_TOKENS = os.environ.get("AZURE_OPENAI_MAX_TOKENS", 1000)
AZURE_OPENAI_STOP_SEQUENCE = os.environ.get("AZURE_OPENAI_STOP_SEQUENCE")
# AZURE_OPENAI_SYSTEM_MESSAGE = os.environ.get("AZURE_OPENAI_SYSTEM_MESSAGE", "You are an AI assistant that helps people find information.")
with open('prompts/system_message_chat.txt', 'r', encoding='utf-8') as file:
    AZURE_OPENAI_SYSTEM_MESSAGE_CHAT = file.read()
with open('prompts/system_message_custom_classifier.txt', 'r', encoding='utf-8') as file:
    AZURE_OPENAI_SYSTEM_MESSAGE_CLASSIFIER = file.read()
AZURE_OPENAI_PREVIEW_API_VERSION = os.environ.get("AZURE_OPENAI_PREVIEW_API_VERSION", "2024-02-01")
AZURE_OPENAI_STREAM = os.environ.get("AZURE_OPENAI_STREAM", "true")
AZURE_OPENAI_MODEL_NAME = os.environ.get("AZURE_OPENAI_MODEL_NAME", "gpt-35-turbo-16k") # Name of the model, e.g. 'gpt-35-turbo-16k' or 'gpt-4'
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT")
AZURE_OPENAI_EMBEDDING_KEY = os.environ.get("AZURE_OPENAI_EMBEDDING_KEY")
AZURE_OPENAI_EMBEDDING_NAME = os.environ.get("AZURE_OPENAI_EMBEDDING_NAME", "")

# CosmosDB Mongo vcore vector db Settings
AZURE_COSMOSDB_MONGO_VCORE_CONNECTION_STRING = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_CONNECTION_STRING")  #This has to be secure string
AZURE_COSMOSDB_MONGO_VCORE_DATABASE = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_DATABASE")
AZURE_COSMOSDB_MONGO_VCORE_CONTAINER = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_CONTAINER")
AZURE_COSMOSDB_MONGO_VCORE_INDEX = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_INDEX")
AZURE_COSMOSDB_MONGO_VCORE_TOP_K = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_TOP_K", AZURE_SEARCH_TOP_K)
AZURE_COSMOSDB_MONGO_VCORE_STRICTNESS = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_STRICTNESS", AZURE_SEARCH_STRICTNESS)  
AZURE_COSMOSDB_MONGO_VCORE_ENABLE_IN_DOMAIN = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_ENABLE_IN_DOMAIN", AZURE_SEARCH_ENABLE_IN_DOMAIN)
AZURE_COSMOSDB_MONGO_VCORE_CONTENT_COLUMNS = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_CONTENT_COLUMNS", "")
AZURE_COSMOSDB_MONGO_VCORE_FILENAME_COLUMN = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_FILENAME_COLUMN")
AZURE_COSMOSDB_MONGO_VCORE_TITLE_COLUMN = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_TITLE_COLUMN")
AZURE_COSMOSDB_MONGO_VCORE_URL_COLUMN = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_URL_COLUMN")
AZURE_COSMOSDB_MONGO_VCORE_VECTOR_COLUMNS = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_VECTOR_COLUMNS")


SHOULD_STREAM = True if AZURE_OPENAI_STREAM.lower() == "true" else False

# Chat History CosmosDB Integration Settings
AZURE_COSMOSDB_DATABASE = os.environ.get("AZURE_COSMOSDB_DATABASE")
AZURE_COSMOSDB_ACCOUNT = os.environ.get("AZURE_COSMOSDB_ACCOUNT")
AZURE_COSMOSDB_CONVERSATIONS_CONTAINER = os.environ.get("AZURE_COSMOSDB_CONVERSATIONS_CONTAINER")
AZURE_COSMOSDB_ACCOUNT_KEY = os.environ.get("AZURE_COSMOSDB_ACCOUNT_KEY")

# Elasticsearch Integration Settings
ELASTICSEARCH_ENDPOINT = os.environ.get("ELASTICSEARCH_ENDPOINT")
ELASTICSEARCH_ENCODED_API_KEY = os.environ.get("ELASTICSEARCH_ENCODED_API_KEY")
ELASTICSEARCH_INDEX = os.environ.get("ELASTICSEARCH_INDEX")
ELASTICSEARCH_QUERY_TYPE = os.environ.get("ELASTICSEARCH_QUERY_TYPE", "simple")
ELASTICSEARCH_TOP_K = os.environ.get("ELASTICSEARCH_TOP_K", SEARCH_TOP_K)
ELASTICSEARCH_ENABLE_IN_DOMAIN = os.environ.get("ELASTICSEARCH_ENABLE_IN_DOMAIN", SEARCH_ENABLE_IN_DOMAIN)
ELASTICSEARCH_CONTENT_COLUMNS = os.environ.get("ELASTICSEARCH_CONTENT_COLUMNS")
ELASTICSEARCH_FILENAME_COLUMN = os.environ.get("ELASTICSEARCH_FILENAME_COLUMN")
ELASTICSEARCH_TITLE_COLUMN = os.environ.get("ELASTICSEARCH_TITLE_COLUMN")
ELASTICSEARCH_URL_COLUMN = os.environ.get("ELASTICSEARCH_URL_COLUMN")
ELASTICSEARCH_VECTOR_COLUMNS = os.environ.get("ELASTICSEARCH_VECTOR_COLUMNS")
ELASTICSEARCH_STRICTNESS = os.environ.get("ELASTICSEARCH_STRICTNESS", SEARCH_STRICTNESS)
ELASTICSEARCH_EMBEDDING_MODEL_ID = os.environ.get("ELASTICSEARCH_EMBEDDING_MODEL_ID")


AZURE_STORAGE_ACCOUNT_KEY = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")

# Initialize a CosmosDB client with AAD auth and containers for Chat History
cosmos_conversation_client = None
if AZURE_COSMOSDB_DATABASE and AZURE_COSMOSDB_ACCOUNT and AZURE_COSMOSDB_CONVERSATIONS_CONTAINER:
    try :
        cosmos_endpoint = f'https://{AZURE_COSMOSDB_ACCOUNT}.documents.azure.com:443/'

        if not AZURE_COSMOSDB_ACCOUNT_KEY:
            credential = DefaultAzureCredential()
        else:
            credential = AZURE_COSMOSDB_ACCOUNT_KEY

        cosmos_conversation_client = CosmosConversationClient(
            cosmosdb_endpoint=cosmos_endpoint, 
            credential=credential, 
            database_name=AZURE_COSMOSDB_DATABASE,
            container_name=AZURE_COSMOSDB_CONVERSATIONS_CONTAINER
        )
    except Exception as e:
        logging.exception("Exception in CosmosDB initialization", e)
        cosmos_conversation_client = None


def is_chat_model():
    if 'gpt-4' in AZURE_OPENAI_MODEL_NAME.lower() or AZURE_OPENAI_MODEL_NAME.lower() in ['gpt-35-turbo-4k', 'gpt-35-turbo-16k']:
        return True
    return False

def should_use_data():
    if AZURE_SEARCH_SERVICE and AZURE_SEARCH_INDEX and AZURE_SEARCH_KEY:
        if DEBUG_LOGGING:
            logging.debug("Using Azure Cognitive Search")
        return True
    
    if AZURE_COSMOSDB_MONGO_VCORE_DATABASE and AZURE_COSMOSDB_MONGO_VCORE_CONTAINER and AZURE_COSMOSDB_MONGO_VCORE_INDEX and AZURE_COSMOSDB_MONGO_VCORE_CONNECTION_STRING:
        if DEBUG_LOGGING:
            logging.debug("Using Azure CosmosDB Mongo vcore")
        return True
    
    return False


def format_as_ndjson(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"

def fetchUserGroups(userToken, nextLink=None):
    # Recursively fetch group membership
    if nextLink:
        endpoint = nextLink
    else:
        endpoint = "https://graph.microsoft.com/v1.0/me/transitiveMemberOf?$select=id"
    
    headers = {
        'Authorization': "bearer " + userToken
    }
    try :
        r = requests.get(endpoint, headers=headers)
        if r.status_code != 200:
            if DEBUG_LOGGING:
                logging.error(f"Error fetching user groups: {r.status_code} {r.text}")
            return []
        
        r = r.json()
        if "@odata.nextLink" in r:
            nextLinkData = fetchUserGroups(userToken, r["@odata.nextLink"])
            r['value'].extend(nextLinkData)
        
        return r['value']
    except Exception as e:
        logging.error(f"Exception in fetchUserGroups: {e}")
        return []


def generateFilterString(userToken):
    # Get list of groups user is a member of
    userGroups = fetchUserGroups(userToken)

    # Construct filter string
    if not userGroups:
        logging.debug("No user groups found")

    group_ids = ", ".join([obj['id'] for obj in userGroups])
    return f"{AZURE_SEARCH_PERMITTED_GROUPS_COLUMN}/any(g:search.in(g, '{group_ids}'))"


def prepare_body_headers_with_data(request):
    request_messages = request.json["messages"]
    refined_request_message = []
    for message in request_messages:
        refined_request_message.append(
            {
                'role': message["role"],
                'content': message["content"]
            }
        )

    body = {
        "messages": refined_request_message,
        "temperature": float(AZURE_OPENAI_TEMPERATURE),
        "max_tokens": int(AZURE_OPENAI_MAX_TOKENS),
        "top_p": float(AZURE_OPENAI_TOP_P),
        "stop": AZURE_OPENAI_STOP_SEQUENCE.split("|") if AZURE_OPENAI_STOP_SEQUENCE else None,
        "stream": SHOULD_STREAM,
        "data_sources": [],
        "seed": 42
    }

    if DATASOURCE_TYPE == "azure_search": # CHANGE AzureCognitiveSearch
        # Set query type
        query_type = "simple"
        if AZURE_SEARCH_QUERY_TYPE:
            query_type = AZURE_SEARCH_QUERY_TYPE
        elif AZURE_SEARCH_USE_SEMANTIC_SEARCH.lower() == "true" and AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG:
            query_type = "semantic"

        # Set filter
        filter = None
        userToken = None
        if AZURE_SEARCH_PERMITTED_GROUPS_COLUMN:
            userToken = request.headers.get('X-MS-TOKEN-AAD-ACCESS-TOKEN', "")
            if DEBUG_LOGGING:
                logging.debug(f"USER TOKEN is {'present' if userToken else 'not present'}")

            filter = generateFilterString(userToken)
            if DEBUG_LOGGING:
                logging.debug(f"FILTER: {filter}")

        body["data_sources"].append(
            {
                "type": "azure_search", # CHANGE AzureCognitiveSearch
                "parameters": {
                    "authentication": {
                        "key": AZURE_SEARCH_KEY,
                        "type": "api_key"
                    },
                    "endpoint": f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
                    "index_name": AZURE_SEARCH_INDEX,
                    "in_scope": True if AZURE_SEARCH_ENABLE_IN_DOMAIN.lower() == "true" else False,
                    "top_n_documents": AZURE_SEARCH_TOP_K,
                    "query_type": query_type, # vector_semantic_hybrid CHANGE
                    "semantic_configuration": AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG if AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG else "",
                    "role_information": AZURE_OPENAI_SYSTEM_MESSAGE_CHAT,
                    "filter": filter,
                    "strictness": int(AZURE_SEARCH_STRICTNESS),
                    "embedding_dependency": {
                        "type": "endpoint",
                        "endpoint": AZURE_OPENAI_EMBEDDING_ENDPOINT,
                        "authentication": {
                            "key": AZURE_OPENAI_KEY,
                            "type": "api_key"
                        }
                    },
                    "fields_mapping": {
                        "content_fields": AZURE_SEARCH_CONTENT_COLUMNS.split("|") if AZURE_SEARCH_CONTENT_COLUMNS else [],
                        "title_field": AZURE_SEARCH_TITLE_COLUMN if AZURE_SEARCH_TITLE_COLUMN else None,
                        "url_field": AZURE_SEARCH_URL_COLUMN if AZURE_SEARCH_URL_COLUMN else None,
                        "filepath_field": AZURE_SEARCH_FILENAME_COLUMN if AZURE_SEARCH_FILENAME_COLUMN else None,
                        "vector_fields": AZURE_SEARCH_VECTOR_COLUMNS.split("|") if AZURE_SEARCH_VECTOR_COLUMNS else []
                    }
                }
            })
    elif DATASOURCE_TYPE == "AzureCosmosDB":
        # Set query type
        query_type = "vector"

        body["dataSources"].append(
            {
                "type": "AzureCosmosDB",
                "parameters": {
                    "connectionString": AZURE_COSMOSDB_MONGO_VCORE_CONNECTION_STRING,
                    "indexName": AZURE_COSMOSDB_MONGO_VCORE_INDEX,
                    "databaseName": AZURE_COSMOSDB_MONGO_VCORE_DATABASE,
                    "containerName": AZURE_COSMOSDB_MONGO_VCORE_CONTAINER,                    
                    "fieldsMapping": {
                        "contentFields": AZURE_COSMOSDB_MONGO_VCORE_CONTENT_COLUMNS.split("|") if AZURE_COSMOSDB_MONGO_VCORE_CONTENT_COLUMNS else [],
                        "titleField": AZURE_COSMOSDB_MONGO_VCORE_TITLE_COLUMN if AZURE_COSMOSDB_MONGO_VCORE_TITLE_COLUMN else None,
                        "urlField": AZURE_COSMOSDB_MONGO_VCORE_URL_COLUMN if AZURE_COSMOSDB_MONGO_VCORE_URL_COLUMN else None,
                        "filepathField": AZURE_COSMOSDB_MONGO_VCORE_FILENAME_COLUMN if AZURE_COSMOSDB_MONGO_VCORE_FILENAME_COLUMN else None,
                        "vectorFields": AZURE_COSMOSDB_MONGO_VCORE_VECTOR_COLUMNS.split("|") if AZURE_COSMOSDB_MONGO_VCORE_VECTOR_COLUMNS else []
                    },
                    "inScope": True if AZURE_COSMOSDB_MONGO_VCORE_ENABLE_IN_DOMAIN.lower() == "true" else False,
                    "topNDocuments": AZURE_COSMOSDB_MONGO_VCORE_TOP_K,
                    "strictness": int(AZURE_COSMOSDB_MONGO_VCORE_STRICTNESS),
                    "queryType": query_type,
                    "roleInformation": AZURE_OPENAI_SYSTEM_MESSAGE_CHAT
                }
            }
        )

    elif DATASOURCE_TYPE == "Elasticsearch":
        body["dataSources"].append(
            {
                "messages": request_messages,
                "temperature": float(AZURE_OPENAI_TEMPERATURE),
                "max_tokens": int(AZURE_OPENAI_MAX_TOKENS),
                "top_p": float(AZURE_OPENAI_TOP_P),
                "stop": AZURE_OPENAI_STOP_SEQUENCE.split("|") if AZURE_OPENAI_STOP_SEQUENCE else None,
                "stream": SHOULD_STREAM,
                "dataSources": [
                    {
                        "type": "AzureCognitiveSearch",
                        "parameters": {
                            "endpoint": ELASTICSEARCH_ENDPOINT,
                            "encodedApiKey": ELASTICSEARCH_ENCODED_API_KEY,
                            "indexName": ELASTICSEARCH_INDEX,
                            "fieldsMapping": {
                                "contentFields": ELASTICSEARCH_CONTENT_COLUMNS.split("|") if ELASTICSEARCH_CONTENT_COLUMNS else [],
                                "titleField": ELASTICSEARCH_TITLE_COLUMN if ELASTICSEARCH_TITLE_COLUMN else None,
                                "urlField": ELASTICSEARCH_URL_COLUMN if ELASTICSEARCH_URL_COLUMN else None,
                                "filepathField": ELASTICSEARCH_FILENAME_COLUMN if ELASTICSEARCH_FILENAME_COLUMN else None,
                                "vectorFields": ELASTICSEARCH_VECTOR_COLUMNS.split("|") if ELASTICSEARCH_VECTOR_COLUMNS else []
                            },
                            "inScope": True if ELASTICSEARCH_ENABLE_IN_DOMAIN.lower() == "true" else False,
                            "topNDocuments": int(ELASTICSEARCH_TOP_K),
                            "queryType": ELASTICSEARCH_QUERY_TYPE,
                            "roleInformation": AZURE_OPENAI_SYSTEM_MESSAGE_CHAT,
                            "embeddingEndpoint": AZURE_OPENAI_EMBEDDING_ENDPOINT,
                            "embeddingKey": AZURE_OPENAI_EMBEDDING_KEY,
                            "embeddingModelId": ELASTICSEARCH_EMBEDDING_MODEL_ID,
                            "strictness": int(ELASTICSEARCH_STRICTNESS)
                        }
                    }
                ]
            }
        )
    else:
        raise Exception(f"DATASOURCE_TYPE is not configured or unknown: {DATASOURCE_TYPE}")

    # CHANGE
    # if "vector" in query_type.lower():
        # if AZURE_OPENAI_EMBEDDING_NAME:
            # body["dataSources"][0]["parameters"]["embeddingDeploymentName"] = AZURE_OPENAI_EMBEDDING_NAME
        # else:
            # body["dataSources"][0]["parameters"]["embeddingEndpoint"] = AZURE_OPENAI_EMBEDDING_ENDPOINT
            # body["dataSources"][0]["parameters"]["embeddingKey"] = AZURE_OPENAI_EMBEDDING_KEY

    # CHANGE
    if DEBUG_LOGGING:
        body_clean = copy.deepcopy(body)
        if body_clean["data_source"][0]["parameters"].get("key"):
            body_clean["data_source"][0]["parameters"]["key"] = "*****"
        if body_clean["data_source"][0]["parameters"].get("connection_string"):
            body_clean["data_source"][0]["parameters"]["connection_string"] = "*****"
        if body_clean["data_source"][0]["parameters"].get("embedding_key"):
            body_clean["data_source"][0]["parameters"]["embedding_key"] = "*****"
        # if body_clean["dataSources"][0]["parameters"].get("key"):
        #     body_clean["dataSources"][0]["parameters"]["key"] = "*****"
        # if body_clean["dataSources"][0]["parameters"].get("connectionString"):
        #     body_clean["dataSources"][0]["parameters"]["connectionString"] = "*****"
        # if body_clean["dataSources"][0]["parameters"].get("embeddingKey"): 
        #     body_clean["dataSources"][0]["parameters"]["embeddingKey"] = "*****" 
            
        logging.debug(f"REQUEST BODY: {json.dumps(body_clean, indent=4)}")

    headers = {
        'Content-Type': 'application/json',
        'api-key': AZURE_OPENAI_KEY,
        "x-ms-useragent": "GitHubSampleWebApp/PublicAPI/3.0.0"
    }

    return body, headers


def stream_with_data(body, headers, endpoint, history_metadata={}):
    s = requests.Session()
    try:
        with s.post(endpoint, json=body, headers=headers, stream=True) as r:
            for line in r.iter_lines(chunk_size=10):
                response = {
                    "id": "",
                    "model": "",
                    "created": 0,
                    "object": "",
                    "choices": [{
                        "messages": []
                    }],
                    "apim-request-id": "",
                    'history_metadata': history_metadata
                }
                if line:
                    if AZURE_OPENAI_PREVIEW_API_VERSION == '2023-06-01-preview':
                        lineJson = json.loads(line.lstrip(b'data:').decode('utf-8'))
                    else:
                        try:
                            rawResponse = json.loads(line.lstrip(b'data:').decode('utf-8'))
                            lineJson = formatApiResponseStreaming(rawResponse)
                        except json.decoder.JSONDecodeError:
                            continue

                    if 'error' in lineJson:
                        yield format_as_ndjson(lineJson)
                    response["id"] = lineJson["id"]
                    response["model"] = lineJson["model"]
                    response["created"] = lineJson["created"]
                    response["object"] = lineJson["object"]
                    response["apim-request-id"] = r.headers.get('apim-request-id')

                    role = lineJson["choices"][0]["messages"][0]["delta"].get("role")

                    if role == "tool":
                        response["choices"][0]["messages"].append(lineJson["choices"][0]["messages"][0]["delta"])
                        yield format_as_ndjson(response)
                    elif role == "assistant": 
                        if response['apim-request-id'] and DEBUG_LOGGING: 
                            logging.debug(f"RESPONSE apim-request-id: {response['apim-request-id']}")
                        response["choices"][0]["messages"].append({
                            "role": "assistant",
                            "content": ""
                        })
                        yield format_as_ndjson(response)
                    else:
                        deltaText = lineJson["choices"][0]["messages"][0]["delta"]["content"]
                        if deltaText != "[DONE]":
                            response["choices"][0]["messages"].append({
                                "role": "assistant",
                                "content": deltaText
                            })
                            yield format_as_ndjson(response)
    except Exception as e:
        yield format_as_ndjson({"error" + str(e)})

def formatApiResponseNoStreaming(rawResponse):
    if 'error' in rawResponse:
        return {"error": rawResponse["error"]}
    response = {
        "id": rawResponse["id"],
        "model": rawResponse["model"],
        "created": rawResponse["created"],
        "object": rawResponse["object"],
        "choices": [{
            "messages": []
        }],
    }
    toolMessage = {
        "role": "tool",
        "content": rawResponse["choices"][0]["message"]["context"]["messages"][0]["content"]
    }
    assistantMessage = {
        "role": "assistant",
        "content": rawResponse["choices"][0]["message"]["content"]
    }
    response["choices"][0]["messages"].append(toolMessage)
    response["choices"][0]["messages"].append(assistantMessage)

    return response

def formatApiResponseStreaming(rawResponse):
    if 'error' in rawResponse:
        return {"error": rawResponse["error"]}
    response = {
        "id": rawResponse["id"],
        "model": rawResponse["model"],
        "created": rawResponse["created"],
        "object": rawResponse["object"],
        "choices": [{
            "messages": []
        }],
    }
    if rawResponse["choices"][0]["delta"].get("context"):
        if len(rawResponse["choices"][0]["delta"]["context"]["citations"])>0:
            content = rawResponse["choices"][0]["delta"]["context"]["citations"]
            intent = json.loads(rawResponse["choices"][0]["delta"]["context"]["intent"])
            # print(f"content: {content}")
            # print(f"intent: {intent}")
            messageObj = {
                "delta": {
                    "role": "tool",
                    "content": json.dumps({
                        "citations": content,
                        "intent": intent
                    })
                }
            }
        else:
            messageObj = {
                "delta": {
                    "role": "tool",
                    "content": str({
                        "citations": [],
                        "intent": str([])
                    })
                }
            }
        response["choices"][0]["messages"].append(messageObj)

    else:
        if rawResponse["choices"][0]["end_turn"]:
            messageObj = {
                "delta": {
                    "content": "[DONE]",
                }
            }
            response["choices"][0]["messages"].append(messageObj)
        else:
            messageObj = {
                "delta": {
                    "content": rawResponse["choices"][0]["delta"]["content"],
                }
            }
            response["choices"][0]["messages"].append(messageObj)
    # print(f"response: {response}")
    # print()
    return response

def conversation_with_data(request_body):
    body, headers = prepare_body_headers_with_data(request)
    base_url = AZURE_OPENAI_ENDPOINT if AZURE_OPENAI_ENDPOINT else f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/"
    # CHANGE
    # endpoint = f"{base_url}openai/deployments/{AZURE_OPENAI_MODEL}/extensions/chat/completions?api-version={AZURE_OPENAI_PREVIEW_API_VERSION}"
    endpoint = f"{base_url}openai/deployments/{AZURE_OPENAI_MODEL}/chat/completions?api-version={AZURE_OPENAI_PREVIEW_API_VERSION}"
    history_metadata = request_body.get("history_metadata", {})

    if not SHOULD_STREAM:
        r = requests.post(endpoint, headers=headers, json=body)
        status_code = r.status_code
        r = r.json()
        if AZURE_OPENAI_PREVIEW_API_VERSION == "2023-06-01-preview":
            r['history_metadata'] = history_metadata
            return Response(format_as_ndjson(r), status=status_code)
        else:
            result = formatApiResponseNoStreaming(r)
            result['history_metadata'] = history_metadata
            return Response(format_as_ndjson(result), status=status_code)

    else:
        return Response(stream_with_data(body, headers, endpoint, history_metadata), mimetype='text/event-stream')

def stream_without_data(response, history_metadata={}):
    responseText = ""
    for i, line in enumerate(response):
        if len(line.choices)>0:
            deltaText = line.choices[0].delta.content
        else:
            deltaText = ""
        if deltaText and deltaText != "[DONE]":
            responseText = deltaText

        response_obj = {
            "id": line.id,
            "model": line.model,
            "created": line.created,
            "object": line.object,
            "choices": [{
                "messages": [{
                    "role": "assistant",
                    "content": responseText
                }]
            }],
            "history_metadata": history_metadata
        }
        yield format_as_ndjson(response_obj)

def conversation_without_data(request_body, function_augmented_prompt):
    key = AZURE_OPENAI_KEY
    api_version = AZURE_OPENAI_PREVIEW_API_VERSION
    base_url = AZURE_OPENAI_ENDPOINT if AZURE_OPENAI_ENDPOINT else f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/"

    client = openai.AzureOpenAI(
        api_key=key,
        api_version=api_version,
        azure_endpoint=base_url
    )

    response = client.chat.completions.create(
        model=AZURE_OPENAI_MODEL,
        messages=function_augmented_prompt,
        stream=SHOULD_STREAM
    )

    history_metadata = request_body.get("history_metadata", {})

    if not SHOULD_STREAM:
        response_obj = {
            "id": response.id,
            "model": response.model,
            "created": response.created,
            "object": response.object,
            "choices": [{
                "messages": [{
                    "role": "assistant",
                    "content": response.choices[0].message.content
                }]
            }],
            "history_metadata": history_metadata
        }

        return jsonify(response_obj), 200
    else:
        return Response(stream_without_data(response, history_metadata), mimetype='text/event-stream')

@app.route("/blob-sas-url", methods=["GET", "POST"])
def get_blob_sas_url():
    from azure.storage.blob import generate_blob_sas, BlobSasPermissions, BlobClient
    from datetime import datetime, timedelta
    blob_url = request.args.get("bloburl")

    blob_client = BlobClient.from_blob_url(blob_url)
    sas_url = generate_blob_sas(account_name=blob_client.account_name,
                      account_key=AZURE_STORAGE_ACCOUNT_KEY,
                      container_name=blob_client.container_name,
                      blob_name=blob_client.blob_name,
                      permission=BlobSasPermissions(read=True),
                      expiry= datetime.utcnow() + timedelta(minutes=5))

    return jsonify({"sas_url": blob_url + "?" + sas_url}), 200
  

@app.route("/conversation", methods=["GET", "POST"])
def conversation():
    request_body = request.json
    return handler.conversation_internal(request_body)
    

# def conversation_internal(request_body):
#     function_to_call, parameters, is_function, classification_message = classify(request_body)
#     if is_function:
#         missing_params = check_missing_parameters(parameters)
#         if missing_params:
#             follow_up_question = ask_follow_up_question(missing_params, classification_message, request_body)
#             return follow_up_question
#         else:
#             function_augmented_prompt = call_function(function_to_call, parameters)
#             return conversation_without_data(request_body, function_augmented_prompt)
#     else:
#         try:
#             use_data = should_use_data()
#             if use_data:
#                 return conversation_with_data(request_body)
#             else:
#                 return conversation_without_data(request_body)
#         except Exception as e:
#             logging.exception("Exception in /conversation")
#             return jsonify({"error": str(e)}), 500
        

## Conversation History API ## 
@app.route("/history/generate", methods=["POST"])
def add_conversation():
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']
    # access_token = authenticated_user['access_token']

    ## check request for conversation_id
    conversation_id = request.json.get("conversation_id", None)

    try:
        # make sure cosmos is configured
        if not cosmos_conversation_client:
            raise Exception("CosmosDB is not configured")

        # check for the conversation_id, if the conversation is not set, we will create a new one
        history_metadata = {}
        if not conversation_id:
            title = generate_title(request.json["messages"])
            conversation_dict = cosmos_conversation_client.create_conversation(user_id=user_id, title=title)
            conversation_id = conversation_dict['id']
            history_metadata['title'] = title
            history_metadata['date'] = conversation_dict['createdAt']
            
        ## Format the incoming message object in the "chat/completions" messages format
        ## then write it to the conversation history in cosmos
        messages = request.json["messages"]
        if len(messages) > 0 and messages[-1]['role'] == "user":
            cosmos_conversation_client.create_message(
                conversation_id=conversation_id,
                user_id=user_id,
                input_message=messages[-1]
            )
        else:
            raise Exception("No user message found")
        
        # Submit request to Chat Completions for response
        request_body = request.json
        history_metadata['conversation_id'] = conversation_id
        request_body['history_metadata'] = history_metadata
        
        # user_token = None
        # user_token = request.headers.get('X-MS-TOKEN-AAD-ACCESS-TOKEN', "")
        return handler.conversation_internal(request_body)
       
    except Exception as e:
        logging.exception("Exception in /history/generate")
        return jsonify({"error": str(e)}), 500


@app.route("/history/update", methods=["POST"])
def update_conversation():
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']

    # check request for conversation_id
    conversation_id = request.json.get("conversation_id", None)

    try:
        # make sure cosmos is configured
        if not cosmos_conversation_client:
            raise Exception("CosmosDB is not configured")

        # check for the conversation_id, if the conversation is not set, we will create a new one
        if not conversation_id:
            raise Exception("No conversation_id found")
            
        ## Format the incoming message object in the "chat/completions" messages format
        ## then write it to the conversation history in cosmos
        messages = request.json["messages"]
        if len(messages) > 0 and messages[-1]['role'] == "assistant":
            if len(messages) > 1 and messages[-2].get('role', None) == "tool":
                # write the tool message first
                cosmos_conversation_client.create_message(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    input_message=messages[-2]
                )
            # write the assistant message
            cosmos_conversation_client.create_message(
                conversation_id=conversation_id,
                user_id=user_id,
                input_message=messages[-1]
            )
        else:
            raise Exception("No bot messages found")
        
        # Submit request to Chat Completions for response
        response = {'success': True}
        return jsonify(response), 200
       
    except Exception as e:
        logging.exception("Exception in /history/update")
        return jsonify({"error": str(e)}), 500

@app.route("/history/delete", methods=["DELETE"])
def delete_conversation():
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']
    
    ## check request for conversation_id
    conversation_id = request.json.get("conversation_id", None)
    try: 
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400
        
        ## delete the conversation messages from cosmos first
        deleted_messages = cosmos_conversation_client.delete_messages(conversation_id, user_id)

        ## Now delete the conversation 
        deleted_conversation = cosmos_conversation_client.delete_conversation(user_id, conversation_id)

        return jsonify({"message": "Successfully deleted conversation and messages", "conversation_id": conversation_id}), 200
    except Exception as e:
        logging.exception("Exception in /history/delete")
        return jsonify({"error": str(e)}), 500

@app.route("/history/list", methods=["GET"])
def list_conversations():
    offset = request.args.get("offset", 0)
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']

    ## get the conversations from cosmos
    conversations = cosmos_conversation_client.get_conversations(user_id, offset=offset, limit=25)
    if not isinstance(conversations, list):
        return jsonify({"error": f"No conversations for {user_id} were found"}), 404

    ## return the conversation ids

    return jsonify(conversations), 200

@app.route("/history/read", methods=["POST"])
def get_conversation():
    session['user_confirmation'] = False
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']

    ## check request for conversation_id
    conversation_id = request.json.get("conversation_id", None)
    
    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400

    ## get the conversation object and the related messages from cosmos
    conversation = cosmos_conversation_client.get_conversation(user_id, conversation_id)
    ## return the conversation id and the messages in the bot frontend format
    if not conversation:
        return jsonify({"error": f"Conversation {conversation_id} was not found. It either does not exist or the logged in user does not have access to it."}), 404
    
    # get the messages for the conversation from cosmos
    conversation_messages = cosmos_conversation_client.get_messages(user_id, conversation_id)

    ## format the messages in the bot frontend format
    messages = [{'id': msg['id'], 'role': msg['role'], 'content': msg['content'], 'createdAt': msg['createdAt']} for msg in conversation_messages]

    return jsonify({"conversation_id": conversation_id, "messages": messages}), 200

@app.route("/history/rename", methods=["POST"])
def rename_conversation():
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']

    ## check request for conversation_id
    conversation_id = request.json.get("conversation_id", None)
    
    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400
    
    ## get the conversation from cosmos
    conversation = cosmos_conversation_client.get_conversation(user_id, conversation_id)
    if not conversation:
        return jsonify({"error": f"Conversation {conversation_id} was not found. It either does not exist or the logged in user does not have access to it."}), 404

    ## update the title
    title = request.json.get("title", None)
    if not title:
        return jsonify({"error": "title is required"}), 400
    conversation['title'] = title
    updated_conversation = cosmos_conversation_client.upsert_conversation(conversation)

    return jsonify(updated_conversation), 200

@app.route("/history/delete_all", methods=["DELETE"])
def delete_all_conversations():
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']

    # get conversations for user
    try:
        conversations = cosmos_conversation_client.get_conversations(user_id, offset=0, limit=None)
        if not conversations:
            return jsonify({"error": f"No conversations for {user_id} were found"}), 404
        
        # delete each conversation
        for conversation in conversations:
            ## delete the conversation messages from cosmos first
            deleted_messages = cosmos_conversation_client.delete_messages(conversation['id'], user_id)

            ## Now delete the conversation 
            deleted_conversation = cosmos_conversation_client.delete_conversation(user_id, conversation['id'])

        return jsonify({"message": f"Successfully deleted conversation and messages for user {user_id}"}), 200
    
    except Exception as e:
        logging.exception("Exception in /history/delete_all")
        return jsonify({"error": str(e)}), 500
    

@app.route("/history/clear", methods=["POST"])
def clear_messages():
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']
    
    ## check request for conversation_id
    conversation_id = request.json.get("conversation_id", None)
    try: 
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400
        
        ## delete the conversation messages from cosmos
        deleted_messages = cosmos_conversation_client.delete_messages(conversation_id, user_id)

        return jsonify({"message": "Successfully deleted messages in conversation", "conversation_id": conversation_id}), 200
    except Exception as e:
        logging.exception("Exception in /history/clear_messages")
        return jsonify({"error": str(e)}), 500

@app.route("/history/ensure", methods=["GET"])
def ensure_cosmos():
    if not AZURE_COSMOSDB_ACCOUNT:
        return jsonify({"error": "CosmosDB is not configured"}), 404
    
    if not cosmos_conversation_client or not cosmos_conversation_client.ensure():
        return jsonify({"error": "CosmosDB is not working"}), 500

    return jsonify({"message": "CosmosDB is configured and working"}), 200


def generate_title(conversation_messages):
    ## make sure the messages are sorted by _ts descending
    title_prompt = 'Summarize the conversation so far into a 4-word or less title. Do not use any quotation marks or punctuation. Respond with a json object in the format {{"title": string}}. Do not include any other commentary or description.'

    messages = [{'role': msg['role'], 'content': msg['content']} for msg in conversation_messages]
    messages.append({'role': 'user', 'content': title_prompt})



    try:
        ## Submit prompt to Chat Completions for response
        # CHANGE-SDK
        base_url = AZURE_OPENAI_ENDPOINT if AZURE_OPENAI_ENDPOINT else f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/"
        client = openai.AzureOpenAI(
            azure_endpoint=base_url,
            api_key=AZURE_OPENAI_KEY,
            api_version="2024-02-01",
        )
        
        completion = client.chat.completions.create(
            model=AZURE_OPENAI_MODEL,
            messages=messages,
            temperature=1,
            max_tokens=64
        )
        title = json.loads(completion.choices[0].text)['title']
        return title
    except Exception as e:
        return messages[-2]['content']

if __name__ == "__main__":
    app.run()