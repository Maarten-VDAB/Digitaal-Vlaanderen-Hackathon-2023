import streamlit as st
import json
import os
import openai
import requests

api_type = "azure"
api_base = "https://tst-vdab.openai.azure.com"
api_version = "2023-08-01-preview"
deployment_id = "test"
api_endpoint = f"{api_base}/openai/deployments/{deployment_id}/extensions/chat/completions?api-version={api_version}" 
api_key = os.getenv("AZURE_OPENAI_KEY")
search_endpoint = "https://hackathon-vdab-cognitive-search.search.windows.net"
search_key = os.getenv("AZURE_SEARCH_KEY")
search_index_name = "azureblob-index"
ada_resource_name = "tst-vdab"
ada_deployment_name = "test_embeddings"
ada_endpoint = f"https://{ada_resource_name}.openai.azure.com/openai/deployments/{ada_deployment_name}/embeddings?api-version=2023-05-15"
ada_key = os.getenv("AZURE_OPENAI_KEY")


def setup_byod(deployment_id: str) -> None:
    """Sets up the OpenAI Python SDK to use your own data for the chat endpoint.

    :param deployment_id: The deployment ID for the model to use with your own data.

    To remove this configuration, simply set openai.requestssession to None.
    """

    class BringYourOwnDataAdapter(requests.adapters.HTTPAdapter):

        def send(self, request, **kwargs):
            request.url = api_endpoint
            return super().send(request, **kwargs)

    session = requests.Session()

    # Mount a custom adapter which will use the extensions endpoint for any call using the given `deployment_id`
    session.mount(
        prefix=f"{openai.api_base}/openai/deployments/{deployment_id}",
        adapter=BringYourOwnDataAdapter()
    )

    openai.requestssession = session

setup_byod(deployment_id)

# Function to generate chatbot response
def get_chatbot_response(message):

    system_message = {
        "role": "system",
        "content": "You are a VDAB consulent who helps and supports unemployed civilians with their search for a job.\nAll questions asked should be considered and answered within a VDAB context.\nAll questions are answered in Dutch.\n\nWhen people ask questions, consider whether you have enough input information to provide them with an accurate answer. If not, generate additional questions that would help you give a more accurate answer. When they have provided answers to these additional questions, combine the answers to produce the final answers to their original question. Always consider previous chat history.\n\nWhen people provide statements with information, generate relevant questions. These questions should be based on the provided information, related VDAB information, and chat history. These questions should be asking for relevant information, needed to further assist the person. \n\nYou have to ask additional questions  when a person seems uncertain about what they are looking for during the conversation. Take the answers to these questions and chat history into account, while providing suggestions.\n\nWhen referring to the VDAB website, try to provide a relevant url.\n\nDuring the conversation, think of additional VDAB information that might be relevant to share with the person.\n\nGenerate friendly and reassuring responses. Be proactive, make suggestions.\n\nDo not explicitly mention to the user when information cannot be found in the data."
    }
    user_message = {"role": "user", "content": message}
    data_sources = [
        {
            "type": "AzureCognitiveSearch",
            "parameters": {
                "endpoint": search_endpoint,
                "key": search_key,
                "indexName": search_index_name,
                "inScope": True,
                "topNDocuments": 20,
                "strictness": 1,
                "fieldsMapping": {
                    "contentFields": ["content"],
                    "titleField": "title",
                    "urlField": "url",
                    "filepathField": "title",
                    "vectorFields": ["ada_vector"]
                },
                "queryType": 'vectorSimpleHybrid',
                "roleInformation": system_message['content'],
                "embeddingEndpoint": ada_endpoint,
                "embeddingKey": ada_key
            }
        }
    ]
    
    body = {
        'messages': [*load_chat_history(), user_message],
        'deployment_id': deployment_id,
        'dataSources': data_sources,
        'temperature': 0,
        'max_tokens': 800,
        'top_p': 0.95,
        'frequency_penalty': 0,
        'presence_penalty': 0,
        'stop': None
    }

    headers = {
        'Content-Type': 'application/json',
        'api-key': api_key
    }

    r = requests.post(api_endpoint, headers=headers, json=body)
    response = r.json()

    return response['choices'][0]['message']['content']

# Function to load chat history from a JSON file
def load_chat_history():
    try:
        with open('chat_history.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []

# Function to save chat history to a JSON file
def save_chat_history(chat_history):
    with open('chat_history.json', 'w') as file:
        json.dump(chat_history, file)

# Streamlit UI
st.title("Chatbot UI")

# Load chat history or initialize an empty one
chat_history = load_chat_history()

# Display chat history in a conversational format
for item in chat_history:
    if item['role'] == 'user':
        st.text(f"You: {item['content']}")
    else:
        st.markdown(f"**Assistant:** {item['content']}")

# Add a horizontal line below the chat history
st.markdown("---")

# Create a placeholder for user input
user_input_placeholder = st.empty()

# Text input for user input
user_input = user_input_placeholder.text_input("Input:")

# Create two columns for Send and Clear buttons
col1, col2 = st.columns([.11,1])

# Put the Send button in the first column and the Clear button in the second column
if col1.button("Send"):
    if user_input:

        # Add user input to chat history
        chat_history.append({'role': 'user', 'content': user_input})

        # Generate chatbot response (replace this with your own chatbot logic)
        chatbot_response = get_chatbot_response(user_input)

        chat_history.append({'role': 'assistant', 'content': chatbot_response})

        # Save the chat history
        save_chat_history(chat_history)

        # Clear the user input field by updating the placeholder
        user_input_placeholder.text_input("Input:", value="", key="unique_key")

if col2.button("Clear"):
    save_chat_history([])
    st.experimental_rerun()
