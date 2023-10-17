#---------------#
# Load packages #
#---------------#
import os
import re
os.environ["OPENAI_API_KEY"] = "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"
from typing import Optional, Any, List
import requests
from contextlib import asynccontextmanager
import os
import logging
import sys
import argparse
import datetime
import pathlib
from pprint import pprint
import base64
from io import BytesIO
import json
import time

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.websockets import WebSocket
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
from PIL import Image
import openai

from constants import DB_PATH
from rag.retrieval import VDABRetrieverClient
from agent import get_agent
from callback import StreamingLLMCallbackHandler
from schemas import ChatResponse
from prompts import SYSTEM_MESSAGE

#---------------#
# Open AI stuff #
#---------------#
# Function to load chat history from a JSON file
def load_chat_history():
    try:
        with open('chat_history.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []


# # Function to save chat history to a JSON file
# def save_chat_history(chat_history : dict[str: Any]) -> None:
#     with open('chat_history.json', 'w') as file:
#         json.dump(chat_history, file)

# Function to generate chatbot response
def get_chatbot_response(msg: list[dict[str:Any]]):

    open_ai_response = openai.ChatCompletion.create(
        engine="test",
        messages=msg,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=True  
    )

    return open_ai_response


#----------#
# Local DB #
#----------#
user_db = {}
retrievers = []

#-----------------#
# FAST API - test #
#-----------------#
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Loading vectorstore")
    retriever_client = VDABRetrieverClient(chunk_size=300, chunk_overlap=0)
    # if the db is not built, build it
    if not (
        os.path.exists(DB_PATH / "store.pickle")
        or os.path.exists(DB_PATH / "chroma_db")
    ):
        logging.info("Building vectorstore from scratch")
        retriever_client.build_retriever()
    retriever = retriever_client.load_retriever()
    global retrievers
    retrievers.append(retriever)
    logging.info("Loaded vectorstore")
    yield
    # Clean up the ML models and release the resources
    del retrievers


app = FastAPI(
    title="Hackathon - Digitaal Vlaanderen",
    lifespan=lifespan)

# app.config["PREFERRED_URL_SCHEME"] = "https"
os.chdir(str(pathlib.Path(__file__).parent))
app.mount("/static", StaticFiles(directory=f"static"), name="static")
templates = Jinja2Templates(directory=f"templates")

@app.on_event("startup")
async def startup_event():
    pass



#-----------------#
# Websocket stuff #
#-----------------#
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        pprint(user_db)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def send_personal_json(self, message: dict[str|Any], websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stream_handler = StreamingLLMCallbackHandler(websocket)
    observations = []
    qa_chain, memory = get_agent(
        stream_handler, websocket, observations, retriever=retrievers[0]
    )
    while True:
        try:

            # Get the data
            data = await websocket.receive_json()
            data['client_id'] = websocket.client.port

            # response placeholder
            response = {'client_id': data['client_id'], 'request': data['request']}


            pprint(data)
            if data['request'] == 'new_user':
                response.update(new_user(data, websocket))
                await manager.send_personal_json(response, websocket)


            elif data['request'] == 'get_session_data':
                response.update(get_session_data(data))
                await manager.send_personal_json(response, websocket)

            elif data['request'] == 'new_message':
                await new_message(data, response, manager, websocket)

            elif True:

                print('SHIIIIIIIIIIIIT')
                # Receive and send back the client message
                qa_chain, memory = get_agent(
                    stream_handler,
                    websocket,
                    observations,
                    retriever=retrievers[0],
                    memory=memory,
                )

                question = "hallo"
                resp = ChatResponse(sender="you", message=question, type="stream")
                await websocket.send_json(resp.dict())

                # Construct a response
                start_resp = ChatResponse(sender="bot", message="", type="start")
                await websocket.send_json(start_resp.dict())
                result = await qa_chain.acall(
                    {"input": question, "stored_observations": observations}
                )
                print(result)
                end_resp = ChatResponse(sender="bot", message="", type="end")
                await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            resp = ChatResponse(
                sender="bot",
                message="",
                type="error",
            )
            logging.error(e)
            await websocket.send_json(resp.dict())

# @app.websocket("/ws/{client_id}")
# async def websocket_endpoint(websocket: WebSocket, client_id: int):
#     await manager.connect(websocket)
#     try:
#         while True:

#             # get the data
#             data = await websocket.receive_json()
#             data['client_id'] = client_id

#             # response placeholder
#             response = {'client_id': client_id, 'request': data['request']}

#             # send the data to the correct method
#             if data['request'] == 'new_user':
#                 response.update(new_user(data))
#                 await manager.send_personal_json(response, websocket)

#             elif data['request'] == 'get_session_data':
#                 response.update(get_session_data(data))
#                 await manager.send_personal_json(response, websocket)

#             elif data['request'] == 'new_topic':
#                 response.update(new_topic(data))
#                 await manager.send_personal_json(response, websocket)

#             elif data['request'] == 'new_message':
#                 await new_message(data, response, manager, websocket)
            
#             elif data['request'] == 'activate_topic':
#                 response.update(activate_topic(data))
#                 await manager.send_personal_json(response, websocket)
                

#             # await manager.broadcast(f"Client #{client_id} says: {data}")
#     except WebSocketDisconnect:
#         manager.disconnect(websocket)
#         await manager.broadcast(f"Client #{client_id} left the chat")



#-------------#
# Application #
#-------------#
@app.get("/")
async def get(request: Request):
    if "OPENAI_API_KEY" not in os.environ:
        raise HTTPException(status_code=404, detail="api key not found")
    else:
        if "WEBSITE_HOSTNAME" in os.environ:
            website_name = os.environ["WEBSITE_HOSTNAME"]
        else:
            website_name = "localhost:8000"
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "insert_endpoint": "ws://" + website_name + "/ws"},
        )


#----------#
# Requests #
#----------#
def new_user(data: dict[str:Any], websocket:Any) -> dict[str|Any]:
    """
        Create a new user
    """

    stream_handler = StreamingLLMCallbackHandler(websocket)
    observations = []
    qa_chain, memory = get_agent(
        stream_handler, websocket, observations, retriever=retrievers[0]
    )


    user_db[data['client_id']] = {'client_language': data['client_language'], 
                                  'last_activity': datetime.datetime.now(), 
                                  'active_session': None,
                                  'sessions': [{'title': None,
                                                'conversation':[{"role": "system", "content": SYSTEM_MESSAGE}],
                                                'stream_handler': stream_handler,
                                                'observations': observations,
                                                'qa_chain': qa_chain,
                                                'memory': memory
                                                }]}
    
    return {'client_language': data['client_language']}


def new_topic(data: dict[str:Any]) -> dict[str|Any]:
    """
    Show the session data
    """
    # get the client data
    client_data = user_db[data['client_id']]

    # Create a new session
    client_data['sessions'].append({'title': None,
                                    'conversation':[{"role": "system", "content": SYSTEM_MESSAGE}]
                                    })
    # set the active session
    client_data['active_session'] = len(client_data['sessions']) - 1

    # set the last activity date
    client_data['last_activity'] = datetime.datetime.now()

    return {"msg": "A new topic was created"}
    


def get_session_data(data: dict[str:Any]) -> dict[str|Any]:
    """
        Get the session data
    """
    # get the client data
    client_data = user_db[data['client_id']]

    # set the last activity date
    client_data['last_activity'] = datetime.datetime.now()

    if client_data['active_session'] is None:
        # return nothing
        return {'title_sessions': []}
    else:
        # return the session titles 
        # + the id of the active session
        # + the old conversation
        # TODO: de historiek
        return {'title_sessions': [x['title'] for x in client_data['sessions']],
                'session_ids': list(range(len(client_data['sessions']))),
                'active_session': client_data['active_session'],
                'session': client_data['sessions'][client_data['active_session']]}
    
def activate_topic(data: dict[str:Any]) -> dict[str|Any]:
    """
        Activate a topic
    """
    # get the client data
    client_data = user_db[data['client_id']]

    # set the last activity date
    client_data['last_activity'] = datetime.datetime.now()

    # set the active session
    client_data['active_session'] = int(data['session_id'])

    # return the session titles 
    # + the id of the active session
    # + the old conversation
    return {'active_session': client_data['active_session'],
            'session': client_data['sessions'][client_data['active_session']]}
    
async def new_message(data: dict[str:Any], response: dict[str:Any], manager: Any, websocket:Any) -> dict[str|Any]:
    """
        A new message comes in
    """
    # clean the msg

    #
    # TODO: ALERT: Als msg niet is ingevuld, en enkel een afbeelding, dan is msg een lege string!!!!
    #
    msg = data['msg'].strip()

    # load the images
    images = []
    #pprint(data['image'])
    # for e, raw_image in enumerate(data['pdf']):

    #     if len(raw_image) < 100:
    #         continue

    #     # get the image
    #     image_path = raw_image.split(';base64,',1)
    #     image_bytes =  base64.b64decode(image_path[1])

    #     # convert to image
    #     image = Image.open(BytesIO(image_bytes))

    #     # write image to file, if you want temp folder?
    #     # image.save(f'temp_{e}.{image_path[0].rsplit("/",1)[-1]}')

    #     # convert to numpy array
    #     images.append(np.array(image))
    #     print(f'Image shape: {images[-1].shape}')

    # get the client data
    client_data = user_db[data['client_id']]

    # if the session does not exist create one
    if client_data['active_session'] is None:
        client_data['active_session'] = 0

    # set the session id
    session_id = client_data['active_session']


    # add the client / user message to the session
    client_data['sessions'][session_id]['conversation'].append({"role": "user", "content" : msg})
    client_data['sessions'][session_id]['conversation'].append({"role": "assistent", "content" : 'Done'})

    # set the last activity date
    client_data['last_activity'] = datetime.datetime.now()


    # get the message
    if client_data['sessions'][session_id]['title'] is None:
        # set the title
        # client_data['sessions'][session_id]['title'] = text[:23].capitalize()
        client_data['sessions'][session_id]['title'] = msg[:23].capitalize()
        await websocket.send_json({'request': 'edit_title', 
                                   'title_sessions': [x['title'] for x in client_data['sessions']],
                                   "session_ids" : list(range(len(client_data['sessions'])))})

    # Construct a response
    start_resp = ChatResponse(sender="bot", message="", type="start").model_dump()
    await websocket.send_json(start_resp)


    result = await client_data['sessions'][session_id]['qa_chain'].acall({"input": msg, "stored_observations": client_data['sessions'][session_id]['observations']})


    end_resp = ChatResponse(sender="bot", message="", type="end").model_dump()
    await websocket.send_json(end_resp)


    # # Get all the chunks
    # for chunk in get_chatbot_response(client_data['sessions'][session_id]['conversation'][:-1]):
    #     if len(chunk.get('choices',[])) == 0:
    #         continue
    #     if chunk['choices'][0].get('delta',{}).get('content') is None:
    #         continue

    #     # get the word
    #     word = chunk['choices'][0]['delta']['content']
    #     text += word

    #     # create the response
    #     data = {'msg': word, 
    #             "msg-count": len(client_data['sessions'][session_id]['conversation']),
    #             'client_id': response['client_id'],
    #             'active_session': client_data['active_session'],
    #             'request': response['request']
    #             }

    #     if (len(text) > 23) and (client_data['sessions'][session_id]['title'] is None):

    #         # set the title
    #         # client_data['sessions'][session_id]['title'] = text[:23].capitalize()
    #         client_data['sessions'][session_id]['title'] = msg[:23].capitalize()

    #         # Add update the titles
    #         data["title_sessions"] = [x['title'] for x in client_data['sessions']]
    #         data["session_ids"] = list(range(len(client_data['sessions'])))



    #     # update the conversation
    #     client_data['sessions'][session_id]['conversation'][-1]['content'] += word
    #     await manager.send_personal_json(data, websocket)







    
def check_arg(args=None):
    """
    Check if the user has specified a host or port number
    """
    parser = argparse.ArgumentParser(description='Start: Universal Sentence Encoder - API')
    parser.add_argument('-H', '--host', help='host ip', default='0.0.0.0')
    
    parser.add_argument('-p', '--port',  help='port of the web server',  default='8000')

    results = parser.parse_args(args)
    try:
        results.port = int(results.port)
    except ValueError:
        print('port is invalid')
        int(int(results.port))
        
    return (results.host, results.port)




if __name__ == "__main__":
    
    # check the argvars
    host, port = check_arg(sys.argv[1:])
    
    #start server   
    uvicorn.run(app, host=host, port=port)
