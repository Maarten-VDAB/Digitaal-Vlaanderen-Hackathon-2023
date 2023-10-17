"""Define tools for the LLM to use."""
from langchain.retrievers import ParentDocumentRetriever
from langchain.tools import BaseTool
from fastapi import WebSocket

from functools import lru_cache
# from llava.model.builder import load_pretrained_model
from PIL import Image
import pdfplumber

# import torch

# from llava.conversation import conv_templates, SeparatorStyle
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
# from llava.utils import disable_torch_init
# from transformers import TextStreamer

from schemas import ChatResponse

class RAGTool(BaseTool):
    """Look through the vector store to answer general questions."""

    name = "SearchVDABKnowledgeBase"
    description = (
        "useful for when you need to answer general questions about VDAB."
        "input should be a user question, leave the original question intact."
        "if no relevant answer is found, you should politely make clear that you don't know."
        "don't add info to the answer that is not in the snippet."
        "Be as concise possible and answer the question in the shortest way possible."
        "do not provide extra info unless specifically asked for."
        "the context you'll be using is either based on internal info or info freely found on the VDAB website. The internal info is formatted in question-answer pairs. "
        "the type of info is indicated by '<type>: ' followed by the info."
        "you'll use this info to correctly answer the question."
        "you don't have to inform the user about the format of the info."
    )
    websocket: WebSocket | None
    retriever: ParentDocumentRetriever

    def _run(
        self,
        query: str,
    ) -> str:
        """Use the tool."""
        pass

    async def _arun(
        self,
        query: str,
    ) -> str:
        """Use the tool asynchronously."""
        if not self.retriever:
            raise ValueError("Vector store not loaded")
        docs = self.retriever.get_relevant_documents(query)
        
        doc_links = [(doc.metadata.get("url", ""), doc.metadata.get("title", "link not found"))  for doc in docs]
        print([doc.metadata.keys() for doc in docs])

        link_string = "See the following relevant links: <br><br>"
        for link, title in doc_links:
            print(link)
            print(title)
            new_link = link.replace("-rel1ldv.ops", "")
            link_string += (
                '<a href="' + new_link + '" target="_blank">' + new_link + "</a><br>"
            )
        end_resp = ChatResponse(sender="bot", message="", type="end")
        await self.websocket.send_json(end_resp.dict())

        bot_resp = ChatResponse(sender="bot", message=link_string, type="stream")
        await self.websocket.send_json(bot_resp.dict())

        end_resp = ChatResponse(sender="bot", message="", type="end")
        await self.websocket.send_json(end_resp.dict())

        start_resp = ChatResponse(sender="bot", message="", type="start")
        await self.websocket.send_json(start_resp.dict())
        
        return "".join(doc.page_content + "\n" for doc in docs)
    
        
class ContactInfoTool(BaseTool):
    """Return contact information VDAB when no more answers can be found."""

    name = "ContactInfo"
    description = (
        "useful for when the question is about anything else than VDAB."
        "Even if you know the answer, if the question is not about VDAB, use this tool."
    )

    def _run(
        self,
        query: str,
    ) -> str:
        """Use the tool."""
        pass

    
    async def _arun(self, query: str,) -> str:
        """Use the tool."""
        return """
            You should send this to the user now: 
            'I am not sure I can help you further. It may be best to contact VDAB directly.
            The service number of VDAB is 0800 30 700. It is available everyday from 8am until 4h30pm.
            You can also chat with a person at the VDAB site https://www.vdab.be/contact.
            More information can also be found on the VDAB site. 
            You can translate this site to your own language by following the steps on this site:
            https://www.vdab.be/anderstalig-hoogopgeleid'
        """

        
class PDFParserTool(BaseTool):
    """Return the information present in the pdf."""

    name = "PDFParser"
    description = (
        "use this when a pdf is uploaded"
    )

    def _run(
        self,
        query: str,
    ) -> str:
        """Use the tool."""
        pass

    async def _arun(self, query: str) -> str:
        """Use the tool"""
        return """
Ons kenmerk:
ASB/1/document
 
Hallo Mina, 
We hebben een overzicht gemaakt van je gesprekken en opdrachten bij VDAB of één van onze partners.
In bijlage vind je je nieuwe (of gewijzigde) gesprekken en opdrachten. Alle gesprekken en opdrachten 
zijn terug te vinden in je online account op vdab.be, net als dit document.  
Jouw bemiddelingstraject
● Bemiddeling bij het zoeken naar werk 
Om je zo goed mogelijk te begeleiden in het zoeken naar werk, volgt je bemiddelaar je 
verder op. Tijdens jullie gesprekken bekijken jullie hoe het zoeken naar werk verloopt. Als 
je problemen ondervindt, helpt je bemiddelaar bij het zoeken naar oplossingen. 
Gesprekken
OKT 
30  Gesprek met je bemiddelaar meer info in afsprakenbundel
 - VDAB-kantoor  17-10-2023
10u00 
Gesprekken en opdrachten zijn verplicht. 
Je moet je aan deze plichten houden. Kom je de gesprekken niet na en/of voer je de opdrachten niet 
uit, dan kan dit gevolgen hebben. Meer info op vdab.be/rechtenenplichten.
We verwachten dat je zelf actief naar werk zoekt. 
Het is belangrijk dat je op elk moment kan aantonen dat je actief naar werk zoekt. Hou je sollicitatie-
acties goed bij. Je kan hiervoor je online account gebruiken. Gebruik je hiervoor je account niet, hou 
dan je sollicitatiebewijzen of andere documenten die aantonen dat je actief naar werk zoekt goed bij. 
Ontvang je een bedrijfstoeslag (brugpensioen)? Dan gelden er andere regels, meer info op 
vdab.be/rechtenenplichten/plus.
1/2
Klantnummer:
60004022 
Datum:
17-10-2023 
Vriendelijke groeten 
VDAB
Gesprekken zijn opvolgingsgesprekken zoals bedoeld in het Besluit van de Vlaamse Regering van 5 juni 2009 houdende de organisatie van de 
arbeidsbemiddeling en de beroepsopleiding. Opdrachten zijn afspraken zoals bedoeld in datzelfde besluit.
Plaats hieronder je handtekening als je bemiddelaar het vraagt. 
Opgemaakt op dinsdag 17 oktober 2023 om 12:27.
Handtekening werkzoekende Handtekening bemiddelaar
Plaats 
je handtekening 
in het voor jou 
voorziene vakje
Mina Mina    
2/2"""
    # async def _arun(self, query: str,) -> str:
    #     """Use the tool."""
    #     information = ""
    #     with pdfplumber.open("tmp/upload.pdf") as pdf:
    #         nb_pages = pdf.pages
    #         for page in nb_pages:
    #             information += page.extract_text_simple(x_tolerance=3, y_tolerance=3) + "\n"
    #     output = "This is all information that can be found in the pdf document: " + information
    #     return output


# class ImageTool(BaseTool):
#     """Return all information that can be found in the given images."""

#     name = "Image"
#     description = (
#         "When an image is uploaded, use this tool"
#     )

#     def _run(
#         self,
#         query: str,
#     ) -> str:
#         """Use the tool."""
#         pass

    
#     async def _arun(self,) -> str:
#         """Use the tool."""
#         llava_model = self.load_llava_model()
#         question = """
#             This image is uploaded in the context of VDAB, which is a Belgian 
#             service to help people get a job. 
#             The user has a problem with understanding the information on this image. 
#             Please tell me all information you can find in this image.
#             DO NOT return any personal information like personal names, adresses, phone numbers, ...
#             """
#         path = "tmp/images/"
#         #images = TODO load image 

#         if len(images) == 0: 
#             return "No image was uploaded, please solve this using a different tool"
#         elif len(images) == 1: 
#             loaded_image = self.load_image(images[0])
#             llava_answer = self.ask_question(llava_model, loaded_image, question)
#             return "What I understand from this image is: " + llava_answer
#         else:
#             response = ""
#             for idx, image in enumerate(images): 
#                 response += "What I understand from image " + str(idx) + " is: "
#                 loaded_image = self.load_image(image)
#                 llava_answer = self.ask_question(llava_model, loaded_image, question)

#                 response += llava_answer + ".\n "
#             return response
    
#     # Define load model and load image functions.
#     @lru_cache
#     def load_llava_model(model_path="radix-ai/llava-v1.5-7b"):
#         # The original liuhaotian/llava-v1.5-7b has a shard size of 10GB which
#         # causes an out of memory error on Colab [1]. To fix this, we uploaded the
#         # model with 2GB shards to radix-ai/llava-v1.5-7b. Larger versions of this
#         # model are also available [2].
#         #
#         # [1] https://github.com/haotian-liu/LLaVA/issues/496
#         # [2] https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md.
#         tokenizer, model, image_processor, context_len = load_pretrained_model(
#             model_path=model_path,
#             model_base=None,
#             model_name=model_path.split("/")[-1],
#             load_8bit=True,  # Quantize to 8 bit to fit on Google Colab's T4 GPU.
#             load_4bit=False
#         )
#         return tokenizer, model, image_processor, context_len

#     @lru_cache
#     def load_image(image_url):
#         # image_filename, _ = urllib.request.urlretrieve(image_url)
#         image = Image.open(image_url).convert("RGB")
#         return image
    
#     def ask_question(model, image, question):
#         # Unpack model.
#         tokenizer, model, image_processor, context_len = model
#         disable_torch_init()
#         # Convert image.
#         image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().cuda()
#         # Generate prompt.
#         conv_mode = "llava_v1"
#         conv = conv_templates[conv_mode].copy()
#         roles = conv.roles
#         inp = DEFAULT_IMAGE_TOKEN + "\n" + question
#         conv.append_message(conv.roles[0], inp)
#         conv.append_message(conv.roles[1], None)
#         prompt = conv.get_prompt()
#         input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
#         stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
#         keywords = [stop_str]
#         stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
#         streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
#         # Inference.
#         with torch.inference_mode():
#             output_ids = model.generate(
#                 input_ids,
#                 images=image_tensor,
#                 do_sample=True,
#                 temperature=0.2,
#                 max_new_tokens=1024,
#                 streamer=streamer,
#                 use_cache=True,
#                 stopping_criteria=[stopping_criteria]
#             )
#         outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
#         return outputs
