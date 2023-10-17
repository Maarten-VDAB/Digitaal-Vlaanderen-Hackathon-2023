"""Define tools for the LLM to use."""
from langchain.retrievers import ParentDocumentRetriever
from langchain.tools import BaseTool
from fastapi import WebSocket

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
            I am not sure I can help you further. It may be best to contact VDAB directly.
            The service number of VDAB is 0800 30 700. It is available everyday from 8am until 4h30pm.
            You can also chat with a person at the VDAB site https://www.vdab.be/contact.
            More information can also be found on the VDAB site. 
            You can translate this site to your own language by following these steps:
            TODO.
        """

