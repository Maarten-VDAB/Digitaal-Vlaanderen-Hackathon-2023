"""Define tools for the LLM to use."""

from langchain.retrievers import ParentDocumentRetriever
from langchain.tools import BaseTool
from fastapi import WebSocket

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
        # TODO: reenable this to show links
        # doc_links_and_titles = [
        #     (doc.metadata.get("url", ""), doc.metadata.get("title", ""))
        #     for doc in docs
        #     if doc.metadata.get("url")
        # ]
        # # only keep the tuples if the title is unique
        # doc_links_and_titles = list(
        #     {title: link for link, title in doc_links_and_titles}.items()
        # )
        # # swap back to link, title
        # doc_links_and_titles = [(link, title) for title, link in doc_links_and_titles]

        # link_string = "See the following relevant links: <br><br>"
        # for link, title in doc_links_and_titles:
        #     link_string += (
        #         '<a href="' + link + '" target="_blank">' + title + "</a><br>"
            # )
        # end_resp = ChatResponse(sender="bot", message="", type="end")
        # await self.websocket.send_json(end_resp.dict())

        # bot_resp = ChatResponse(sender="bot", message=link_string, type="stream")
        # await self.websocket.send_json(bot_resp.dict())

        # end_resp = ChatResponse(sender="bot", message="", type="end")
        # await self.websocket.send_json(end_resp.dict())

        # start_resp = ChatResponse(sender="bot", message="", type="start")
        # await self.websocket.send_json(start_resp.dict())
        return "".join(doc.page_content + "\n" for doc in docs)