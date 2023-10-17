from typing import Any
from pathlib import Path
from typing import Sequence
import csv
import pickle
import ast

from langchain.document_loaders.base import BaseLoader
from langchain.schema.document import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

from constants import DB_PATH, DATA_PATH

import ast

class VDABRetrieverClient:
    def __init__(self, chunk_size: int | None = 300, chunk_overlap: int | None = 0):
        self.documents = self._load_documents()
        self.splitter = self._load_splitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.vectorstore = None
        self.documentstore = None

    def load_retriever(
        self,
        search_type: str | None = "similarity_score_threshold",
        threshold: float | None = 0.3,
        filter: dict[str, Any] | None = {},
        k: int | None = 3,
    ) -> ParentDocumentRetriever:
        with open(DB_PATH / "store.pickle", "rb") as f:
            documentstore = pickle.load(f)
        vectorstore = Chroma(
            collection_name="full_documents",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=f"{DB_PATH}/chroma_db",
        )
        return ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=documentstore,
            child_splitter=self.splitter,
            search_kwargs={
                "search_type": search_type,
                "threshold": threshold,
                "filter": filter or {},
                "k": k,
            },
        )

    def build_retriever(
        self,
        search_type: str | None = "similarity_score_threshold",
        threshold: float | None = 0.3,
        filter: dict[str, Any] | None = {},
        k: int | None = 3,
    ) -> None:
        vectorstore = Chroma(
            collection_name="full_documents",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=f"{DB_PATH}/chroma_db",
        )
        documentstore = InMemoryStore()
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=documentstore,
            child_splitter=self.splitter,
            search_kwargs={
                "search_type": search_type,
                "threshold": threshold,
                "filter": filter or {},
                "k": k,
            },
        )
        retriever.add_documents(self.documents, ids=None)

        # persist document store
        with open(DB_PATH / "store.pickle", "wb") as f:
            pickle.dump(documentstore, f)

    def _load_vectorstore(self):
        vectorstore = Chroma(
            collection_name="full_documents",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=f"{DB_PATH}/chroma_db",
        )
        self.vectorstore = vectorstore

    def _load_documentstore(self):
        # self.documentstore = InMemoryStore()
        with open(DB_PATH / "store.pickle", "rb") as f:
            self.documentstore = pickle.load(f)

    def _load_documents(self):
        bac_loader = WebLoader(
            file_path=DATA_PATH / "web_cleaned.csv",
            text_columns=["title_translation", "content_translation"],
            columns=["url"],
            csv_args={"quoting": csv.QUOTE_ALL},
        )
        shops_loader = IntranetLoader(
            file_path=DATA_PATH / "intranet_cleaned.csv",
            text_columns=[
                "title_translation",
                "content_translation",
            ],
            columns=["url"],
            csv_args={"quoting": csv.QUOTE_ALL},
        )
        documents_shops = shops_loader.load()
        documents_website = bac_loader.load()
        docs = documents_website + documents_shops
        return docs

    def _load_splitter(self, chunk_size: int, chunk_overlap: int):
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

class IntranetLoader(BaseLoader):
    def __init__(
        self,
        file_path: Path,
        text_columns: list[str],
        columns: Sequence[str] | None = None,
        csv_args: dict[str, Any] | None = None,
    ):
        """Custom loader for the VDAB web dataset.

        Args:
        ----
        file_path
            Path to the csv file.
        text_column
            Name of the column containing the text.
        columns
            Additional columns to include in the dataset. If None, no additional columns are loaded.
        csv_args
            Arguments to pass to csv.DictReader.
        """
        self.file_path = file_path
        self.text_columns = text_columns
        self.columns = columns
        self.csv_args = csv_args

    def load(self) -> list[Document]:
        """Load data into document objects."""
        csv.field_size_limit(10485760)

        docs = []
        with open(self.file_path, newline="") as file:
            csv_reader = csv.DictReader(file, **self.csv_args)
            for row in csv_reader:
                content_strings = []
                for col in self.text_columns:
                    if col == "content_translation":
                        try:
                            question_answer = " ".join(f'Question: {q} \n Answer: {a}' for q, a in ast.literal_eval(row[col]))
                            content_strings.append(f"Question answer pairs of the document: {question_answer}")
                        except:
                            print("failed")
                    else:
                        content = f'{col.replace("_translation", "")} of the document: {row[col]}'.replace("\n", "")
                        content_strings.append(content)
                content = "\n".join(content_strings)         
                try:
                    metadata = {k: v for k, v in row.items() if k in self.columns}
                except KeyError as e:
                    raise ValueError(
                        f"Columns {self.columns} not found in file {self.file_path}."
                    ) from e
                docs.append(Document(page_content=content, metadata=metadata))
        return docs


class WebLoader(BaseLoader):
    """Custom loader for the VDAB web dataset."""
    def __init__(
        self,
        file_path: Path,
        text_columns: list[str],
        columns: Sequence[str] | None = None,
        csv_args: dict[str, Any] | None = None,
    ):
        """Custom loader for the VDAB web dataset.

        Args:
        ----
        file_path
            Path to the csv file.
        text_column
            Name of the column containing the text.
        columns
            Additional columns to include in the dataset. If None, no additional columns are loaded.
        csv_args
            Arguments to pass to csv.DictReader.
        """
        self.file_path = file_path
        self.text_columns = text_columns
        self.columns = columns
        self.csv_args = csv_args

    def load(self) -> list[Document]:
        """Load data into document objects."""
        csv.field_size_limit(10485760)

        docs = []
        with open(self.file_path, newline="") as file:
            csv_reader = csv.DictReader(file, **self.csv_args)
            for row in csv_reader:
                content = "\n".join(
                    f'{col.replace("_translation", "")} of the document: {row[col]}'.replace("\n", "")
                    for col in self.text_columns
                )
                try:
                    metadata = {k: v for k, v in row.items() if k in self.columns}
                except KeyError as e:
                    raise ValueError(
                        f"Columns {self.columns} not found in file {self.file_path}."
                    ) from e
                docs.append(Document(page_content=content, metadata=metadata))
        return docs
