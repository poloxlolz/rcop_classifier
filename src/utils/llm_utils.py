import re
from collections import defaultdict

import pymupdf
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import utils.common_utils as common_utils
import utils.prompt_templates as prompt_templates
from models.documents import DocumentMetadata

CORPUS = "Singapore Statues Online"
CORPUS_DIR = common_utils.get_project_root() / CORPUS


class LLM_Utils:
    def __init__(
        self, llm_model="gpt-5-mini", embedding_model="text-embedding-3-small"
    ):
        self.llm_model = llm_model
        self.embedding_model = embedding_model

        self.llm = ChatOpenAI(model=llm_model)
        self.embedding = OpenAIEmbeddings(model=embedding_model)

        self.collection_name = "sso_vectorstore"

        self.load_vectorstore() if (
            common_utils.get_project_root() / "sso_vectorstore"
        ).exists() else self.create_vectorstore()

        self.create_cohere_retriever()
        self.create_rag_chain()

    # @staticmethod
    # def build_corpus_json():
    #     hierarchy = defaultdict(
    #         lambda: defaultdict(list)
    #     )  # Chapter/Part → list or Heading → list

    #     for file in CORPUS_DIR.rglob("*.pdf"):
    #         parts = file.relative_to(CORPUS_DIR).parts

    #         act = parts[0]
    #         part_or_chapter = parts[1] if len(parts) > 1 else None
    #         section_name = file.stem

    #         # Only treat as heading if there are more than 3 parts
    #         if len(parts) > 3:
    #             heading = parts[2]
    #             if part_or_chapter not in hierarchy[act]:
    #                 hierarchy[act][part_or_chapter] = defaultdict(list)
    #             hierarchy[act][part_or_chapter][heading].append(
    #                 {"section": section_name, "file": str(file)}
    #             )
    #         else:
    #             # No heading → append directly to list
    #             if part_or_chapter not in hierarchy[act]:
    #                 hierarchy[act][part_or_chapter] = []
    #             hierarchy[act][part_or_chapter].append(
    #                 {"section": section_name, "file": str(file)}
    #             )

    #     # Convert defaultdicts to normal dicts for JSON
    #     def convert(d):
    #         if isinstance(d, defaultdict):
    #             return {k: convert(v) for k, v in d.items()}
    #         elif isinstance(d, list):
    #             return list(d)
    #         else:
    #             return d

    #     return convert(hierarchy)

    def create_documents(self) -> list[Document]:
        # TO-DO : Parse out document headers
        documents = []

        for file in CORPUS_DIR.rglob("*.pdf"):
            act, part, *part_hdr_ita, _ = file.relative_to(CORPUS_DIR).parts

            match = re.match(
                pattern=r"^(?:Part|Chapter) (?P<part_no>\d+)(?:\s*—\s*|\s+)(?P<part_hdr>.*)$",
                string=part,
            )

            provision_no, provision_hdr = file.stem.split(maxsplit=1)

            doc_metadata = DocumentMetadata(
                corpus=CORPUS,
                act=act,
                part_no=match.group("part_no"),
                part_hdr=match.group("part_hdr"),
                part_hdr_ita=part_hdr_ita[0] if part_hdr_ita else None,
                provision_no=provision_no,
                provision_hdr=provision_hdr,
                ext=file.suffix,
                source=str(CORPUS / file.relative_to(CORPUS_DIR)),
            )

            with pymupdf.open(filename=file) as doc:
                text = chr(12).join([page.get_text() for page in doc])

                footer = r"Singapore Statutes Online\nCurrent version as at .*?\nPDF created date on: .*?\n"
                amendments = r"\[.*?\]|\[.*?\n"
                page_break = r"\x0c)"

                pattern = f"(?:{footer}|{amendments}|{page_break}"

                cleaned_text: str = re.sub(pattern=pattern, repl="", string=text)

                documents.append(
                    Document(
                        page_content=cleaned_text, metadata=doc_metadata.model_dump()
                    )
                )

        return documents

    def create_vectorstore(self) -> None:
        self.vectorstore = Chroma.from_documents(
            documents=self.create_documents(),
            embedding=OpenAIEmbeddings(model=self.embedding_model),
            collection_name=self.collection_name,
            persist_directory=common_utils.get_project_root() / self.collection_name,
        )

    def load_vectorstore(self) -> None:
        self.vectorstore = Chroma(
            embedding_function=OpenAIEmbeddings(model=self.embedding_model),
            collection_name=self.collection_name,
            persist_directory=common_utils.get_project_root() / self.collection_name,
        )

    def create_cohere_retriever(self):
        self.compressor = ContextualCompressionRetriever(
            base_compressor=CohereRerank(top_n=5, model="rerank-english-v3.0"),
            base_retriever=self.vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 5}
            ),
        )

    def create_rag_chain(self, reasoning_mode: bool = False):
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.compressor,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt_templates.get_classification_prompt(
                    reasoning_mode=reasoning_mode
                )
            },
        )

    def refresh_rag_chain(self, reasoning_mode: bool = False):
        self.create_rag_chain(reasoning_mode=reasoning_mode)

    def invoke(self, query):
        return self.rag_chain.invoke(query)
