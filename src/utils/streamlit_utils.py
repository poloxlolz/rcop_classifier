import json
import re

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils.copywriting_utils import Copies
from utils.llm_utils import LLM_Utils

load_dotenv()


@st.cache_resource(show_spinner=False)
def get_llm_instance():
    return LLM_Utils()


class StreamlitUtils:
    def __init__(self):
        self.llm = get_llm_instance()

    def chat(self, query: str):
        with st.spinner(text=Copies.SPINNER.value, show_time=True):
            response = self.llm.rag_chain.invoke(query)

            json_response = json.loads(response["result"])
            citation = json_response["final_classification"]

            match = re.search(pattern=r"s\.([0-9]+[A-Za-z]?)", string=citation)

            cited_section = match.group(1) if match else citation

            rows = []
            act = ""

            for doc in response["source_documents"]:
                if match and cited_section == doc.metadata["section"]:
                    act = f" {doc.metadata['act']}"

                row = dict(doc.metadata)
                rows.append(row)

            df = pd.DataFrame(rows)
            df = df[
                [
                    "relevance_score",
                    "act",
                    "part_or_chapter",
                    "part_or_chapter_heading",
                    "section",
                    "section_heading",
                ]
            ]

            st.caption("⚖️ Final Classification")
            st.write(f"**{citation}{act}**")

            st.caption("Confidence Level")
            st.write(f"**{json_response['confidence_level']}**")

            st.caption("Retrieved Documents")
            st.dataframe(df)
