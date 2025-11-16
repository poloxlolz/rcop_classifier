import streamlit as st

from utils.streamlit_utils import StreamlitUtils

st.set_page_config(page_title="SSO Knowledge Base", layout="centered")
st.title("🏛️ Bot Knowledge Base")
st_util = StreamlitUtils().llm


def render_clickable_hierarchy(nested_json):
    for act, chapters in nested_json.items():
        st.subheader(act)  # Act
        for chapter, headings in chapters.items():
            st.markdown(f"**{chapter}**")  # Chapter/Part

            if isinstance(headings, dict):
                # Has heading
                for heading, sections in headings.items():
                    st.markdown(f"_{heading}_")  # Heading italic
                    for sec in sections:
                        st.markdown(f"- {sec['section']}")
            else:
                # No heading, sections is a list
                for sec in headings:
                    st.markdown(f"- {sec['section']}")


# Display in Streamlit
render_clickable_hierarchy(st_util.build_corpus_json())
