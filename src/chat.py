import streamlit as st

from utils.streamlit_utils import StreamlitUtils

st.set_page_config(layout="centered", page_title="My Streamlit App")
st.title("R-COP Classifier")
st.caption("Singapore Statues Online | https://sso.agc.gov.sg//Act/PC1871")


if "reasoning_mode" not in st.session_state:
    st.session_state.reasoning_mode = False

if "last_response" not in st.session_state:
    st.session_state.last_response = None


def main():
    with st.sidebar:
        st.toggle(
            label="🧠 View Reasoning",
            value=st.session_state.reasoning_mode,
            key="reasoning_mode",
        )

    st_util = StreamlitUtils()

    form = st.form(key="form")
    query = form.text_area("Facts of Case", height=200)

    if form.form_submit_button("Submit"):
        st.session_state.last_reasoning_mode = st.session_state.reasoning_mode
        st_util.chat(query)

    if st.session_state.last_response:
        st_util.render_response(st.session_state.last_response)


if __name__ == "__main__":
    main()
