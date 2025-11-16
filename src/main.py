import streamlit as st

from utils.streamlit_utils import StreamlitUtils

st.set_page_config(layout="centered", page_title="My Streamlit App")
st.title("R-COP Classifier")
st.caption("Singapore Statues Online | https://sso.agc.gov.sg//Act/PC1871")


def main():
    st_util = StreamlitUtils()

    form = st.form(key="form")
    query = form.text_area("Facts of Case", height=200)

    if form.form_submit_button("Submit"):
        st_util.chat(query)


if __name__ == "__main__":
    main()
