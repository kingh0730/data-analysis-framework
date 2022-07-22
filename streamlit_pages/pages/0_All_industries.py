from typing import Any
import streamlit as st
from techniques.models import InputToOutputDocument
from utils import EIGHTY_TWO_INDUSTRIES, django_setup


from base.models import (
    Month,
)


st.write(
    """
# All industries
"""
)


with st.sidebar:
    month = st.selectbox(
        "Month",
        Month.objects.all(),
    )


with st.echo():

    def main() -> None:
        for daas_id in EIGHTY_TWO_INDUSTRIES["daas_id"]:
            try:
                InputToOutputDocument.objects.create(
                    index_id=0,
                    gov_id=2273,
                    month=month,
                    sample_selection_kwargs={"daas_id": daas_id},
                )
            except ValueError:
                # Todo: write
                st.warning(daas_id)


if st.button("Execute main()"):
    st.info("Executing...")
    st.warning("Check terminal.")
    st.success(f"Success: {main()}")
