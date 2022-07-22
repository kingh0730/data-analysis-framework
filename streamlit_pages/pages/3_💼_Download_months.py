import streamlit as st
from utils import django_setup


from base.models import (
    DownloadOneMonth,
    Job,
    Month,
)


st.write(
    """
# Download months
"""
)


with st.sidebar:
    months = st.multiselect(
        "Months",
        Month.objects.all(),
    )


with st.echo():

    def main() -> list[Job]:
        result = [
            DownloadOneMonth.objects.create(month=month_instance)
            for month_instance in months
        ]

        return result


if st.button("Execute main()"):
    st.info("Executing...")
    st.warning("Check terminal.")
    st.success(f"Success: {main()}")
