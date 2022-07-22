import streamlit as st
from utils import django_setup


from base.models import Job

st.write(
    """
# Jobs progresses
"""
)


for job in reversed(Job.objects.all()[Job.objects.count() - 100 :]):
    progress = job.progress
    st.write(str(job))
    progress_bar = st.progress(progress)
