"""
Application entrypoint.

Registers the app's pages with `st.navigation`. The navigation widget is hidden
(`position="hidden"`) because pages are reached through the top bar links and
the About page's "Back to the application" link. Page scripts live next to this
file:

- `home.py`  : the main app (welcome screen + dataset dashboard)
- `about.py` : the standalone "About the Author" page, served at /about
"""

import streamlit as st

home_page = st.Page("home.py", title="Home", icon="🧬", default=True)
about_page = st.Page("about.py", title="About the Author", icon=":material/person:")

pg = st.navigation([home_page, about_page], position="hidden")
pg.run()
