"""
About the Author page, served at /about.

This is a standalone page of the multi-page Streamlit app, so the author
profile can be opened on its own (in a new browser tab) without loading the
dataset dashboard.
"""

import streamlit as st

st.set_page_config(page_title="About the Author - BioData", page_icon="🧬", layout="wide")

LOGO_PATH = "app/logo/logo.png"
st.logo(LOGO_PATH, size="large", icon_image=LOGO_PATH)

# --- AUTHOR INFORMATION ---
GITHUB_URL = "https://github.com/manaves"
LINKEDIN_URL = "https://www.linkedin.com/in/maria-navarro-paredes/"

AUTHOR_NAME = "Maria Navarro Paredes"
AUTHOR_EMAIL = "navarroparedesmaria@gmail.com"
AUTHOR_ROLE = "Developer & maintainer of BioData"
AUTHOR_BIO = (
    "Hi! I'm Maria and I'm the developer behind BioData. I built this application to make quality "
    "control and structural inspection of protein–protein binding data more "
    "approachable, combining thermodynamic calculations, statistical outlier "
    "detection and interactive 3D visualization in a single workflow.\n\n"
    "I have a background in bioinformatics and biotechnology, and I enjoy creating tools that help "
    "researchers and students explore and analyze biological data more effectively. If you want to "
    "get in touch, feel free to reach out via GitHub, LinkedIn or email!"
)

INITIALS = "".join(part[0] for part in AUTHOR_NAME.split() if part)[:2].upper()

ABOUT_AUTHOR_STYLE = """
<style>
.about-author {
    display: flex; gap: 1.5rem; align-items: flex-start;
    padding: 1.25rem 1.5rem;
    border: 1px solid color-mix(in srgb, currentColor 18%, transparent);
    border-radius: 0.75rem;
    background: color-mix(in srgb, var(--secondary-background-color, #808080) 25%, transparent);
}
.about-author__avatar {
    position: relative; overflow: hidden;
    flex: 0 0 auto; width: 4.5rem; height: 4.5rem; border-radius: 50%;
    background: var(--primary-color, #FF4B4B); color: #ffffff;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem; font-weight: 600; letter-spacing: 0.02em;
}
.about-author__initials {
    position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
}
.about-author__avatar img {
    position: absolute; inset: 0; width: 100%; height: 100%;
    object-fit: cover; border-radius: 50%; display: block; color: transparent;
}
.about-author__body h3 { margin: 0 0 0.15rem 0; }
.about-author__role { margin: 0 0 0.6rem 0; opacity: 0.65; font-size: 0.85rem; }
.about-author__bio { margin: 0 0 0.85rem 0; }
.about-author__links { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.about-author__links a {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.35rem 0.75rem; border-radius: 999px;
    border: 1px solid color-mix(in srgb, currentColor 30%, transparent);
    background: transparent; color: inherit;
    text-decoration: none; font-size: 0.85rem;
    transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
}
.about-author__links a:hover {
    background: var(--primary-color, #FF4B4B);
    border-color: var(--primary-color, #FF4B4B);
    color: #ffffff;
}
</style>
"""

ABOUT_AUTHOR_CARD = f"""
<div class="about-author">
    <div class="about-author__avatar">
        <span class="about-author__initials">{INITIALS}</span>
        <img src="https://shorturl.at/xw2MI" alt="{AUTHOR_NAME}" loading="lazy" referrerpolicy="no-referrer"
             onerror="this.style.display='none'">
    </div>
  <div class="about-author__body">
    <h3>{AUTHOR_NAME}</h3>
    <p class="about-author__role">{AUTHOR_ROLE}</p>
    <p class="about-author__bio">{AUTHOR_BIO}</p>
    <div class="about-author__links">
      <a href="{GITHUB_URL}" target="_blank" rel="noopener noreferrer">GitHub</a>
      <a href="{LINKEDIN_URL}" target="_blank" rel="noopener noreferrer">LinkedIn</a>
      <a href="mailto:{AUTHOR_EMAIL}">Email</a>
    </div>
  </div>
</div>
"""

st.title("About the Author")

st.markdown(ABOUT_AUTHOR_STYLE + ABOUT_AUTHOR_CARD, unsafe_allow_html=True)

st.divider()
