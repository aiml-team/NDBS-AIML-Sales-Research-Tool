import os
import re
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_tavily import TavilySearch
from langchain_community.utilities import SerpAPIWrapper
from fill_template import fill_word_template

# -------------------------
# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# -------------------------
def google_search(query):
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    for g in soup.find_all('div', class_='tF2Cxc'):
        link = g.find('a')['href']
        return link
    return None

# -------------------------
def scrape_company_website(company_name):
    info = {k: "" for k in [
        "company_name", "address", "employee_count", "annual_revenue", "leadership_changes",
        "recent_news", "recent_funding", "current_erp", "recent_sap_job_postings",
        "phone_number", "sic_codes", "company_official_website",
        "strengths", "weaknesses", "opportunities", "threats"]}
    info["company_name"] = company_name
    info["company_official_website"] = google_search(f"{company_name} official site") or ""

    try:
        if info["company_official_website"]:
            response = requests.get(info["company_official_website"], timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)

            patterns = {
                "phone_number": r'(\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9})',
                "address": r'\d{1,5}\s[\w\s.,-]+,\s\w+,\s[A-Z]{2}\s\d{5}(-\d{4})?',
                "employee_count": r'([0-9,]+)\s+(employees|staff|workers|team)',
                "annual_revenue": r'(revenue|sales|turnover)[\s\w]{0,20}?\$?([\d,.]+)\s?(million|billion)?',
                "sic_codes": r'SIC Code[:\s]*([\d]{4})'
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, text, re.I)
                if match:
                    value = match.group(2).replace(',', '') if key == "annual_revenue" else match.group(1)
                    info[key] = f"${value} {match.group(3)}".strip() if key == "annual_revenue" else value

            keywords = {
                "leadership_changes": ['ceo', 'appointed', 'joined', 'leadership'],
                "recent_news": ['news', 'announcement', 'press release'],
                "strengths": ['strength'],
                "weaknesses": ['weakness'],
                "opportunities": ['opportunit'],
                "threats": ['threat']
            }
            for key, kwds in keywords.items():
                snippets = [line.strip() for line in text.split('.') if any(k in line.lower() for k in kwds)]
                info[key] = ' '.join(snippets[:3]) or "Not Available"

            for erp in ['SAP', 'Oracle ERP', 'Microsoft Dynamics', 'NetSuite', 'Infor']:
                if erp.lower() in text.lower():
                    info["current_erp"] = erp
                    break

            postings = [a.get_text(strip=True) for a in soup.find_all('a')
                        if any(k in a.get_text(strip=True).lower() for k in ['sap', 'erp'])]
            info["recent_sap_job_postings"] = ', '.join(postings) or "No SAP job postings found"

    except Exception as e:
        print(f"Scraping error: {e}")
    return info

# -------------------------
llm = AzureChatOpenAI(deployment_name="gpt-4o", model_name="gpt-4o", temperature=0.7)

# -------------------------
prompt_template = PromptTemplate(
    input_variables=["company_name", "scraped_data"],
    template="""
You are a business intelligence assistant creating a report on **{company_name}**.

Return a fact-based, **plain text** report with no markdown formatting but with proper alignment. Use no asterisks (*) or hashtags (#) in the final document.
Use the **Tavily Search tool** to enrich any missing information & give me more descriptive answers.

**Company Report**

## Company Fundamentals
- **Company Name:** {company_name}
- **Size:** {scraped_data[employee_count]}
- **Annual Revenue:** {scraped_data[annual_revenue]}
- **Industry Classification:** {scraped_data[sic_codes]}
- **Business Model:** Not Available (refer to {scraped_data[company_official_website]})
- **Geographic Presence:** Not Available (refer to {scraped_data[company_official_website]})
- **Ownership:** Not Available (refer to Crunchbase or Bloomberg)

## Financial Health & Performance
- **Recent Financials:** {scraped_data[annual_revenue]}
- **Stability Indicators:** Not Available (refer to investor reports or 10-K)
- **Capital Investments:** {scraped_data[recent_funding]}
- **Stock Performance:** Not Available (check Google Finance or Yahoo Finance)

## Products, Operations & Technology
- **Core Offerings:** Not Available (check company website)
- **ERP System:** {scraped_data[current_erp]}
- **Technology Stack:** Not Available (refer to job postings or CIO LinkedIn)

## Leadership & Governance
- **Executive Team:** {scraped_data[leadership_changes]}
- **Board of Directors:** Not Available (refer to official site or Crunchbase)
- **Leadership Strategy:** Not Available (refer to press releases/interviews)

## Strategic Initiatives & Challenges
- **Growth Priorities:** Not Available (check investor presentations)
- **Digital Initiatives:** {scraped_data[current_erp]}
- **Challenges Identified:** {scraped_data[weaknesses]}, {scraped_data[threats]}

## Market Context & Competitors
- **Recent News:** {scraped_data[recent_news]}
- **Competitive Landscape:** Not Available (use Tavily or Crunchbase)
- **Industry Trends:** Not Available (check news and analyst reports)

## SAP-Relevant Signals
- **Recent SAP Job Postings:** {scraped_data[recent_sap_job_postings]}
- **Integration Maturity:** Not Available (check LinkedIn/job roles)
- **Tech Budget Indicators:** Not Available (refer to earnings calls)

## SWOT Analysis
- **Strengths:** {scraped_data[strengths]}
- **Weaknesses:** {scraped_data[weaknesses]}
- **Opportunities:** {scraped_data[opportunities]}
- **Threats:** {scraped_data[threats]}

## Contact Information
- **Phone:** {scraped_data[phone_number]}
- **Address:** {scraped_data[address]}
- **Official Website:** {scraped_data[company_official_website]}

## Disclaimer
Some data may be incomplete or outdated. For the most accurate and timely information, please verify through the company's official website, investor relations, or public disclosures.
"""
)

def generate_summary(company_name, scraped_data):
    try:
        prompt = prompt_template.format(company_name=company_name, scraped_data=scraped_data)
        return llm.invoke(prompt).content.strip()
    except Exception as e:
        st.error(f"Summary generation error: {e}")
        return "Summary generation failed."

# -------------------------
# Streamlit UI

st.set_page_config(page_title="AI Sales Research", page_icon="🤖", layout="wide")
logo = Image.open("Logo-White.png")

# --- CSS Styling ---
st.markdown(
    """
    <style>
    .top-right {
        position: absolute;
        top: 12px;
        right: 12px;
        z-index: 9999;
    }
    .stApp, .block-container {
        background-color: #070f26;
    }
    section[data-testid="stSidebar"] {
        background-color: #1e458e !important;
    }
    div.stChatInputContainer {
        background-color: #1e458e !important;
        border-top: 1px solid #133168;
    }
    div.stChatInputContainer input {
        background-color: #133168 !important;
        color: white !important;
    }
    button[kind="secondary"] {
        background-color: #133168 !important;
        color: white !important;
    }
    body, .markdown-text-container, .stTextInput>div>div>input, .css-qrbaxs {
        color: white !important;
    }
    button {
        background-color: #133168 !important;
        color: white !important;
        border: none;
    }
    .css-1kyxreq {
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar: Logo + Title + Search History ---
with st.sidebar:
    st.image(logo, width=250)
    st.sidebar.title("Search History")

# --- Initialize session state ---
if "search_history" not in st.session_state:
    st.session_state["search_history"] = []
if "selected_company" not in st.session_state:
    st.session_state["selected_company"] = None
if "clear_screen" not in st.session_state:
    st.session_state["clear_screen"] = False

# --- Sidebar: Radio Button for History — only if not in clear mode ---
selected_index = (
    st.session_state["search_history"].index(st.session_state["selected_company"])
    if st.session_state["selected_company"] in st.session_state["search_history"]
    else None
)
selected_company = st.sidebar.radio(
    "Click a company to reload report:",
    st.session_state["search_history"],
    index=selected_index if selected_index is not None and not st.session_state["clear_screen"] else 0,
    key="company_radio"
)

# --- If user selects a company from the sidebar, override clear_screen ---
if selected_company and selected_company != st.session_state.get("selected_company"):
    st.session_state["selected_company"] = selected_company
    st.session_state["clear_screen"] = False

# --- New Research Button Logic ---
if st.sidebar.button("New Research"):
    st.session_state["clear_screen"] = False
    st.session_state["selected_company"] = None
    st.rerun()

# --- Instruction Note for New Research ---
st.sidebar.markdown(
    """
    <div style='font-size: 11px; color: white; margin-top: 10px; margin-bottom: -5px;'>
        <i>Note: Clicking <b>New Research</b> will refresh the chat <br>history & open a new chat.<br></i>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Page Header ---
st.title("AI Sales Research")
st.write("ℹ️ Enter a company name to fetch insights and generate a structured summary.")
st.markdown("<hr style='border: 1px solid #ffffff55;'>", unsafe_allow_html=True)

# --- Note Box ---
st.markdown(
    """
    <div style='color: white; font-size: 13px;'>
        <strong>Note:</strong>
        <ul style="padding-left: 18px; line-height: 1.6; margin-top: 10px;">
            <li>This app uses a dark theme. If your system uses a light/default theme, go to the top-right settings ( : ) and switch to <strong>Dark</strong> mode for optimal experience.</li>
            <li>Search history will reset on page reload. Each report is navigable, so ensure you <strong>download</strong> your reports before refreshing.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Report Viewer (Only if screen is not cleared and a company is selected)
if not st.session_state["clear_screen"] and selected_company:
    st.write(f"### Report for {selected_company}")
    if selected_company in st.session_state:
        report_text = st.session_state[selected_company]
        st.markdown(report_text, unsafe_allow_html=True)

        template_path = "ModelTemplate.docx"
        doc_file = fill_word_template(template_path, report_text)

        st.download_button(
            label="📄 Download",
            data=doc_file,
            file_name=f"{selected_company}_Report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    else:
        st.warning("⚠️ No previous report found for this company.")

# --- Info block if cleared
if st.session_state["clear_screen"]:
    st.info("Ready for new research. Please enter a new company name below.")
    # Do not render previous report here

# --- New Company Chat Input
user_input = st.chat_input("Enter a company name (Ex. Apple)...")

if user_input:
    # Reset state and store new entry
    st.session_state["clear_screen"] = False
    st.session_state["selected_company"] = user_input

    if user_input not in st.session_state["search_history"]:
        st.session_state["search_history"].append(user_input)

    with st.spinner(f"Searching for **{user_input}**..."):
        company_info = scrape_company_website(user_input)

    with st.spinner("Generating report..."):
        report = generate_summary(user_input, company_info)

    st.session_state[user_input] = report

    st.write(f"### Report for {user_input}")
    st.markdown(report, unsafe_allow_html=True)

    template_path = "ModelTemplate.docx"
    doc_file = fill_word_template(template_path, report)

    st.download_button(
        label="📄 Download Report",
        data=doc_file,
        file_name=f"{user_input}_Report.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )