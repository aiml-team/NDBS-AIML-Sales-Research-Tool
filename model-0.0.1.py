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
    /* New style for the container */
    .st-emotion-cache-1pxazr4 { /* This is a common class name for st.container, might change slightly in future versions */
        border: 1px solid #133168;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
        background-color: #133168; /* Slightly different background for the container */
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
    selected_company = st.radio(
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
    if st.button("New Research"):
        st.session_state["clear_screen"] = False
        st.session_state["selected_company"] = None
        st.rerun()
 
    # --- Instruction Note for New Research ---
    st.markdown(
        """
        <div style='font-size: 11px; color: white; margin-top: 10px; margin-bottom: -5px;'>
            <i>Note: Clicking <b>New Research</b> will refresh the chat <br>history & open a new chat.<br></i>
            <br></br>
        </div>
        """,
        unsafe_allow_html=True
    )
 
    # --- Container for Project Version to Feedback Form ---
    with st.container(border=True): # Using border=True to add a default border for visual separation
        st.markdown("""
        **Project Version: v0.0.1**
        - Version Document: [Sales Research V0.0.1](https://example.com/version-doc)
    
        **Current Enhancements**
        - UI/UX Improvements  
        - AI Integration  
        - User Authentication  
        - Performance Optimization  
        - API Enhancements
    
        **Upcoming Enhancements**
        - UI Development  
        - Search Capabilities  
        - Detailed Company Insights  
        - Report Enhancements
    
        **User Manual**
        - [User Manual Document](https://itellicloud.sharepoint.com/:p:/r/sites/US-saleshub/Shared%20Documents/06-%20Digital%20Innovations%20Internal%20Documents/Artificial%20Intelligence%20and%20Machine%20Learning%20Projects/2025%20-%20AI%20Sales%20Research/AI%20Sales%20Research%201.0%20-%20User%20Manual_V1.pptx?d=w979cf32d28f043c78f6fe37fe961494b&csf=1&web=1&e=3tUgDq)
    
        **Feedback Form**
        - [Sales Research Feedback](https://forms.office.com/Pages/DesignPageV2.aspx?subpage=design&FormId=Gd7ERvwPVEyqhu5ydK-p1ypLotnXSFBLsOBkdbM0WSpUMFc3WlVGSUQ4UkJSNE1EWENGRjRJUEVXSC4u&Token=1a4f070507194248af35fd0f193cd3d1)
        """)
 
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
 
# --- Hamburger Menu HTML ---
import streamlit.components.v1 as components
 
components.html(
    """
    <div class="hamburger-menu" onclick="toggleMenu()">
        <div class="hamburger-icon">
            <div class="hamburger-line"></div>
            <div class="hamburger-line"></div>
            <div class="hamburger-line"></div>
        </div>
    </div>
   
    <div class="menu-dropdown" id="menuDropdown">
        <div class="menu-item" onclick="showHelp()">
            <span>🔍 Search Tips</span>
        </div>
        <div class="menu-item" onclick="showAbout()">
            <span>ℹ️ About</span>
        </div>
        <div class="menu-divider"></div>
        <div class="menu-item" onclick="showSettings()">
            <span>⚙️ Settings</span>
        </div>
        <div class="menu-item" onclick="showContact()">
            <span>📞 Contact</span>
        </div>
        <div class="menu-divider"></div>
        <div class="menu-item" onclick="refreshApp()">
            <span>🔄 Refresh</span>
        </div>
    </div>
   
    <style>
    /* Hamburger Menu Styles */
    .hamburger-menu {
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 1000;
        background-color: #1e458e;
        border-radius: 50px;
        padding: 12px 16px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
   
    .hamburger-menu:hover {
        background-color: #2a5ba0;
        transform: scale(1.05);
    }
   
    .hamburger-icon {
        width: 24px;
        height: 18px;
        position: relative;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
   
    .hamburger-line {
        width: 100%;
        height: 2px;
        background-color: white;
        border-radius: 1px;
        transition: all 0.3s ease;
    }
   
    .menu-dropdown {
        position: fixed;
        bottom: 70px;
        left: 20px;
        background-color: #1e458e;
        border-radius: 12px;
        padding: 16px 0;
        min-width: 200px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        z-index: 999;
        opacity: 0;
        transform: translateY(20px);
        transition: all 0.3s ease;
        pointer-events: none;
    }
   
    .menu-dropdown.show {
        opacity: 1;
        transform: translateY(0);
        pointer-events: auto;
    }
   
    .menu-item {
        padding: 12px 20px;
        color: white;
        text-decoration: none;
        display: block;
        transition: background-color 0.2s ease;
        border: none;
        background: none;
        width: 100%;
        text-align: left;
        cursor: pointer;
        font-size: 14px;
    }
   
    .menu-item:hover {
        background-color: #133168;
        color: white;
    }
   
    .menu-divider {
        height: 1px;
        background-color: #133168;
        margin: 8px 0;
    }
    </style>
   
    <script>
    function toggleMenu() {
        const dropdown = document.getElementById('menuDropdown');
        dropdown.classList.toggle('show');
    }
   
    function showHelp() {
        alert('Search Tips:\\n\\n• Enter company names clearly (e.g., "Apple Inc." or "Microsoft")\\n• Use official company names for best results\\n• Reports include financial data, leadership info, and SWOT analysis\\n• Download reports before refreshing the page');
        document.getElementById('menuDropdown').classList.remove('show');
    }
   
    function showAbout() {
        alert('AI Sales Research v1.0\\n\\nThis application helps you generate comprehensive company research reports using AI-powered web scraping and analysis.\\n\\nFeatures:\\n• Company data extraction\\n• Financial analysis\\n• SWOT analysis\\n• Report generation\\n• Word document export');
        document.getElementById('menuDropdown').classList.remove('show');
    }
   
    function showSettings() {
        alert('Settings\\n\\nFor optimal experience:\\n• Use Dark mode (top-right settings)\\n• Ensure stable internet connection\\n• Download reports before page refresh\\n\\nNote: Settings are currently view-only.');
        document.getElementById('menuDropdown').classList.remove('show');
    }
   
    function showContact() {
        alert('Contact Information\\n\\nFor support or feedback:\\n• Email: support@example.com\\n• Phone: +1-XXX-XXX-XXXX\\n• Documentation: Available in sidebar\\n\\nFeedback form link available in sidebar.');
        document.getElementById('menuDropdown').classList.remove('show');
    }
   
    function refreshApp() {
        if (confirm('Are you sure you want to refresh the application? This will clear all current data.')) {
            window.location.reload();
        }
        document.getElementById('menuDropdown').classList.remove('show');
    }
   
    // Close menu when clicking outside
    document.addEventListener('click', function(event) {
        const menu = document.querySelector('.hamburger-menu');
        const dropdown = document.getElementById('menuDropdown');
       
        if (!menu.contains(event.target) && !dropdown.contains(event.target)) {
            dropdown.classList.remove('show');
        }
    });
    </script>
    """,
    height=0,
    width=0,
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