import os
import json
import time
from dotenv import load_dotenv
import streamlit as st
import altair as alt
import yfinance as yf
from ddgs import DDGS
from google import genai
from google.genai import types


# ==========================================
# 1. PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(
    page_title="AI Hedge Fund | Multi-Agent Stock Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom styling for clean portfolio look
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #888888;
        margin-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(28, 131, 225, 0.05);
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 15px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini Client
@st.cache_resource
def get_gemini_client():
    if not GEMINI_API_KEY:
        st.error("🚨 GEMINI_API_KEY not found in environment variables. Please check with owner.")
        st.stop()
    return genai.Client()

client = get_gemini_client()

# ==========================================
# 2. HELPER & BACKOFF FUNCTIONS
# ==========================================
def safe_generate_content(prompt: str, primary_model: str = "gemini-3.5-flash", fallback_model: str = "gemini-3.1-flash-lite", max_retries: int = 3) -> str:
    """Calls the Gemini API with automatic exponential backoff and model fallback."""
    base_delay = 2
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=primary_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2)
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            if any(code in error_str for code in ["503", "UNAVAILABLE", "429"]):
                time.sleep(base_delay * (2 ** attempt))
            else:
                break
                
    try:
        response = client.models.generate_content(
            model=fallback_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2)
        )
        return response.text + "\n\n*(Note: Generated using backup model due to high server demand)*"
    except Exception as fallback_error:
        return f"CRITICAL ERROR: Both primary and fallback models failed. Details: {str(fallback_error)}"

# ==========================================
# 3. CACHED DATA & AGENT FUNCTIONS
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_recent_news(ticker: str, max_results: int = 5) -> str:
    """Fetches news via DuckDuckGo with a schema-robust Yahoo Finance fallback."""
    results = []
    
    # 1. Try DuckDuckGo
    try:
        query = f"{ticker} stock finance market news"
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=max_results):
                title = r.get('title', 'No Title')
                summary = r.get('body', '')
                source = r.get('source', r.get('url', 'Unknown'))
                results.append(f"Title: {title}\nSummary: {summary}\nSource: {source}\n---")
    except Exception:
        pass  # Silently drops to Yahoo Finance if DDGS rate-limits

    # 2. Fallback to Yahoo Finance (Updated for new nested schema)
    if not results:
        try:
            stock = yf.Ticker(ticker)
            for r in stock.news[:max_results]:
                content = r.get('content', {})
                
                # Check top-level first, fall back to nested 'content' dict
                title = r.get('title') or content.get('title', 'No Title')
                summary = r.get('summary') or content.get('summary', 'No summary available.')
                
                # Dig safely for publisher name
                publisher = r.get('publisher') or content.get('provider', {}).get('displayName', 'Yahoo Finance')
                
                results.append(f"Title: {title}\nSummary: {summary}\nSource: {publisher}\n---")
        except Exception as e:
            return f"Error: All news tools failed to retrieve data. Reason: {str(e)}"

    return "\n".join(results) if results else "No recent news articles could be found for this ticker."

@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamental_data(ticker: str) -> dict:
    """Pulls quantitative financial metrics using yfinance. Cached for 1 hour."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "Company Name": info.get("longName", ticker),
            "Current Price": info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
            "Market Cap": info.get("marketCap", "N/A"),
            "Trailing P/E": info.get("trailingPE", "N/A"),
            "Forward P/E": info.get("forwardPE", "N/A"),
            "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52 Week Low": info.get("fiftyTwoWeekLow", "N/A"),
            "50 Day MA": info.get("fiftyDayAverage", "N/A"),
            "200 Day MA": info.get("twoHundredDayAverage", "N/A"),
            "Revenue Growth (YoY)": info.get("revenueGrowth", "N/A"),
            "Analyst Rec": info.get("recommendationKey", "N/A").upper(),
            "Target Price": info.get("targetMeanPrice", "N/A")
        }
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def get_price_history(ticker: str):
    """Fetches 1-year price history for charts."""
    try:
        return yf.Ticker(ticker).history(period="1y")['Close']
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def run_news_agent(ticker: str, raw_news: str) -> str:
    """Agent 1: Analyzes market sentiment and recent news."""
    if "No recent news articles" in raw_news or "Error:" in raw_news:
        return f"Agent 1 Analysis Aborted: {raw_news}"
    
    prompt = f"""
    You are an expert Financial News & Sentiment Analyst.
    Analyze the following recent news headlines for ticker symbol: {ticker}.
    
    Raw News Data:
    {raw_news}
    
    Task:
    1. Summarize the overarching market sentiment (Bullish, Bearish, or Neutral).
    2. Identify the top 2-3 key catalysts or risks mentioned in the news.
    3. Keep your analysis concise, objective, and factual.

    Constraint: Maintain a strictly objective, scientific tone in the third person or passive voice. Do not use personal pronouns (I, we, you).
    """
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2)
    )
    return response.text

@st.cache_data(ttl=3600, show_spinner=False)
def run_quant_agent(ticker: str, metrics_json: str) -> str:
    """Agent 2: Evaluates quantitative metrics and valuation."""
    prompt = f"""
    You are an expert Quantitative Financial Analyst.
    Evaluate the following valuation metrics and technical indicators for: {ticker}.
    
    Raw Financial Data:
    {metrics_json}
    
    Task:
    1. Evaluate the valuation (e.g., is the P/E ratio overvalued or undervalued relative to historical norms?).
    2. Assess the price trend based on the 50-day and 200-day moving averages.
    3. Provide a clear assessment of the company's fundamental strength.

    Constraint: Maintain a strictly objective, scientific tone in the third person or passive voice. Do not use personal pronouns (I, we, you).
    """
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1)
    )
    return response.text

@st.cache_data(ttl=3600, show_spinner=False)
def run_portfolio_manager_agent(ticker: str, news_analysis: str, quant_analysis: str) -> str:
    """Agent 3: Synthesizes research and issues an investment recommendation."""
    prompt = f"""
        You are a Quantitative Research Director publishing an institutional investment thesis for ticker symbol: {ticker}.
        You have been provided with two independent empirical data reports:
        
        === REPORT 1: MACROECONOMIC & SENTIMENT DATA ===
        {news_analysis}
        
        === REPORT 2: QUANTITATIVE & VALUATION METRICS ===
        {quant_analysis}
        
        [STYLE & TONE CONSTRAINTS - STRICT ENFORCEMENT REQUIRED]
        1. Academic & Scientific Tone: The analysis must be written with the academic rigor, objectivity, and precision of a peer-reviewed financial economics paper.
        2. Third-Person & Passive Voice Only: All assertions, evaluations, and conclusions must be framed in the third person or passive voice (e.g., "It is observed that...", "The data indicates...", "A valuation anomaly is present...").
        3. Absolute Pronoun Ban: Do NOT use first- or second-person pronouns under any circumstances. Forbidden words include: I, me, my, we, our, us, you, your.
        4. Evidential Framing: Base all conclusions strictly on the provided reports. Avoid speculative language; use probabilistic and empirical framing (e.g., "Historical momentum suggests..." rather than "The stock will probably go up...").
        
        [TASK & OUTPUT STRUCTURE]
        Synthesize the provided reports into a formal research memorandum. Format the output EXACTLY using the following Markdown structure:
        
        # Empirical Investment Thesis: {ticker}
        ## Abstract & Executive Summary
        ## Key Bullish Catalysts & Positive Expectancies
        ## Downside Risks & Macroeconomic Headwinds
        ## Final Rating & Strategic Allocation: [BUY / HOLD / SELL]
        *(Provide a definitive BUY, HOLD, or SELL rating, followed by an objective, third-person justification of the time horizon and risk-adjusted return expectations).*
        """
    return safe_generate_content(prompt=prompt, primary_model="gemini-3.5-flash", fallback_model="gemini-3.1-flash-lite")

# ==========================================
# 4. USER INTERFACE
# ==========================================
st.markdown('<div class="main-title">🤖 AI Hedge Fund Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A Multi-Agent AI system synthesizing live news sentiment and quantitative valuation into institutional-grade investment memos.</div>', unsafe_allow_html=True)

# Expander Configuration at the Top
with st.expander("⚙️ Configuration & Input", expanded=True):
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker_input = st.text_input("Enter Stock Ticker Symbol:", value="NVDA", max_chars=10, help="e.g., AAPL, NVDA, TSLA, MSFT").upper().strip()
    with col2:
        st.write("") # Spacing
        st.write("") # Spacing
        execute_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Ensure results ONLY appear after execution button is clicked
if execute_btn:
    if not ticker_input:
        st.warning("⚠️ Please enter a valid stock ticker symbol.")
    else:
        st.session_state["active_ticker"] = ticker_input
        
        # Real-time agent status tracker
        with st.status(f"🕵️‍♂️ Deploying AI Agents for **{ticker_input}**...", expanded=True) as status:
            st.write("📊 **Agent 1:** Scraping real-time market news & sentiment...")
            raw_news = get_recent_news(ticker_input)
            news_report = run_news_agent(ticker_input, raw_news)
            
            st.write("📈 **Agent 2:** Pulling financial balance sheets & technical moving averages...")
            metrics_dict = get_fundamental_data(ticker_input)
            quant_report = run_quant_agent(ticker_input, json.dumps(metrics_dict, indent=2))
            
            st.write("👔 **Agent 3:** Chief Investment Officer synthesizing research into final thesis...")
            final_memo = run_portfolio_manager_agent(ticker_input, news_report, quant_report)
            
            # Save all outputs to Session State so switching tabs won't clear results
            st.session_state["metrics_dict"] = metrics_dict
            st.session_state["news_report"] = news_report
            st.session_state["quant_report"] = quant_report
            st.session_state["final_memo"] = final_memo
            st.session_state["raw_news"] = raw_news
            st.session_state["price_history"] = get_price_history(ticker_input)
            
            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

# ==========================================
# 5. RESULTS DISPLAY (Post-Execution)
# ==========================================
if "active_ticker" in st.session_state:
    ticker = st.session_state["active_ticker"]
    metrics = st.session_state.get("metrics_dict", {})
    
    st.divider()
    
    # Top KPI Metrics Bar
    if "error" not in metrics:
        st.subheader(f"📊 {metrics.get('Company Name', ticker)} — Snapshot")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        with m_col1:
            price = metrics.get('Current Price', 'N/A')
            st.metric(label="Current Price", value=f"${price:,.2f}" if isinstance(price, (int, float)) else price)
        with m_col2:
            pe = metrics.get('Trailing P/E', 'N/A')
            st.metric(label="Trailing P/E", value=f"{pe:.1f}x" if isinstance(pe, (int, float)) else pe)
        with m_col3:
            mcap = metrics.get('Market Cap', 'N/A')
            if isinstance(mcap, (int, float)):
                mcap_str = f"${mcap/1e12:.2f}T" if mcap >= 1e12 else f"${mcap/1e9:.2f}B"
            else:
                mcap_str = mcap
            st.metric(label="Market Cap", value=mcap_str)
        with m_col4:
            st.metric(label="Analyst Consensus", value=metrics.get('Analyst Rec', 'N/A'))
        st.write("")
        
    # Tabbed Content Sections
# Tabbed Content Sections
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Investment Memo", "📰 Sentiment Agent", "📈 Valuation Agent", "🔍 Raw Data Feed"])
    
    with tab1:
        # Escape dollar signs so Streamlit doesn't turn them into math equations
        memo_text = st.session_state.get("final_memo", "").replace("$", "\\$")
        st.markdown(memo_text)
        
    with tab2:
        st.subheader("📰 Market Sentiment & Catalyst Analysis")
        news_text = st.session_state.get("news_report", "").replace("$", "\\$")
        st.markdown(news_text)
        
    with tab3:
        st.subheader("📈 Quantitative & Valuation Assessment")
        col_left, col_right = st.columns([1, 1])
        with col_left:
            quant_text = st.session_state.get("quant_report", "").replace("$", "\\$")
            st.markdown(quant_text)
        with col_right:
            st.write("**1-Year Price History ($)**")
            history = st.session_state.get("price_history")
            if history is not None and not history.empty:
                # Reset index so 'Date' is a standard column for Altair
                chart_data = history.reset_index()
        
                # Build chart with zero=False to dynamically scale the y-axis
                chart = alt.Chart(chart_data).mark_line(color='#1c83e1', strokeWidth=2).encode(
                    x=alt.X('Date:T', title=None, axis=alt.Axis(format='%b %Y')),
                    y=alt.Y('Close:Q', title='Price ($)', scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), alt.Tooltip('Close:Q', format='$.2f')]
                ).interactive()
                
                st.altair_chart(chart, use_container_width=True)
                # st.line_chart(history, use_container_width=True)
            else:
                st.info("Chart data unavailable.")
                
    with tab4:
        st.subheader("🔍 Raw Data Feeds Pulled by Agents")
        col_raw1, col_raw2 = st.columns(2)
        with col_raw1:
            st.markdown("**Fundamental Metrics (JSON)**")
            st.json(metrics)
        with col_raw2:
            st.markdown("**Scraped News Headlines**")
            st.text_area("DuckDuckGo / Yahoo Feeds", value=st.session_state.get("raw_news", ""), height=350)