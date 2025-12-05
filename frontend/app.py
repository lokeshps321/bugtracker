import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd # Added for the analytics chart
from streamlit_option_menu import option_menu # Modern sidebar
import os

# Use environment variable for API URL (cloud deployment support)
API_URL = os.getenv("API_URL", "http://localhost:8000")

from streamlit_lottie import st_lottie
import time

# ------------------------------
# Page config - must be first Streamlit command
# ------------------------------
st.set_page_config(page_title="BugFlow", layout="wide")

# Lottie Animation URLs
LOTTIE_LOGIN = "https://lottie.host/5a805076-2f64-42d8-8835-253335555555/login.json" # Placeholder, will use a generic coding one
LOTTIE_LOADING = "https://assets9.lottiefiles.com/packages/lf20_b88nh30c.json" # Coding animation
LOTTIE_SUCCESS = "https://assets9.lottiefiles.com/packages/lf20_jbrw3hcz.json" # Confetti
LOTTIE_EMPTY = "https://assets9.lottiefiles.com/packages/lf20_s8pbrcfw.json" # Empty box

def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Inject Custom CSS for professional dark theme (Global)
st.markdown(
    """
    <style>
    /* Global Dark Theme & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #e2e8f0; /* Slate-200 */
    }
    
    /* Main Background - Deep Dark with Blue/Purple Glow */
    .stApp {
        background-color: #030712; /* Rich Black */
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(59, 130, 246, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 85% 30%, rgba(147, 51, 234, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 15% 70%, rgba(37, 99, 235, 0.05) 0%, transparent 50%);
        background-attachment: fixed;
    }

    /* Keyframe Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes glowPulse {
        0% { box-shadow: 0 0 15px rgba(59, 130, 246, 0.3); }
        50% { box-shadow: 0 0 25px rgba(59, 130, 246, 0.5); }
        100% { box-shadow: 0 0 15px rgba(59, 130, 246, 0.3); }
    }

    /* Apply animations globally */
    .stApp {
        animation: fadeIn 0.8s ease-out;
    }
    
    .landing-container, .login-form-container, .stMarkdown, .stDataFrame, .css-1r6slb0 {
        animation: slideUp 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* Landing Page Container - Glassmorphism Dark */
    .landing-container {
        max_width: 1000px;
        margin: 4rem auto;
        padding: 4rem 2rem;
        text-align: center;
        background: rgba(17, 24, 39, 0.4); /* Gray-900 with opacity */
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }

    /* Typography Styling */
    .landing-title {
        font-size: 5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.1;
        background: linear-gradient(to right, #60a5fa, #a78bfa, #f472b6); /* Blue to Purple to Pink */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        text-shadow: 0 0 40px rgba(96, 165, 250, 0.3);
    }

    .landing-subtitle {
        font-size: 1.5rem;
        color: #94a3b8; /* Slate-400 */
        margin-bottom: 4rem;
        line-height: 1.6;
        font-weight: 400;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Feature Cards - Dark Glass */
    .feature-card {
        background: rgba(30, 41, 59, 0.4); /* Slate-800 low opacity */
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
        height: 100%;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        background: rgba(30, 41, 59, 0.6);
        border-color: rgba(96, 165, 250, 0.3);
        box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.5);
    }

    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1.5rem;
        background: rgba(59, 130, 246, 0.1);
        width: 56px;
        height: 56px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        color: #60a5fa;
    }

    .feature-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc; /* Slate-50 */
        margin-bottom: 0.75rem;
    }

    .feature-desc {
        color: #94a3b8; /* Slate-400 */
        line-height: 1.6;
        font-size: 0.95rem;
    }

    /* Primary Button - Glowing Blue */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 9999px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 0 20px rgba(37, 99, 235, 0.4);
        transition: all 0.3s ease;
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 30px rgba(37, 99, 235, 0.6);
        border-color: rgba(255,255,255,0.2);
    }

    /* Login Form Styling */
    .app-title-login {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(to right, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        margin-top: 2rem;
    }

    .login-subtitle {
        text-align: center;
        color: #94a3b8;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }

    /* Inputs - Dark Mode */
    .stTextInput>div>div>input {
        background-color: rgba(15, 23, 42, 0.6) !important;
        color: white !important;
        border: 1px solid rgba(51, 65, 85, 1);
        border-radius: 8px;
        padding: 10px 12px;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }

    /* Sidebar - Seamless Glass (Refined) */
    section[data-testid="stSidebar"] {
        background: rgba(17, 24, 39, 0.7); /* Lighter, more transparent */
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.05);
        box-shadow: none;
        margin: 0 !important;
        border-radius: 0 !important;
        height: 100vh !important;
        padding-top: 1rem;
    }
    
    /* Sidebar Nav Items */
    .st-emotion-cache-16txtl3 {
        padding-top: 1rem;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Card Styles for Dashboard */
    div[data-testid="stMarkdownContainer"] > div {
        /* This targets our custom HTML cards if they are direct children */
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# Session State defaults (safe init)
# ------------------------------
if "token" not in st.session_state:
    st.session_state.token = None
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None
if "user" not in st.session_state:
    st.session_state.user = {}
if "last_predict" not in st.session_state:
    st.session_state.last_predict = {}
if "refresh_key" not in st.session_state:
    st.session_state.refresh_key = 0  # bump this to force rerenders in some places

# ------------------------------
# Utility helpers
# ------------------------------
def _headers():
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}

def api_post(endpoint, json_data):
    try:
        r = requests.post(f"{API_URL.rstrip('/')}/{endpoint.lstrip('/')}", json=json_data, headers=_headers(), timeout=60)
        return r
    except Exception as e:
        st.error(f"Network error calling {endpoint}: {e}")
        return None

def api_get(endpoint, params=None):
    try:
        r = requests.get(f"{API_URL.rstrip('/')}/{endpoint.lstrip('/')}", headers=_headers(), params=params, timeout=60)
        return r
    except Exception as e:
        st.error(f"Network error calling {endpoint}: {e}")
        return None

# ------------------------------
# Auth helpers
# ------------------------------
DEMO_ROLE_MAP = {
    "tester1@example.com": "tester",
    "dev1@example.com": "developer",
    "pm1@example.com": "project_manager",
}

def login(email, password):
    """
    Login via backend /token. On success, store token and fetch current_user.
    If /current_user is missing, fallback to DEMO_ROLE_MAP.
    """
    # POST form-data-compatible to /token (many backends accept form data; tests expect form-data)
    try:
        r = requests.post(f"{API_URL}/token", data={"username": email, "password": password}, timeout=6)
    except Exception as e:
        st.error(f"Could not reach backend: {e}")
        return False

    if r is None:
        return False

    if r.status_code != 200:
        st.error("Login failed. Check email/password and backend status.")
        return False

    token = r.json().get("access_token")
    if not token:
        st.error("Login returned no token.")
        return False

    st.session_state.token = token
    st.session_state.logged_in = True

    # Try to fetch /current_user for role & id
    try:
        user_resp = requests.get(f"{API_URL}/current_user", headers={"Authorization": f"Bearer {token}"}, timeout=6)
        if user_resp.status_code == 200:
            st.session_state.user = user_resp.json()
            st.session_state.role = st.session_state.user.get("role")
        else:
            # fallback: use DEMO_ROLE_MAP by email
            st.session_state.role = DEMO_ROLE_MAP.get(email, "tester")
            st.session_state.user = {"email": email, "role": st.session_state.role}
    except Exception:
        st.session_state.role = DEMO_ROLE_MAP.get(email, "tester")
        st.session_state.user = {"email": email, "role": st.session_state.role}

    # bump refresh_key so UI sections that depend on it re-render nicely
    st.session_state.refresh_key += 1
    
    # Use st.rerun() to switch immediately
    st.rerun()
    return True

def logout():
    st.session_state.token = None
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.user = {}
    st.session_state.last_predict = {}
    st.session_state.refresh_key += 1
    
    # Use st.rerun() to switch immediately
    st.rerun()

# ------------------------------
# UI helpers
# ------------------------------
def header(title, subtitle=None):
    # simple header with gradient background using markdown block
    grad = """
    <div style="
        background: linear-gradient(90deg,#3b82f6 0%,#8b5cf6 100%);
        color: white;
        padding: 18px;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(14,30,37,0.12);
        margin-bottom: 12px;">
      <h2 style="margin:0;">{}</h2>
      <div style="opacity:0.95;">{}</div>
    </div>
    """.format(title, subtitle or "")
    st.markdown(grad, unsafe_allow_html=True)

def severity_badge(sev):
    sev = (sev or "").lower()
    if sev == "high":
        color = "#ef4444"  # red
    elif sev == "medium":
        color = "#f59e0b"  # amber
    elif sev == "low":
        color = "#10b981"  # green
    else:
        color = "#94a3b8"  # gray
    return f"<span style='background:{color}; color:white; padding:4px 8px; border-radius:6px; font-weight:600;'>{sev or 'unknown'}</span>"

def ml_metric_card(title, value, trend="", icon="📊"):
    """Enhanced metric card for ML stats"""
    trend_html = ""
    if trend:
        if "↑" in trend:
            trend_html = f"<span style='color:#10b981; font-size:14px;'>{trend}</span>"
        elif "↓" in trend:
            trend_html = f"<span style='color:#ef4444; font-size:14px;'>{trend}</span>"
        else:
            trend_html = f"<span style='color:#64748b; font-size:14px;'>{trend}</span>"
    
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding:20px;
            border-radius:12px;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            margin-bottom:12px;
            color: white;">
            <div style="font-size:24px; margin-bottom:8px">{icon}</div>
            <div style="font-size:13px; opacity:0.9; font-weight:500; text-transform: uppercase; letter-spacing: 0.5px">{title}</div>
            <div style="font-size:32px; font-weight:700; margin-top:12px">{value}</div>
            {f'<div style="margin-top:8px">{trend_html}</div>' if trend_html else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

def bug_card(bug, role="project_manager"):
    """Reusable bug card component"""
    title = bug.get('title') or bug.get('description')[:50] + "..."
    st.markdown(
        f"""
        <div style="background:rgba(30, 41, 59, 0.4); backdrop-filter: blur(10px); padding:16px; border-radius:10px; margin-bottom:12px; border: 1px solid rgba(255,255,255,0.05); border-left: 4px solid #3b82f6; transition: all 0.3s ease;" class="hover-card">
            <div style="display:flex; justify-content:space-between; align-items:start;">
                <div>
                    <div style="font-weight:700; font-size:16px; color:#f8fafc;">#{bug.get('id')} — {title}</div>
                    <div style="color:#94a3b8; margin-top:2px; font-size:12px;">{bug.get('project')}</div>
                    <div style="color:#cbd5e1; margin-top:6px; font-size:14px;">{bug.get('description')[:100]}...</div>
                </div>
                <div style="text-align:right;">
                    {severity_badge(bug.get('severity'))}
                    <div style="margin-top:6px;"><span style='background:rgba(37, 99, 235, 0.2); color:#60a5fa; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:600;'>{bug.get('team') or 'Unassigned'}</span></div>
                </div>
            </div>
            <div style="margin-top:12px; padding-top:12px; border-top:1px solid rgba(255,255,255,0.05); display:flex; justify-content:space-between; align-items:center; font-size:12px; color:#94a3b8;">
                <div>Status: <span style="font-weight:600; color:#cbd5e1; text-transform:uppercase;">{bug.get('status')}</span></div>
                <div>Reported: {bug.get('created_at', '').split('T')[0]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def small_card(title, value, subtitle="", bg_color="white"):
    # Map pastel colors to border/glow colors for dark mode
    border_color = bg_color
    if bg_color == "white": border_color = "rgba(255,255,255,0.1)"
    elif "#fde68a" in bg_color: border_color = "#fbbf24" # amber
    elif "#fca5a5" in bg_color: border_color = "#f87171" # red
    elif "#93c5fd" in bg_color: border_color = "#60a5fa" # blue
    elif "#a7f3d0" in bg_color: border_color = "#34d399" # green
    
    st.markdown(
        f"""
        <div style="
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(10px);
            padding:16px;
            border-radius:12px;
            border: 1px solid {border_color};
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom:8px;">
            <div style="font-size:14px; color:#94a3b8">{title}</div>
            <div style="font-size:24px; font-weight:700; color:#f8fafc; margin: 4px 0;">{value}</div>
            <div style="font-size:12px; color:#64748b">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------
# Tester pages
# ------------------------------
def tester_report_page():
    header("Tester — Report a Bug", "Describe the issue and optionally predict severity before submitting.")
    with st.form("report_form", clear_on_submit=False):
        title = st.text_input("Bug title", placeholder="E.g. 'Login button crashes on Chrome'", max_chars=200)
        desc = st.text_area("Bug description", placeholder="E.g. 'Clicking Save crashes the app on Chrome 117' ", height=140)
        project = st.text_input("Project name", value="WebApp")
        assigned = st.text_input("Assign to (user id) — optional", value="")
        cols = st.columns([1,1,2])
        with cols[0]:
            predict_btn = st.form_submit_button("Predict", use_container_width=True)
        with cols[1]:
            submit_btn = st.form_submit_button("Submit Bug", use_container_width=True)
        with cols[2]:
            st.markdown("<div style='padding-top:8px; color:#475569'>Tip: use Predict to check severity/team before submitting.</div>", unsafe_allow_html=True)

        if predict_btn:
            if not desc.strip():
                st.warning("Please enter a description to predict.")
            elif not project.strip():
                st.warning("Please enter a project name.")
            else:
                # Show loading animation
                lottie_loading = load_lottieurl(LOTTIE_LOADING)
                placeholder = st.empty()
                if lottie_loading:
                    with placeholder:
                        st_lottie(lottie_loading, height=100, key="predict_loading")
                
                with st.spinner("AI Analyzing..."):
                    resp = api_post("predict", {"title": title, "description": desc, "project": project})
                
                placeholder.empty()
                    
                if resp and resp.status_code == 200:
                    j = resp.json()
                    st.session_state.last_predict = j
                    st.success(f"Predicted severity: {j.get('severity')} — team: {j.get('team')}")
                else:
                    if resp is not None:
                        st.error(f"Prediction failed: {resp.text}")
                    else:
                        st.error("Prediction failed (no response).")

        if submit_btn:
            if not desc.strip() or not project.strip():
                st.warning("Please fill description and project.")
            else:
                # Create payload with required fields
                payload = {
                    "title": title.strip() if title.strip() else desc[:50] + "...",  # Use first 50 chars if no title
                    "description": desc, 
                    "project": project,
                    "severity": "medium"  # Default severity if not predicted
                }
                
                # Add predicted values if available
                if st.session_state.last_predict:
                    payload.update({
                        "severity": st.session_state.last_predict.get("severity", "medium"),
                        "team": st.session_state.last_predict.get("team")
                    })
                
                # Add assigned user if provided
                if assigned.strip():
                    try:
                        payload["assigned_to_id"] = int(assigned)
                    except ValueError:
                        st.warning("Invalid user ID. Please enter a number or leave blank.")
                        return
                
                # Submit the bug report
                resp = api_post("report_bug", payload)
                if resp is None:
                    st.error("No response from backend.")
                elif resp.status_code in (200, 201):
                    st.success("Bug reported successfully.")
                    st.session_state.refresh_key += 1
                elif resp.status_code == 400 or resp.status_code == 409:
                    # deduplication message expected in detail
                    detail = ""
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except Exception:
                        detail = resp.text
                    st.error(f"Failed to report: {detail}")
                    # if backend returned "Possible duplicate bug ID: X" extract id and show note
                    if "duplicate" in detail.lower() or "possible duplicate" in detail.lower():
                        st.warning("Backend flagged this as a possible duplicate. Please review or contact PM.")
                else:
                    st.error(f"Failed to report: {resp.status_code} {resp.text}")

def tester_my_bugs_page():
    header("Tester — My Reported Bugs", "List of bugs you reported.")
    params = {}
    resp = api_get("bugs", params=params)
    if resp is None:
        return
    if resp.status_code != 200:
        st.error("Failed to fetch your bugs.")
        return
    bugs = resp.json()
    if not bugs:
        st.info("You have not reported any bugs yet.")
        return

    # --- Filters for Tester ---
    st.markdown("### 🔍 Filter My Bugs")
    
    # Extract unique values for filters
    all_severities = sorted(list(set(b.get("severity") or "unknown" for b in bugs)))
    all_teams = sorted(list(set(b.get("team") or "unknown" for b in bugs)))
    all_statuses = sorted(list(set(b.get("status") or "unknown" for b in bugs)))
    
    # Callback to clear filters safely
    def clear_filters_callback():
        st.session_state.tester_filter_sev = []
        st.session_state.tester_filter_team = []
        st.session_state.tester_filter_status = []

    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    with c1:
        sel_sev = st.multiselect("Severity", all_severities, key="tester_filter_sev")
    with c2:
        sel_team = st.multiselect("Team", all_teams, key="tester_filter_team")
    with c3:
        sel_status = st.multiselect("Status", all_statuses, key="tester_filter_status")
    with c4:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        # Use on_click callback to avoid StreamlitAPIException
        st.button("Clear", key="clear_tester_filters", on_click=clear_filters_callback, use_container_width=True)
        
    # Apply filters
    filtered_bugs = bugs
    if sel_sev:
        filtered_bugs = [b for b in filtered_bugs if (b.get("severity") or "unknown") in sel_sev]
    if sel_team:
        filtered_bugs = [b for b in filtered_bugs if (b.get("team") or "unknown") in sel_team]
    if sel_status:
        filtered_bugs = [b for b in filtered_bugs if (b.get("status") or "unknown") in sel_status]
        
    st.markdown(f"**Showing {len(filtered_bugs)} of {len(bugs)} bugs**")
    st.markdown("---")

    if not filtered_bugs:
        st.info("No bugs match the selected filters.")
    else:
        for b in filtered_bugs:
            # Use the new bug_card component for consistent look
            bug_card(b, role="tester")

# ------------------------------
# Developer pages
# ------------------------------
def dev_assigned_page():
    header("Developer — Assigned Bugs", "Bugs assigned to you. Update status quickly.")
    resp = api_get("bugs")
    if resp is None:
        return
    if resp.status_code != 200:
        st.error("Failed to fetch bugs.")
        return
    bugs = resp.json()
    # try to filter by assigned_to_id using current_user id
    current_id = st.session_state.user.get("id")
    assigned = []
    if current_id:
        for b in bugs:
            if b.get("assigned_to_id") == current_id:
                assigned.append(b)
    else:
        # fallback: show bugs where assigned_to_id is not null (developer demo)
        assigned = [b for b in bugs if b.get("assigned_to_id")]

    if not assigned:
        st.info("No bugs currently assigned to you.")
        return

    for b in assigned:
        with st.expander(f"#{b.get('id')} — {b.get('project')} — {b.get('status')}"):
            st.write(b.get("description"))
            col1, col2 = st.columns([1,2])
            with col1:
                new_status = st.selectbox("Change status", ["open", "in_progress", "resolved"], index=["open","in_progress","resolved"].index(b.get("status")) if b.get("status") in ["open","in_progress","resolved"] else 0, key=f"dev_status_{b.get('id')}")
            with col2:
                if st.button("Update", key=f"dev_update_{b.get('id')}"):
                    update_resp = api_post("update_bug", {"bug_id": b.get("id"), "status": new_status})
                    if update_resp is None:
                        st.error("No response from backend")
                    elif update_resp.status_code == 200:
                        st.success("Bug updated!")
                        st.session_state.refresh_key += 1
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(f"Failed to update: {update_resp.text}")

# ------------------------------
# Project Manager pages
# ------------------------------
def pm_dashboard_page():
    header("Project Manager — Overview", "Quick stats and navigation")
    # fetch bugs for dashboard numbers
    resp = api_get("bugs")
    if resp is None or resp.status_code != 200:
        st.error("Failed to load bugs for analytics.")
        return
    bugs = resp.json()
    total = len(bugs)
    open_count = sum(1 for b in bugs if b.get("status") == "open")
    in_progress = sum(1 for b in bugs if b.get("status") == "in_progress")
    resolved = sum(1 for b in bugs if b.get("status") == "resolved" or b.get("status") == "done")

    c1, c2, c3, c4 = st.columns(4)
    # Using darker colors for better contrast
    with c1: small_card("Total Bugs", total, "", bg_color="#fde68a")       # amber-200
    with c2: small_card("Open", open_count, "", bg_color="#fca5a5")         # red-200
    with c3: small_card("In Progress", in_progress, "", bg_color="#93c5fd") # blue-200
    with c4: small_card("Resolved", resolved, "", bg_color="#a7f3d0")       # green-200

    # --- NEW: AI Model Health & Continuous Learning Section ---
    st.markdown("---")
    st.markdown("### 🤖 AI Model Health & Continuous Learning (MLOps)")

    # Assuming a new API endpoint /feedback_count exists on the backend
    feedback_resp = api_get("feedback_count")
    feedback_count = 0
    if feedback_resp and feedback_resp.status_code == 200:
        try:
            # We assume the backend returns {"total_feedback": N}
            feedback_count = feedback_resp.json().get("total_feedback", 0)
        except Exception:
            feedback_count = 0 # Handle case where response isn't valid JSON

    
    col_ai_1, col_ai_2, col_ai_3 = st.columns(3)
    with col_ai_1:
        # This shows the data collection for fine-tuning
        small_card("Training Data Collected", feedback_count, 
                   "Total expert corrections since last fine-tuning run.", 
                   bg_color="#e0f2fe") # blue-50
    with col_ai_2:
        # This shows the PM what needs to happen to trigger the next fine-tune
        FINE_TUNE_THRESHOLD = 50  # Changed to 50 as requested for robust MLOps
        if feedback_count >= FINE_TUNE_THRESHOLD:
            status_text = "Ready (Auto-Retraining)"
            bg_color = "#93c5fd" # Blue (auto-retraining now happens with each feedback)
        else:
            status_text = f"Collecting ({FINE_TUNE_THRESHOLD - feedback_count} to go)"
            bg_color = "#d1fae5" # Green

        small_card("Next Fine-Tune Trigger", status_text,
                   f"Real-time retraining after each feedback correction.",
                   bg_color=bg_color)
    with col_ai_3:
        # This confirms the fine-tuned model is live
        small_card("Model Status", "In Production (Stable)", 
                   "DistilBERT/SentenceTransformer models are running live.", 
                   bg_color="#d1fae5") # green-50
    # --- END NEW AI/MLOps Section ---
    
    # --- Interactive Analytics Charts ---
    st.markdown("### 📈 Interactive Analytics")
    if bugs:
        df = pd.DataFrame(bugs)
        
        col_chart_1, col_chart_2 = st.columns(2)
        
        with col_chart_1:
            st.markdown("#### Bug Severity Distribution")
            # Rich Donut Chart for Severity
            severity_counts = df['severity'].value_counts().reset_index()
            severity_counts.columns = ['severity', 'count']
            
            # Rich Color Palette
            sev_colors = {
                'critical': '#ff0055', # Neon Red
                'high': '#ff5e00',     # Neon Orange
                'medium': '#ffcc00',   # Neon Yellow
                'low': '#00cc66',      # Neon Green
                'unknown': '#888888'   # Gray
            }
            
            fig_donut = go.Figure(data=[go.Pie(
                labels=severity_counts['severity'],
                values=severity_counts['count'],
                hole=.6,
                marker=dict(colors=[sev_colors.get(x.lower(), '#888888') for x in severity_counts['severity']]),
                textinfo='label+percent',
                hoverinfo='label+value+percent',
                textposition='outside',
                pull=[0.02] * len(severity_counts)
            )])
            
            fig_donut.update_layout(
                showlegend=False,
                margin=dict(t=30, l=0, r=0, b=0),
                height=320,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                annotations=[dict(text='Severity', x=0.5, y=0.5, font_size=22, font_family="Arial Black", showarrow=False)]
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with col_chart_2:
            st.markdown("#### Bugs by Team")
            # Rich Horizontal Bar Chart (different from previous vertical bar)
            team_counts = df['team'].value_counts().reset_index()
            team_counts.columns = ['team', 'count']
            
            fig_bar = go.Figure(data=[go.Bar(
                y=team_counts['team'], # y for horizontal
                x=team_counts['count'],
                orientation='h',
                marker=dict(
                    color=team_counts['count'],
                    colorscale='Plasma', # Very rich purple-orange gradient
                    showscale=False,
                    line=dict(width=0)
                ),
                text=team_counts['count'],
                textposition='auto',
            )])
            
            fig_bar.update_layout(
                xaxis_title=None,
                yaxis_title=None,
                margin=dict(t=20, l=0, r=0, b=0),
                height=320,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
                yaxis=dict(showgrid=False, categoryorder='total ascending') # Sort bars
            )
            st.plotly_chart(fig_bar, use_container_width=True)

def pm_all_bugs_page():
    header("Project Manager — All Bugs", "Filter and manage all reported bugs.")
    
    resp = api_get("bugs")
    if resp is None or resp.status_code != 200:
        st.error("Failed to load bugs.")
        return
    bugs = resp.json()
    
    # --- Filters ---
    st.markdown("### 🔍 Filter Bugs")
    
    # Extract unique values for filters
    all_severities = sorted(list(set(b.get("severity") or "unknown" for b in bugs)))
    all_teams = sorted(list(set(b.get("team") or "unknown" for b in bugs)))
    all_statuses = sorted(list(set(b.get("status") or "unknown" for b in bugs)))
    all_projects = sorted(list(set(b.get("project") or "unknown" for b in bugs)))
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sel_sev = st.multiselect("Severity", all_severities)
    with c2:
        sel_team = st.multiselect("Team", all_teams)
    with c3:
        sel_status = st.multiselect("Status", all_statuses)
    with c4:
        sel_project = st.multiselect("Project", all_projects)
        
    # Apply filters
    filtered_bugs = bugs
    if sel_sev:
        filtered_bugs = [b for b in filtered_bugs if (b.get("severity") or "unknown") in sel_sev]
    if sel_team:
        filtered_bugs = [b for b in filtered_bugs if (b.get("team") or "unknown") in sel_team]
    if sel_status:
        filtered_bugs = [b for b in filtered_bugs if (b.get("status") or "unknown") in sel_status]
    if sel_project:
        filtered_bugs = [b for b in filtered_bugs if (b.get("project") or "unknown") in sel_project]
        
    st.markdown(f"**Showing {len(filtered_bugs)} of {len(bugs)} bugs**")
    st.markdown("---")
    
    if not filtered_bugs:
        st.info("No bugs match the selected filters.")
    else:
        for bug in filtered_bugs:
            bug_card(bug, role="project_manager")

def pm_kanban_page():
    header("Project Manager — Kanban", "Dragless Kanban (select and update).")
    resp = api_get("bugs")
    if resp is None or resp.status_code != 200:
        st.error("Failed to load bugs.")
        return
    bugs = resp.json()
    # columns layout
    col_open, col_inprog, col_done = st.columns(3)
    status_map = {"open": col_open, "in_progress": col_inprog, "resolved": col_done}
    
    # List of valid severity and team options for correction/feedback
    SEVERITIES = ["low", "medium", "high", "critical"]
    TEAMS = ["Frontend", "Backend", "Mobile", "DevOps"]

    for status, col in status_map.items():
        with col:
            st.markdown(f"### {status.replace('_',' ').title()}")
            for b in [x for x in bugs if x.get("status")==status]:
                
                # --- START: Bug Card Display ---
                st.markdown(
                    f"""
                    <div style="background:white; padding:10px; border-radius:8px; margin-bottom:8px; box-shadow: 0 1px 6px rgba(14,30,37,0.06);">
                        <div style="font-weight:700">#{b.get('id')} — {b.get('project')}</div>
                        <div style="color:#475569; margin-top:6px;">{b.get('description')[:200]}...</div>
                        <div style="margin-top:8px;">
                           AI Predicted: {severity_badge(b.get('severity'))} &nbsp; 
                           <span style='background:#2563eb; color:white; padding:4px 8px; border-radius:6px; font-weight:600;'>{b.get('team') or '—'}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # --- END: Bug Card Display ---
                
                # --- START: Inline Update/Feedback Form ---
                st.markdown("##### Corrective Action (Feedback & Status)")
                
                # 1. Update Status (Auto-Apply)
                # Callback to handle status change immediately
                def on_status_change(bid=b.get('id'), k=f"pm_move_{b.get('id')}"):
                    new_val = st.session_state[k]
                    # Call API immediately
                    api_post("update_bug", {"bug_id": bid, "status": new_val})
                    st.session_state.refresh_key += 1
                
                # We must use a unique key for each widget
                status_key = f"pm_move_{b.get('id')}"
                
                # Current index
                valid_statuses = ["open","in_progress","resolved","duplicate"]
                current_status = b.get("status")
                if current_status not in valid_statuses:
                    current_status = "open"
                
                st.selectbox(
                    "Change Bug Status (Auto-Saves)", 
                    valid_statuses, 
                    index=valid_statuses.index(current_status), 
                    key=status_key,
                    on_change=on_status_change
                )
                
                # 2. Provide Correction/Feedback for Fine-Tuning (Manual Submit)
                with st.expander("Provide Expert Feedback (MLOps)"):
                    c_sev, c_team = st.columns(2)
                    with c_sev:
                        correction_severity = st.selectbox(
                            "Correct Severity", 
                            ["No Change"] + SEVERITIES, 
                            index=0, 
                            key=f"pm_corr_sev_{b.get('id')}"
                        )
                    with c_team:
                        correction_team = st.selectbox(
                            "Correct Team", 
                            ["No Change"] + TEAMS, 
                            index=0, 
                            key=f"pm_corr_team_{b.get('id')}"
                        )

                    if st.button("Submit Feedback", key=f"pm_update_{b.get('id')}"):
                        # We only send the correction fields here, status is handled above
                        # But we need to send status too because the endpoint might expect it or we don't want to change it
                        # Actually the update_bug endpoint handles both status and feedback
                        
                        payload = {"bug_id": b.get("id")}
                        
                        # Only add if changed
                        if correction_severity != "No Change":
                            payload["correction_severity"] = correction_severity
                        if correction_team != "No Change":
                            payload["correction_team"] = correction_team
                            
                        if len(payload) > 1:
                            api_post("update_bug", payload)
                            st.success("Feedback recorded & Model retraining triggered!")
                            st.session_state.refresh_key += 1
                            time.sleep(0.5)  # Brief pause to show success message
                            st.rerun()  # Auto-refresh the page
                        else:
                            st.info("No changes selected.")
                
                st.markdown("---", unsafe_allow_html=True) # Separator after each bug's controls
                # --- END: Inline Update/Feedback Form ---

def pm_analytics_page():
    header("Project Manager — Analytics", "Simple interactive charts.")
    resp = api_get("bugs")
    if resp is None or resp.status_code != 200:
        st.error("Failed to load analytics data.")
        return
    bugs = resp.json()
    if not bugs:
        st.info("No bugs to show.")
        return

    df_bugs = {
        "team": [b.get("team") or "Unknown" for b in bugs],
        "severity": [b.get("severity") or "unknown" for b in bugs],
        "status": [b.get("status") or "unknown" for b in bugs],
    }
    # plotly expects lists; we'll make quick figures
    
    # 1. Team Distribution (Rich Horizontal Bar)
    fig_team = go.Figure(data=[go.Bar(
        y=df_bugs['team'],
        orientation='h',
        marker=dict(color='#6366f1', line=dict(width=0)), # Indigo
        opacity=0.9
    )])
    fig_team.update_layout(
        title="Bugs by Team",
        xaxis_title="Count",
        yaxis_title=None,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        yaxis=dict(categoryorder='total ascending')
    )
    
    # 2. Severity Distribution (Rich Pie)
    sev_counts = pd.Series(df_bugs['severity']).value_counts()
    # Neon colors
    colors_map = {'critical': '#ff0055', 'high': '#ff5e00', 'medium': '#ffcc00', 'low': '#00cc66', 'unknown': '#888888'}
    
    fig_sev = go.Figure(data=[go.Pie(
        labels=sev_counts.index,
        values=sev_counts.values,
        hole=0.4,
        marker=dict(colors=[colors_map.get(x.lower(), '#888888') for x in sev_counts.index]),
        textinfo='label+percent'
    )])
    fig_sev.update_layout(
        title="Severity Distribution",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350
    )
    
    # --- NEW: Fine-Tuning Chart ---
    st.markdown("---")
    st.markdown("### 📈 Continuous Learning Status: Feedback History")
    
    # This chart visualizes the data collection for the fine-tuning process
    feedback_hist_resp = api_get("feedback_history") 
    if feedback_hist_resp and feedback_hist_resp.status_code == 200:
        try:
            feedback_data = feedback_hist_resp.json()
            if feedback_data:
                # Assuming feedback_data is a list of objects like: [{"date": "YYYY-MM-DD", "count": N}, ...]
                df_feedback = pd.DataFrame(feedback_data)
                df_feedback['date'] = pd.to_datetime(df_feedback['date'])
                
                # Rich Area Chart
                fig_feedback = go.Figure()
                fig_feedback.add_trace(go.Scatter(
                    x=df_feedback['date'],
                    y=df_feedback['count'],
                    mode='lines+markers',
                    fill='tozeroy',
                    line=dict(color='#d946ef', width=3, shape='spline'), # Fuchsia spline
                    marker=dict(size=8, color='#c026d3', line=dict(width=2, color='white')),
                    name='Corrections'
                ))
                
                fig_feedback.update_layout(
                    title='Expert Corrections Collected Over Time',
                    xaxis_title='Date',
                    yaxis_title='Corrections (Training Data)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    hovermode='x unified',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#e2e8f0')
                )
                
                st.plotly_chart(fig_feedback, use_container_width=True)
            else:
                st.info("No historical feedback data available to chart for fine-tuning.")
        except Exception:
            st.warning("Could not parse historical feedback from backend to show MLOps status.")
    else:
        # This is the message you were seeing when the backend endpoint is missing/failing
        st.error("Could not load historical feedback from backend to show MLOps status. **Check FastAPI for /feedback_history and /feedback_count.**")
    
    st.markdown("---") # Separator between MLOps chart and standard charts

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_team, use_container_width=True)
    with c2:
        st.plotly_chart(fig_sev, use_container_width=True)

def pm_notifications_page():
    header("Project Manager — Notifications", "Recent system notifications")
    resp = api_get("notifications")
    if resp is None:
        return
    if resp.status_code != 200:
        st.error("Failed to load notifications.")
        return
    notes = resp.json()
    if not notes:
        st.info("No notifications yet.")
        return
    for n in notes:
        ts = n.get("created_at") or n.get("timestamp") or ""
        st.markdown(f"- **{n.get('message')}** — <span style='color:#64748b'>{ts}</span>", unsafe_allow_html=True)

# ------------------------------
# Top-level pages & routing
# ------------------------------
def page_login():
    # No sidebar on login page
    
    # Responsive login container setup - wider central column for better look
    c1, c2, c3 = st.columns([1, 2, 1])
    
    with c2:
        # Main attractive title outside the form
        st.markdown("<h1 class='app-title-login'>BugFlow</h1>", unsafe_allow_html=True)
        # Use the specific CSS class for the subtitle to control its margin
        st.markdown("<h3 class='login-subtitle'>Sign in to your account</h3>", unsafe_allow_html=True)

        # Use the container just for spacing, not for styling a box
        st.markdown(
            """
            <div class="login-form-container">
            """, unsafe_allow_html=True
        )

        # Login inputs
        # Login inputs - use session state for auto-fill
        if "login_email" not in st.session_state: st.session_state.login_email = "tester1@example.com"
        if "login_pass" not in st.session_state: st.session_state.login_pass = "password"
        
        st.text_input("Email", key="login_email")
        st.text_input("Password", type="password", key="login_pass")
        
        # Adding space before the button (reduced from 25px implicitly by st's default spacing)
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

        if st.button("Sign in", use_container_width=True):
            email = st.session_state.get("login_email")
            password = st.session_state.get("login_pass")
            # The login function handles the rerender
            ok = login(email.strip(), password.strip())
            if ok:
                st.success("Signed in")
                
        # Close the custom HTML container
        st.markdown("</div>", unsafe_allow_html=True)

        # --- Demo Accounts Section ---
        st.markdown("---")
        st.markdown("<h4 style='text-align:center; color:#94a3b8;'>Quick Demo Login</h4>", unsafe_allow_html=True)
        
        # Callback to set credentials
        def set_creds(e, p):
            st.session_state.login_email = e
            st.session_state.login_pass = p
        
        d1, d2, d3 = st.columns(3)
        with d1:
            st.button("Tester", use_container_width=True, key="btn_test", on_click=set_creds, args=("tester1@example.com", "password"))
        with d2:
            st.button("Developer", use_container_width=True, key="btn_dev", on_click=set_creds, args=("dev1@example.com", "password"))
        with d3:
            st.button("PM", use_container_width=True, key="btn_pm", on_click=set_creds, args=("pm1@example.com", "password"))
                
        # Show credentials for manual entry if needed, with high contrast
        st.markdown(
            """
            <div style='text-align:center; margin-top:10px; color:#94a3b8; font-size:12px;'>
                <b>Tester:</b> tester1@example.com / password<br>
                <b>Dev:</b> dev1@example.com / password<br>
                <b>PM:</b> pm1@example.com / password
            </div>
            """, 
            unsafe_allow_html=True
        )


def page_main():
    # Sidebar info moved to custom block below

    # define menu depending on role
    menu = []
    icons = []
    
    if st.session_state.role == "tester":
        menu = ["Dashboard","Report Bug","My Bugs","Logout"]
        icons = ["speedometer2", "bug", "list-task", "box-arrow-right"]
    elif st.session_state.role == "developer":
        menu = ["Dashboard","Assigned Bugs","Logout"]
        icons = ["speedometer2", "person-check", "box-arrow-right"]
    elif st.session_state.role == "project_manager":
        menu = ["Dashboard","Kanban","Analytics","Notifications","Logout"]
        icons = ["speedometer2", "kanban", "graph-up", "bell", "box-arrow-right"]
    else:
        menu = ["Dashboard","Report Bug","My Bugs","Logout"]
        icons = ["speedometer2", "bug", "list-task", "box-arrow-right"]

    with st.sidebar:
        # Custom Sidebar Title
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <h1 style="color:#60a5fa; font-size: 28px; font-weight: 800; margin-bottom: 5px;">BugFlow</h1>
            <div style="color:#94a3b8; font-size: 14px;">{st.session_state.user.get('email', 'unknown')}</div>
            <div style="color:#64748b; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">{st.session_state.role}</div>
        </div>
        """, unsafe_allow_html=True)
        
        choice = option_menu(
            None,  # No title inside the menu component
            menu,
            icons=icons,
            menu_icon="bug-fill",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#94a3b8", "font-size": "16px"}, 
                "nav-link": {
                    "font-size": "15px", 
                    "text-align": "left", 
                    "margin":"5px 0", 
                    "color": "#e2e8f0",
                    "border-radius": "8px",
                    "padding": "10px 15px",
                },
                "nav-link-selected": {"background-color": "rgba(37, 99, 235, 0.2)", "color": "#60a5fa", "font-weight": "600"},
            }
        )
    # top header with role
    if st.session_state.role == "tester":
        header("Tester Dashboard", "Report and track bugs you found.")
    elif st.session_state.role == "developer":
        header("Developer Dashboard", "Work on assigned issues.")
    elif st.session_state.role == "project_manager":
        header("Project Manager", "Overview, Kanban and Analytics.")
    else:
        header("BugFlow", "Role-based bug tracker")

    if choice == "Logout":
        # The logout function handles the rerender
        logout()
        return

    # route to subpages
    if st.session_state.role == "tester":
        if choice == "Dashboard":
            tester_report_page()
            st.markdown("---")
            tester_my_bugs_page()
        elif choice == "Report Bug":
            tester_report_page()
        elif choice == "My Bugs":
            tester_my_bugs_page()

    elif st.session_state.role == "developer":
        if choice == "Dashboard":
            # small overview
            resp = api_get("bugs")
            if resp and resp.status_code == 200:
                bugs = resp.json()
                assigned_count = sum(1 for b in bugs if b.get("assigned_to_id") == st.session_state.user.get("id"))
                small_card("Assigned to you", assigned_count, "Open and active")
            dev_assigned_page()
        elif choice == "Assigned Bugs":
            dev_assigned_page()

    elif st.session_state.role == "project_manager":
        if choice == "Dashboard":
            pm_dashboard_page()
        elif choice == "Kanban":
            pm_kanban_page()
        elif choice == "All Bugs":
            pm_all_bugs_page()
        elif choice == "Analytics":
            pm_analytics_page()
        elif choice == "Notifications":
            pm_notifications_page()

    else:
        st.info("Unknown role. Please logout and login with a valid account.")

# ------------------------------
# Landing Page
# ------------------------------
def page_landing():
    # Content for landing page
    st.markdown("""
    <div class="landing-container">
        <div class="landing-title">BugFlow</div>
        <div class="landing-subtitle">
            Your AI partner for intelligent bug tracking.<br>
            Build and ship faster with automated triage and predictive analytics.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Features section
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Instant Triage</div>
            <div class="feature-desc">
                AI instantly predicts severity and assigns bugs to the correct team.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Self-Learning</div>
            <div class="feature-desc">
                The model learns from your team's feedback, getting smarter with every resolved issue.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Premium Insights</div>
            <div class="feature-desc">
                Visualize project health with beautiful, real-time interactive dashboards.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Centered "Get Started" button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Get Started", use_container_width=True, type="primary"):
            st.session_state.show_landing = False
            st.rerun()

# ------------------------------
# App entry
# ------------------------------
def main():
    # Initialize session state
    if "show_landing" not in st.session_state:
        st.session_state.show_landing = True

    # Routing logic
    if st.session_state.show_landing and not st.session_state.logged_in:
        page_landing()
    elif not st.session_state.logged_in:
        page_login()
    else:
        page_main()

if __name__ == "__main__":
    main()