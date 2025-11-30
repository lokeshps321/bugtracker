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

# ------------------------------
# Page config - must be first Streamlit command
# ------------------------------
st.set_page_config(page_title="BugFlow", layout="wide")

# Inject Custom CSS for attractive login form and general styling
st.markdown(
    """
    <style>
    /* Styling for the central login form area (no more explicit white box/shadow) */
    .login-form-container {
        padding: 0px 20px; /* Greatly reduced padding for better fit on smaller screens */
        margin-top: 10px;
        width: 100%;
    }
    
    /* Styling for the primary sign-in button */
    .stButton>button {
        background-color: #1a73e8; /* Google Blue-like color */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #1764cc; /* Slightly darker on hover */
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2); /* Shadow on hover for responsiveness */
        transform: translateY(-2px);
    }
    
    /* Center and style the main application title on the login page */
    .app-title-login {
        text-align: center;
        color: #1a73e8; 
        margin-bottom: 5px; /* Reduced bottom margin for the main title */
        font-size: 38px; /* Larger title */
        font-weight: 800;
        letter-spacing: 1.5px;
        margin-top: 50px; /* Add space above the title */
    }

    /* Style for the sidebar header for consistency */
    .st-emotion-cache-vk3wp9, .st-emotion-cache-vk3wp9 > h1 {
        color: #1a73e8 !important; /* Make sidebar header blue */
    }

    /* Input styling for minimalist look */
    .stTextInput>div>div>input {
        border-bottom: 2px solid #ccc; /* Simple bottom border instead of a full box */
        border-top: none;
        border-left: none;
        border-right: none;
        border-radius: 0;
        padding: 8px 0;
    }
    .stTextInput>div>div>input:focus {
        border-bottom: 2px solid #1a73e8; /* Blue focus line */
        box-shadow: none;
    }

    /* Target the subtitle H3 to control its margin */
    .login-subtitle {
        text-align: center; 
        color: #6b7280; 
        margin-bottom: 10px; /* Greatly reduced from 30px to close the gap */
        font-size: 20px; /* Consistent font size */
    }
    
    /* Modern Sidebar Styling - Removed hardcoded bg for dark mode compatibility */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(49, 51, 63, 0.2);
    }
    
    /* Hide default radio buttons if any remain */
    .stRadio > div {
        display: none;
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
        r = requests.post(f"{API_URL.rstrip('/')}/{endpoint.lstrip('/')}", json=json_data, headers=_headers(), timeout=8)
        return r
    except Exception as e:
        st.error(f"Network error calling {endpoint}: {e}")
        return None

def api_get(endpoint, params=None):
    try:
        r = requests.get(f"{API_URL.rstrip('/')}/{endpoint.lstrip('/')}", headers=_headers(), params=params, timeout=8)
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
        <div style="background:white; padding:16px; border-radius:10px; margin-bottom:12px; box-shadow: 0 2px 8px rgba(14,30,37,0.08); border-left: 4px solid #3b82f6;">
            <div style="display:flex; justify-content:space-between; align-items:start;">
                <div>
                    <div style="font-weight:700; font-size:16px; color:#1e293b;">#{bug.get('id')} — {title}</div>
                    <div style="color:#64748b; margin-top:2px; font-size:12px;">{bug.get('project')}</div>
                    <div style="color:#475569; margin-top:6px; font-size:14px;">{bug.get('description')[:100]}...</div>
                </div>
                <div style="text-align:right;">
                    {severity_badge(bug.get('severity'))}
                    <div style="margin-top:6px;"><span style='background:#eff6ff; color:#2563eb; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:600;'>{bug.get('team') or 'Unassigned'}</span></div>
                </div>
            </div>
            <div style="margin-top:12px; padding-top:12px; border-top:1px solid #f1f5f9; display:flex; justify-content:space-between; align-items:center; font-size:12px; color:#94a3b8;">
                <div>Status: <span style="font-weight:600; color:#475569; text-transform:uppercase;">{bug.get('status')}</span></div>
                <div>Reported: {bug.get('created_at', '').split('T')[0]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def small_card(title, value, subtitle="", bg_color="white"):
    st.markdown(
        f"""
        <div style="
            background: {bg_color};
            padding:12px;
            border-radius:8px;
            box-shadow: 0 1px 6px rgba(14,30,37,0.08);
            margin-bottom:8px;">
            <div style="font-size:14px; color:#64748b">{title}</div>
            <div style="font-size:20px; font-weight:700">{value}</div>
            <div style="font-size:12px; color:#94a3b8">{subtitle}</div>
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
                resp = api_post("predict", {"description": desc, "project": project})
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

    for b in bugs:
        st.markdown(
            f"""
            <div style="background:white; padding:12px; border-radius:8px; margin-bottom:8px; box-shadow: 0 1px 6px rgba(14,30,37,0.06);">
                <div style="display:flex; justify-content:space-between;">
                    <div style="font-weight:700">#{b.get('id')} — {b.get('project')}</div>
                    <div>{severity_badge(b.get('severity'))} &nbsp; <span style='background:#2563eb; color:white; padding:4px 8px; border-radius:6px; font-weight:600;'>{b.get('team') or '—'}</span></div>
                </div>
                <div style="color:#475569; margin-top:8px;">{b.get('description')}</div>
                <div style="margin-top:8px; color:#94a3b8; font-size:12px;">Status: {b.get('status')} — Reported: {b.get('created_at', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
                        st.success("Bug updated")
                        st.session_state.refresh_key += 1
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
                        payload = {"bug_id": b.get('id'), "status": b.get("status")} # Keep current status

                        if correction_severity != "No Change":
                            payload["correction_severity"] = correction_severity
                        if correction_team != "No Change":
                            payload["correction_team"] = correction_team

                        upd = api_post("update_bug", payload)

                        if upd is None:
                            st.error("No response from backend.")
                        elif upd.status_code == 200:
                            response_msg = upd.json().get("message", "")
                            if "(AI Feedback Recorded" in response_msg:
                                st.success("✅ Feedback Sent & Model Retraining Triggered!")
                            else:
                                st.success("✅ Feedback Sent!")
                            st.session_state.refresh_key += 1
                        else:
                            st.error(f"❌ Failed: {upd.text}")
                
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
        st.markdown("<h4 style='text-align:center; color:#64748b;'>Quick Demo Login</h4>", unsafe_allow_html=True)
        
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
            <div style='text-align:center; margin-top:10px; color:#64748b; font-size:12px;'>
                <b>Tester:</b> tester1@example.com / password<br>
                <b>Dev:</b> dev1@example.com / password<br>
                <b>PM:</b> pm1@example.com / password
            </div>
            """, 
            unsafe_allow_html=True
        )


def page_main():
    st.sidebar.title("BugFlow")
    st.sidebar.markdown(f"Signed in as: **{st.session_state.user.get('email', 'unknown')}**")
    st.sidebar.markdown(f"Role: **{st.session_state.role}**")
    st.sidebar.markdown("---")

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
        choice = option_menu(
            "BugFlow", 
            menu,
            icons=icons,
            menu_icon="bug-fill",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"font-size": "14px"}, 
                "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px"},
                "nav-link-selected": {"background-color": "#2563eb", "font-weight": "600"},
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
    # Custom CSS for landing page
    st.markdown("""
    <style>
    .landing-container {
        max_width: 800px;
        margin: 0 auto;
        padding: 2rem;
        text-align: center;
    }
    .landing-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2563eb, #9333ea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .landing-subtitle {
        font-size: 1.5rem;
        color: #4b5563;
        margin-bottom: 3rem;
        line-height: 1.6;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 2rem;
        margin-bottom: 3rem;
        text-align: left;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    .feature-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .feature-desc {
        color: #6b7280;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="landing-container">
        <div class="landing-title">BugFlow</div>
        <div class="landing-subtitle">
            The intelligent bug tracking system powered by AI.<br>
            Automate triage, predict severity, and streamline your workflow.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Features section using Streamlit columns for better layout control
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">AI Classification</div>
            <div class="feature-desc">
                Automatically detects bug severity and assigns to the right team using fine-tuned ML models.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔄</div>
            <div class="feature-title">MLOps Feedback</div>
            <div class="feature-desc">
                Continuous learning loop. Correct AI predictions to improve model accuracy over time.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Smart Analytics</div>
            <div class="feature-desc">
                Real-time dashboards and Kanban boards to track project health and team velocity.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Centered "Get Started" button
    # Use responsive columns for button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Let's Get Started 🚀", use_container_width=True, type="primary"):
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