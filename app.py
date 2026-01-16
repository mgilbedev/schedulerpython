"""
Nuclear Maintenance Scheduling AI Agent - Dash Application
A comprehensive web application for predictive maintenance scheduling
in the nuclear industry using multi-agent supervision architecture.
"""

import dash
from dash import dcc, html, callback, Input, Output, State, clientside_callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
app.title = "Nuclear Maintenance AI Scheduler"

# =============================================================================
# SAMPLE DATA GENERATION
# =============================================================================

def generate_sensor_data():
    """Generate simulated AVEVA Historian sensor data"""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=720, freq="h")

    sensors = {
        "Reactor Coolant Pump A": {
            "vibration": np.random.normal(2.5, 0.3, 720) + np.linspace(0, 1.5, 720),
            "temperature": np.random.normal(285, 5, 720) + np.linspace(0, 15, 720),
            "pressure": np.random.normal(155, 2, 720),
            "failure_prob": np.clip(np.linspace(0.05, 0.85, 720) + np.random.normal(0, 0.05, 720), 0, 1)
        },
        "Steam Generator B": {
            "vibration": np.random.normal(1.8, 0.2, 720),
            "temperature": np.random.normal(275, 4, 720),
            "pressure": np.random.normal(68, 1.5, 720) - np.linspace(0, 5, 720),
            "failure_prob": np.clip(np.linspace(0.1, 0.45, 720) + np.random.normal(0, 0.03, 720), 0, 1)
        },
        "Containment Fan Unit C": {
            "vibration": np.random.normal(3.2, 0.4, 720) + np.sin(np.linspace(0, 10, 720)) * 0.5,
            "temperature": np.random.normal(45, 3, 720),
            "pressure": np.random.normal(1.02, 0.02, 720),
            "failure_prob": np.clip(0.15 + np.sin(np.linspace(0, 5, 720)) * 0.1 + np.random.normal(0, 0.02, 720), 0, 1)
        },
        "Emergency Diesel Generator 1": {
            "vibration": np.random.normal(4.5, 0.6, 720),
            "temperature": np.random.normal(95, 8, 720),
            "pressure": np.random.normal(6.5, 0.3, 720),
            "failure_prob": np.clip(np.random.normal(0.12, 0.04, 720), 0, 1)
        },
        "Main Feedwater Pump D": {
            "vibration": np.random.normal(2.1, 0.25, 720) + np.linspace(0, 0.8, 720),
            "temperature": np.random.normal(65, 4, 720),
            "pressure": np.random.normal(85, 2, 720),
            "failure_prob": np.clip(np.linspace(0.08, 0.55, 720) + np.random.normal(0, 0.04, 720), 0, 1)
        }
    }

    return dates, sensors


def generate_workers():
    """Generate worker data with skills"""
    workers = [
        {"id": "W001", "name": "John Mitchell", "skills": ["Reactor Systems", "Pumps", "Electrical"], "certification": "Senior", "availability": 0.9, "shift": "Day"},
        {"id": "W002", "name": "Sarah Chen", "skills": ["Instrumentation", "Controls", "Calibration"], "certification": "Senior", "availability": 0.85, "shift": "Day"},
        {"id": "W003", "name": "Michael Rodriguez", "skills": ["Mechanical", "Pumps", "Valves"], "certification": "Journeyman", "availability": 0.95, "shift": "Day"},
        {"id": "W004", "name": "Emily Watson", "skills": ["Electrical", "Motors", "Generators"], "certification": "Senior", "availability": 0.8, "shift": "Night"},
        {"id": "W005", "name": "David Kim", "skills": ["Reactor Systems", "Safety Systems"], "certification": "Senior", "availability": 0.9, "shift": "Day"},
        {"id": "W006", "name": "Lisa Thompson", "skills": ["HVAC", "Containment", "Ventilation"], "certification": "Journeyman", "availability": 0.88, "shift": "Night"},
        {"id": "W007", "name": "Robert Garcia", "skills": ["Welding", "Piping", "Mechanical"], "certification": "Senior", "availability": 0.92, "shift": "Day"},
        {"id": "W008", "name": "Amanda Foster", "skills": ["Instrumentation", "Radiation Monitoring"], "certification": "Senior", "availability": 0.87, "shift": "Day"},
        {"id": "W009", "name": "James Wilson", "skills": ["Diesel Generators", "Electrical", "Mechanical"], "certification": "Journeyman", "availability": 0.9, "shift": "Night"},
        {"id": "W010", "name": "Jennifer Lee", "skills": ["Controls", "PLC", "Safety Systems"], "certification": "Senior", "availability": 0.85, "shift": "Day"},
    ]
    return pd.DataFrame(workers)


def generate_maintenance_tasks():
    """Generate maintenance tasks based on predictive analysis"""
    tasks = [
        {"id": "MT001", "equipment": "Reactor Coolant Pump A", "task": "Bearing Replacement", "priority": "Critical",
         "skills_required": ["Pumps", "Mechanical"], "estimated_hours": 8, "due_date": "2024-01-05", "status": "Scheduled"},
        {"id": "MT002", "equipment": "Reactor Coolant Pump A", "task": "Vibration Analysis Follow-up", "priority": "High",
         "skills_required": ["Instrumentation"], "estimated_hours": 2, "due_date": "2024-01-03", "status": "In Progress"},
        {"id": "MT003", "equipment": "Steam Generator B", "task": "Pressure Sensor Calibration", "priority": "Medium",
         "skills_required": ["Instrumentation", "Calibration"], "estimated_hours": 4, "due_date": "2024-01-08", "status": "Pending"},
        {"id": "MT004", "equipment": "Containment Fan Unit C", "task": "Motor Inspection", "priority": "Medium",
         "skills_required": ["Electrical", "Motors"], "estimated_hours": 3, "due_date": "2024-01-06", "status": "Scheduled"},
        {"id": "MT005", "equipment": "Emergency Diesel Generator 1", "task": "Routine Testing", "priority": "High",
         "skills_required": ["Diesel Generators", "Electrical"], "estimated_hours": 6, "due_date": "2024-01-04", "status": "Scheduled"},
        {"id": "MT006", "equipment": "Main Feedwater Pump D", "task": "Seal Inspection", "priority": "High",
         "skills_required": ["Pumps", "Mechanical"], "estimated_hours": 5, "due_date": "2024-01-07", "status": "Pending"},
        {"id": "MT007", "equipment": "Reactor Coolant Pump A", "task": "Thermal Imaging Survey", "priority": "Medium",
         "skills_required": ["Instrumentation"], "estimated_hours": 2, "due_date": "2024-01-09", "status": "Pending"},
        {"id": "MT008", "equipment": "Steam Generator B", "task": "Tube Inspection", "priority": "High",
         "skills_required": ["Reactor Systems", "Instrumentation"], "estimated_hours": 12, "due_date": "2024-01-10", "status": "Pending"},
    ]
    return pd.DataFrame(tasks)


def generate_schedule():
    """Generate optimized schedule assignments"""
    schedule = [
        {"task_id": "MT002", "worker_id": "W002", "worker_name": "Sarah Chen", "start": "2024-01-03 08:00", "end": "2024-01-03 10:00", "equipment": "Reactor Coolant Pump A"},
        {"task_id": "MT005", "worker_id": "W009", "worker_name": "James Wilson", "start": "2024-01-04 20:00", "end": "2024-01-05 02:00", "equipment": "Emergency Diesel Generator 1"},
        {"task_id": "MT001", "worker_id": "W003", "worker_name": "Michael Rodriguez", "start": "2024-01-05 08:00", "end": "2024-01-05 16:00", "equipment": "Reactor Coolant Pump A"},
        {"task_id": "MT001", "worker_id": "W007", "worker_name": "Robert Garcia", "start": "2024-01-05 08:00", "end": "2024-01-05 16:00", "equipment": "Reactor Coolant Pump A"},
        {"task_id": "MT004", "worker_id": "W004", "worker_name": "Emily Watson", "start": "2024-01-06 20:00", "end": "2024-01-06 23:00", "equipment": "Containment Fan Unit C"},
        {"task_id": "MT006", "worker_id": "W003", "worker_name": "Michael Rodriguez", "start": "2024-01-07 08:00", "end": "2024-01-07 13:00", "equipment": "Main Feedwater Pump D"},
        {"task_id": "MT003", "worker_id": "W002", "worker_name": "Sarah Chen", "start": "2024-01-08 08:00", "end": "2024-01-08 12:00", "equipment": "Steam Generator B"},
        {"task_id": "MT007", "worker_id": "W008", "worker_name": "Amanda Foster", "start": "2024-01-09 08:00", "end": "2024-01-09 10:00", "equipment": "Reactor Coolant Pump A"},
        {"task_id": "MT008", "worker_id": "W005", "worker_name": "David Kim", "start": "2024-01-10 08:00", "end": "2024-01-10 20:00", "equipment": "Steam Generator B"},
        {"task_id": "MT008", "worker_id": "W002", "worker_name": "Sarah Chen", "start": "2024-01-10 08:00", "end": "2024-01-10 20:00", "equipment": "Steam Generator B"},
    ]
    return pd.DataFrame(schedule)


# Generate data
dates, sensors = generate_sensor_data()
workers_df = generate_workers()
tasks_df = generate_maintenance_tasks()
schedule_df = generate_schedule()


# =============================================================================
# NAVIGATION AND LAYOUT COMPONENTS
# =============================================================================

# Sidebar navigation
sidebar = html.Div(
    [
        html.Div(
            [
                html.I(className="fas fa-atom fa-2x", style={"color": "#00d4ff"}),
                html.H4("NuclAI Scheduler", className="ms-2 mb-0"),
            ],
            className="d-flex align-items-center mb-4 px-3 pt-3"
        ),
        html.Hr(style={"borderColor": "#444"}),
        dbc.Nav(
            [
                dbc.NavLink(
                    [html.I(className="fas fa-tachometer-alt me-2"), "Dashboard"],
                    href="/", active="exact", className="mb-2"
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-chart-line me-2"), "Predictive Analytics"],
                    href="/predictive", active="exact", className="mb-2"
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-users me-2"), "Workforce"],
                    href="/workforce", active="exact", className="mb-2"
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-calendar-alt me-2"), "Schedule"],
                    href="/schedule", active="exact", className="mb-2"
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-database me-2"), "Data Architecture"],
                    href="/data-model", active="exact", className="mb-2"
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-robot me-2"), "AI Agent Architecture"],
                    href="/agents", active="exact", className="mb-2"
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-dollar-sign me-2"), "Value & ROI"],
                    href="/value", active="exact", className="mb-2"
                ),
            ],
            vertical=True,
            pills=True,
        ),
        html.Div(
            [
                html.Hr(style={"borderColor": "#444"}),
                html.Div(
                    [
                        html.Span("Agent Status: ", style={"color": "#888"}),
                        html.Span("● Active", style={"color": "#00ff88"}),
                    ],
                    className="px-3 small"
                ),
                html.Div(
                    [
                        html.Span("MCP Servers: ", style={"color": "#888"}),
                        html.Span("8 Connected", style={"color": "#00d4ff"}),
                    ],
                    className="px-3 small mt-1"
                ),
            ],
            className="mt-auto pb-3"
        ),
    ],
    className="bg-dark vh-100 position-fixed",
    style={"width": "250px", "display": "flex", "flexDirection": "column"}
)

# Main content area
content = html.Div(
    id="page-content",
    style={"marginLeft": "250px", "padding": "20px"}
)

# App layout
app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content,
    dcc.Interval(id="interval-component", interval=2000, n_intervals=0),
    dcc.Interval(id="animation-interval", interval=500, n_intervals=0),
    dcc.Store(id="animation-state", data={"step": 0, "active_agents": []}),
])

# Add custom CSS via index string
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @keyframes pulse {
                0% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.6; transform: scale(1.05); }
                100% { opacity: 1; transform: scale(1); }
            }

            @keyframes flowRight {
                0% { transform: translateX(-20px); opacity: 0; }
                50% { opacity: 1; }
                100% { transform: translateX(20px); opacity: 0; }
            }

            @keyframes glow {
                0% { box-shadow: 0 0 5px rgba(0, 212, 255, 0.5); }
                50% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.8), 0 0 30px rgba(0, 212, 255, 0.4); }
                100% { box-shadow: 0 0 5px rgba(0, 212, 255, 0.5); }
            }

            @keyframes messageFlow {
                0% { opacity: 0; transform: scale(0.8) translateY(10px); }
                20% { opacity: 1; transform: scale(1) translateY(0); }
                80% { opacity: 1; transform: scale(1) translateY(0); }
                100% { opacity: 0; transform: scale(0.8) translateY(-10px); }
            }

            @keyframes dataStream {
                0% { background-position: 0% 50%; }
                100% { background-position: 100% 50%; }
            }

            .data-source {
                animation: pulse 2s ease-in-out infinite;
            }

            .data-flow-arrow {
                animation: flowRight 1.5s ease-in-out infinite;
            }

            .agent-active {
                animation: glow 2s ease-in-out infinite;
                border-radius: 8px;
            }

            .message-bubble {
                animation: messageFlow 4s ease-in-out infinite;
            }

            .message-bubble:nth-child(2) { animation-delay: 1s; }
            .message-bubble:nth-child(3) { animation-delay: 2s; }
            .message-bubble:nth-child(4) { animation-delay: 3s; }

            .star-schema-center {
                animation: pulse 3s ease-in-out infinite;
            }

            .dimension-table {
                transition: all 0.3s ease;
            }

            .dimension-table:hover {
                transform: scale(1.03);
                box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
            }

            .mcp-server-card {
                transition: all 0.3s ease;
                border: 2px solid transparent;
            }

            .mcp-server-card:hover {
                border-color: #00d4ff;
                transform: translateY(-3px);
            }

            .agent-node {
                transition: all 0.3s ease;
            }

            .agent-node:hover {
                transform: scale(1.1);
            }

            .data-stream-line {
                background: linear-gradient(90deg, transparent, #00d4ff, transparent);
                background-size: 200% 100%;
                animation: dataStream 2s linear infinite;
            }

            .typing-indicator {
                display: inline-flex;
                gap: 4px;
            }

            .typing-indicator span {
                width: 8px;
                height: 8px;
                background: #00d4ff;
                border-radius: 50%;
                animation: pulse 1s ease-in-out infinite;
            }

            .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
            .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# =============================================================================
# DASHBOARD PAGE
# =============================================================================

def create_dashboard():
    """Create the main dashboard overview"""

    # KPI Cards
    kpi_cards = dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle fa-2x", style={"color": "#ff6b6b"}),
                        html.Div([
                            html.H2("3", className="mb-0", style={"color": "#ff6b6b"}),
                            html.P("Critical Alerts", className="mb-0 text-muted small"),
                        ], className="ms-3")
                    ], className="d-flex align-items-center")
                ])
            ], className="bg-dark border-danger"),
            width=3
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-tools fa-2x", style={"color": "#ffd93d"}),
                        html.Div([
                            html.H2("8", className="mb-0", style={"color": "#ffd93d"}),
                            html.P("Scheduled Tasks", className="mb-0 text-muted small"),
                        ], className="ms-3")
                    ], className="d-flex align-items-center")
                ])
            ], className="bg-dark border-warning"),
            width=3
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-user-hard-hat fa-2x", style={"color": "#00d4ff"}),
                        html.Div([
                            html.H2("10", className="mb-0", style={"color": "#00d4ff"}),
                            html.P("Available Workers", className="mb-0 text-muted small"),
                        ], className="ms-3")
                    ], className="d-flex align-items-center")
                ])
            ], className="bg-dark border-info"),
            width=3
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-check-circle fa-2x", style={"color": "#00ff88"}),
                        html.Div([
                            html.H2("94%", className="mb-0", style={"color": "#00ff88"}),
                            html.P("Schedule Efficiency", className="mb-0 text-muted small"),
                        ], className="ms-3")
                    ], className="d-flex align-items-center")
                ])
            ], className="bg-dark border-success"),
            width=3
        ),
    ], className="mb-4")

    # Equipment health overview
    equipment_health = []
    for equip, data in sensors.items():
        health = 100 - (data["failure_prob"][-1] * 100)
        equipment_health.append(
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H6(equip, className="text-truncate mb-2"),
                        dbc.Progress(
                            value=health,
                            color="danger" if health < 30 else "warning" if health < 60 else "success",
                            className="mb-2",
                            style={"height": "10px"}
                        ),
                        html.Div([
                            html.Span(f"Health: {health:.0f}%", className="small"),
                            html.Span(f"Fail Prob: {data['failure_prob'][-1]*100:.1f}%",
                                     className="small text-danger ms-2" if data['failure_prob'][-1] > 0.5 else "small text-muted ms-2")
                        ])
                    ])
                ], className="bg-dark h-100"),
                width=True, className="mb-3"
            )
        )

    equipment_row = dbc.Row(equipment_health, className="mb-4")

    # Recent alerts timeline
    alerts = [
        {"time": "2 min ago", "type": "critical", "message": "Reactor Coolant Pump A vibration exceeding threshold", "icon": "exclamation-circle"},
        {"time": "15 min ago", "type": "warning", "message": "Main Feedwater Pump D showing degradation trend", "icon": "exclamation-triangle"},
        {"time": "1 hour ago", "type": "info", "message": "Scheduled maintenance MT005 completed successfully", "icon": "check-circle"},
        {"time": "3 hours ago", "type": "warning", "message": "Steam Generator B pressure trending below normal", "icon": "exclamation-triangle"},
        {"time": "5 hours ago", "type": "info", "message": "Worker shift change completed - Night shift active", "icon": "users"},
    ]

    alert_items = []
    for alert in alerts:
        color = "#ff6b6b" if alert["type"] == "critical" else "#ffd93d" if alert["type"] == "warning" else "#00d4ff"
        alert_items.append(
            html.Div([
                html.Div([
                    html.I(className=f"fas fa-{alert['icon']}", style={"color": color}),
                ], className="me-3"),
                html.Div([
                    html.P(alert["message"], className="mb-0"),
                    html.Small(alert["time"], className="text-muted")
                ])
            ], className="d-flex align-items-start mb-3 pb-3 border-bottom border-secondary")
        )

    # Upcoming schedule preview
    upcoming_tasks = schedule_df.head(5)
    schedule_items = []
    for _, task in upcoming_tasks.iterrows():
        schedule_items.append(
            html.Div([
                html.Div([
                    html.Strong(task["equipment"]),
                    html.Span(f" - {task['worker_name']}", className="text-muted")
                ]),
                html.Small(f"{task['start']} to {task['end']}", className="text-info")
            ], className="mb-3 pb-2 border-bottom border-secondary")
        )

    return html.Div([
        html.H2("Nuclear Maintenance Dashboard", className="mb-4"),
        html.P("Real-time overview of plant maintenance status and AI-driven scheduling", className="text-muted mb-4"),
        kpi_cards,
        html.H5("Equipment Health Status", className="mb-3"),
        equipment_row,
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-bell me-2"),
                        "Recent Alerts"
                    ]),
                    dbc.CardBody(alert_items, style={"maxHeight": "400px", "overflowY": "auto"})
                ], className="bg-dark")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-calendar me-2"),
                        "Upcoming Scheduled Tasks"
                    ]),
                    dbc.CardBody(schedule_items, style={"maxHeight": "400px", "overflowY": "auto"})
                ], className="bg-dark")
            ], width=6),
        ])
    ])


# =============================================================================
# PREDICTIVE ANALYTICS PAGE
# =============================================================================

def create_predictive_page():
    """Create predictive maintenance analytics page"""

    # Create sensor trend charts
    fig_vibration = go.Figure()
    fig_temperature = go.Figure()
    fig_pressure = go.Figure()
    fig_failure = go.Figure()

    for equip, data in sensors.items():
        fig_vibration.add_trace(go.Scatter(x=dates, y=data["vibration"], name=equip, mode="lines"))
        fig_temperature.add_trace(go.Scatter(x=dates, y=data["temperature"], name=equip, mode="lines"))
        fig_pressure.add_trace(go.Scatter(x=dates, y=data["pressure"], name=equip, mode="lines"))
        fig_failure.add_trace(go.Scatter(x=dates, y=data["failure_prob"]*100, name=equip, mode="lines", fill="tozeroy"))

    chart_template = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#fff"},
        "xaxis": {"gridcolor": "#333", "showgrid": True},
        "yaxis": {"gridcolor": "#333", "showgrid": True},
        "legend": {"orientation": "h", "y": -0.2}
    }

    fig_vibration.update_layout(title="Vibration Trends (mm/s)", height=300, **chart_template)
    fig_temperature.update_layout(title="Temperature Trends (°C)", height=300, **chart_template)
    fig_pressure.update_layout(title="Pressure Trends (bar)", height=300, **chart_template)
    fig_failure.update_layout(title="Predicted Failure Probability (%)", height=350, **chart_template)

    # AVEVA Historian integration card
    aveva_card = dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-industry me-2", style={"color": "#00d4ff"}),
            "AVEVA Historian Integration"
        ], className="bg-primary"),
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.I(className="fas fa-database fa-lg me-2", style={"color": "#00d4ff"}),
                    html.Span("Data Source Status")
                ], className="d-flex align-items-center mb-2"),
                html.Hr(className="my-2", style={"borderColor": "#444"}),
                html.Div([
                    html.Span("● ", style={"color": "#00ff88"}),
                    html.Span("Historian Server: ", className="text-muted"),
                    html.Span("Connected", style={"color": "#00ff88"})
                ], className="small mb-1"),
                html.Div([
                    html.Span("● ", style={"color": "#00ff88"}),
                    html.Span("Data Points/sec: ", className="text-muted"),
                    html.Span("12,847", style={"color": "#00d4ff"})
                ], className="small mb-1"),
                html.Div([
                    html.Span("● ", style={"color": "#00ff88"}),
                    html.Span("Tags Monitored: ", className="text-muted"),
                    html.Span("2,456", style={"color": "#00d4ff"})
                ], className="small mb-1"),
                html.Div([
                    html.Span("● ", style={"color": "#00ff88"}),
                    html.Span("Last Sync: ", className="text-muted"),
                    html.Span("2 sec ago", style={"color": "#00d4ff"})
                ], className="small"),
            ])
        ])
    ], className="bg-dark mb-4")

    # ML Model info card
    ml_card = dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-brain me-2"),
            "Predictive ML Model"
        ]),
        dbc.CardBody([
            html.Div([
                html.Span("Model Type: ", className="text-muted"),
                html.Span("LSTM Neural Network")
            ], className="mb-2"),
            html.Div([
                html.Span("Accuracy: ", className="text-muted"),
                html.Span("96.7%", style={"color": "#00ff88"})
            ], className="mb-2"),
            html.Div([
                html.Span("Training Data: ", className="text-muted"),
                html.Span("5 years historical")
            ], className="mb-2"),
            html.Div([
                html.Span("Features: ", className="text-muted"),
                html.Span("Vibration, Temp, Pressure, Runtime")
            ], className="mb-2"),
            html.Div([
                html.Span("Last Retrained: ", className="text-muted"),
                html.Span("2024-01-01")
            ]),
        ])
    ], className="bg-dark")

    return html.Div([
        html.H2("Predictive Maintenance Analytics", className="mb-4"),
        html.P("Sensor data from AVEVA Historian powering ML-based failure prediction", className="text-muted mb-4"),

        dbc.Row([
            dbc.Col([aveva_card, ml_card], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=fig_failure, config={"displayModeBar": False})
                    ])
                ], className="bg-dark mb-4"),
            ], width=9),
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=fig_vibration, config={"displayModeBar": False})
                    ])
                ], className="bg-dark")
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=fig_temperature, config={"displayModeBar": False})
                    ])
                ], className="bg-dark")
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=fig_pressure, config={"displayModeBar": False})
                    ])
                ], className="bg-dark")
            ], width=4),
        ])
    ])


# =============================================================================
# WORKFORCE PAGE
# =============================================================================

def create_workforce_page():
    """Create workforce management page"""

    # Skills matrix heatmap
    all_skills = set()
    for _, worker in workers_df.iterrows():
        all_skills.update(worker["skills"])
    all_skills = sorted(list(all_skills))

    skill_matrix = []
    for _, worker in workers_df.iterrows():
        row = [1 if skill in worker["skills"] else 0 for skill in all_skills]
        skill_matrix.append(row)

    fig_skills = go.Figure(data=go.Heatmap(
        z=skill_matrix,
        x=all_skills,
        y=workers_df["name"].tolist(),
        colorscale=[[0, "#1a1a2e"], [1, "#00d4ff"]],
        showscale=False
    ))
    fig_skills.update_layout(
        title="Worker Skills Matrix",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#fff"},
        height=400,
        xaxis={"tickangle": 45}
    )

    # Worker availability chart
    fig_availability = go.Figure(data=[
        go.Bar(
            x=workers_df["name"],
            y=workers_df["availability"] * 100,
            marker_color=["#00ff88" if a > 0.85 else "#ffd93d" if a > 0.7 else "#ff6b6b"
                         for a in workers_df["availability"]]
        )
    ])
    fig_availability.update_layout(
        title="Worker Availability (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#fff"},
        height=300,
        yaxis={"range": [0, 100], "gridcolor": "#333"},
        xaxis={"tickangle": 45}
    )

    # Worker cards
    worker_cards = []
    for _, worker in workers_df.iterrows():
        cert_color = "#00ff88" if worker["certification"] == "Senior" else "#ffd93d"
        shift_icon = "sun" if worker["shift"] == "Day" else "moon"

        skills_badges = [dbc.Badge(skill, className="me-1 mb-1", color="info") for skill in worker["skills"]]

        worker_cards.append(
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-user-circle fa-2x", style={"color": "#00d4ff"}),
                            ], className="me-3"),
                            html.Div([
                                html.H6(worker["name"], className="mb-0"),
                                html.Small(worker["id"], className="text-muted")
                            ])
                        ], className="d-flex align-items-center mb-3"),
                        html.Div([
                            dbc.Badge(worker["certification"], style={"backgroundColor": cert_color}, className="me-2"),
                            html.I(className=f"fas fa-{shift_icon}", style={"color": "#ffd93d" if worker["shift"] == "Day" else "#7c83fd"}),
                            html.Span(f" {worker['shift']} Shift", className="small text-muted")
                        ], className="mb-2"),
                        html.Div(skills_badges),
                        html.Div([
                            html.Small("Availability: ", className="text-muted"),
                            html.Small(f"{worker['availability']*100:.0f}%",
                                      style={"color": "#00ff88" if worker["availability"] > 0.85 else "#ffd93d"})
                        ], className="mt-2")
                    ])
                ], className="bg-dark h-100"),
                width=3, className="mb-3"
            )
        )

    return html.Div([
        html.H2("Workforce Management", className="mb-4"),
        html.P("Skills inventory and availability tracking for optimal task assignment", className="text-muted mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=fig_skills, config={"displayModeBar": False})
                    ])
                ], className="bg-dark")
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=fig_availability, config={"displayModeBar": False})
                    ])
                ], className="bg-dark")
            ], width=4),
        ], className="mb-4"),

        html.H5("Worker Directory", className="mb-3"),
        dbc.Row(worker_cards)
    ])


# =============================================================================
# SCHEDULE PAGE
# =============================================================================

def create_schedule_page():
    """Create scheduling page with Gantt chart"""

    # Prepare Gantt chart data
    gantt_data = []
    colors = {"Critical": "#ff6b6b", "High": "#ffd93d", "Medium": "#00d4ff", "Low": "#00ff88"}

    for _, row in schedule_df.iterrows():
        task_info = tasks_df[tasks_df["id"] == row["task_id"]].iloc[0] if len(tasks_df[tasks_df["id"] == row["task_id"]]) > 0 else None
        priority = task_info["priority"] if task_info is not None else "Medium"
        task_name = task_info["task"] if task_info is not None else row["task_id"]

        gantt_data.append({
            "Task": f"{row['equipment']}<br>{task_name}",
            "Start": row["start"],
            "Finish": row["end"],
            "Resource": row["worker_name"],
            "Priority": priority
        })

    gantt_df = pd.DataFrame(gantt_data)
    gantt_df["Start"] = pd.to_datetime(gantt_df["Start"])
    gantt_df["Finish"] = pd.to_datetime(gantt_df["Finish"])

    # Create Gantt chart
    fig_gantt = px.timeline(
        gantt_df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Priority",
        color_discrete_map=colors,
        hover_data=["Resource"]
    )
    fig_gantt.update_layout(
        title="Maintenance Schedule - Gantt View",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#fff"},
        height=500,
        xaxis={"gridcolor": "#333"},
        yaxis={"gridcolor": "#333"}
    )
    fig_gantt.update_yaxes(autorange="reversed")

    # Resource utilization
    worker_hours = schedule_df.copy()
    worker_hours["start"] = pd.to_datetime(worker_hours["start"])
    worker_hours["end"] = pd.to_datetime(worker_hours["end"])
    worker_hours["hours"] = (worker_hours["end"] - worker_hours["start"]).dt.total_seconds() / 3600

    utilization = worker_hours.groupby("worker_name")["hours"].sum().reset_index()

    fig_utilization = go.Figure(data=[
        go.Bar(
            x=utilization["worker_name"],
            y=utilization["hours"],
            marker_color="#00d4ff"
        )
    ])
    fig_utilization.update_layout(
        title="Worker Utilization (Hours Scheduled)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#fff"},
        height=300,
        yaxis={"gridcolor": "#333"},
        xaxis={"tickangle": 45}
    )

    # Task priority distribution
    priority_counts = tasks_df["priority"].value_counts()
    fig_priority = go.Figure(data=[
        go.Pie(
            labels=priority_counts.index,
            values=priority_counts.values,
            marker_colors=[colors.get(p, "#888") for p in priority_counts.index],
            hole=0.4
        )
    ])
    fig_priority.update_layout(
        title="Tasks by Priority",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#fff"},
        height=300,
        showlegend=True
    )

    # Task table
    task_table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("ID"),
                html.Th("Equipment"),
                html.Th("Task"),
                html.Th("Priority"),
                html.Th("Est. Hours"),
                html.Th("Due Date"),
                html.Th("Status")
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td(row["id"]),
                html.Td(row["equipment"]),
                html.Td(row["task"]),
                html.Td(dbc.Badge(row["priority"],
                        color="danger" if row["priority"] == "Critical" else
                              "warning" if row["priority"] == "High" else "info")),
                html.Td(row["estimated_hours"]),
                html.Td(row["due_date"]),
                html.Td(dbc.Badge(row["status"],
                        color="success" if row["status"] == "Scheduled" else
                              "primary" if row["status"] == "In Progress" else "secondary"))
            ]) for _, row in tasks_df.iterrows()
        ])
    ], striped=True, hover=True, responsive=True, className="bg-dark")

    return html.Div([
        html.H2("Maintenance Schedule", className="mb-4"),
        html.P("AI-optimized schedule balancing equipment criticality, worker skills, and availability", className="text-muted mb-4"),

        dbc.Card([
            dbc.CardBody([
                dcc.Graph(figure=fig_gantt, config={"displayModeBar": False})
            ])
        ], className="bg-dark mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=fig_utilization, config={"displayModeBar": False})
                    ])
                ], className="bg-dark")
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=fig_priority, config={"displayModeBar": False})
                    ])
                ], className="bg-dark")
            ], width=4),
        ], className="mb-4"),

        html.H5("Maintenance Tasks", className="mb-3"),
        dbc.Card([
            dbc.CardBody([task_table])
        ], className="bg-dark")
    ])


# =============================================================================
# DATA MODEL / ARCHITECTURE PAGE
# =============================================================================

def create_data_model_page():
    """Create data model visualization page with animated data flows and star schema"""

    # View selector
    view_selector = dbc.ButtonGroup([
        dbc.Button("Data Flow Animation", id="btn-data-flow", color="primary", outline=True, className="me-1"),
        dbc.Button("Star Schema", id="btn-star-schema", color="primary", outline=True, className="me-1"),
        dbc.Button("Source Systems", id="btn-source-systems", color="primary", outline=True),
    ], className="mb-4")

    return html.Div([
        html.H2("Data Architecture", className="mb-4"),
        html.P("Enterprise data integration from multiple source systems into a unified analytics model", className="text-muted mb-4"),

        view_selector,

        html.Div(id="data-model-content", children=[
            create_data_flow_view()
        ])
    ])


def create_data_flow_view():
    """Create animated data flow visualization"""

    # Create the animated data flow diagram
    fig = go.Figure()

    # Source systems (left side)
    sources = [
        {"name": "AVEVA Historian", "y": 0.85, "color": "#00d4ff", "icon": "database"},
        {"name": "SAP ERP", "y": 0.7, "color": "#ffd93d", "icon": "building"},
        {"name": "Salesforce CRM", "y": 0.55, "color": "#00ff88", "icon": "users"},
        {"name": "HR System", "y": 0.4, "color": "#ff6b6b", "icon": "id-card"},
        {"name": "Safety DB", "y": 0.25, "color": "#9b59b6", "icon": "shield-alt"},
        {"name": "Inventory System", "y": 0.1, "color": "#e67e22", "icon": "boxes"},
    ]

    # Add source nodes
    for src in sources:
        fig.add_trace(go.Scatter(
            x=[0.05], y=[src["y"]],
            mode="markers+text",
            marker=dict(size=40, color=src["color"], symbol="square"),
            text=[src["name"]],
            textposition="middle right",
            textfont=dict(size=11, color="#fff"),
            hoverinfo="text",
            hovertext=src["name"]
        ))

    # Data Lake (center-left)
    fig.add_trace(go.Scatter(
        x=[0.35], y=[0.5],
        mode="markers+text",
        marker=dict(size=80, color="#1a5276", symbol="diamond"),
        text=["Databricks<br>Lakehouse"],
        textposition="middle center",
        textfont=dict(size=12, color="#fff"),
    ))

    # Processing layer (center)
    fig.add_trace(go.Scatter(
        x=[0.55], y=[0.5],
        mode="markers+text",
        marker=dict(size=60, color="#00d4ff", symbol="hexagon"),
        text=["Delta Lake<br>Processing"],
        textposition="middle center",
        textfont=dict(size=10, color="#fff"),
    ))

    # Star Schema (center-right)
    fig.add_trace(go.Scatter(
        x=[0.75], y=[0.5],
        mode="markers+text",
        marker=dict(size=70, color="#ffd93d", symbol="star"),
        text=["Star<br>Schema"],
        textposition="middle center",
        textfont=dict(size=11, color="#000"),
    ))

    # AI/ML Layer (right)
    fig.add_trace(go.Scatter(
        x=[0.92], y=[0.5],
        mode="markers+text",
        marker=dict(size=50, color="#00ff88", symbol="circle"),
        text=["AI/ML<br>Models"],
        textposition="middle center",
        textfont=dict(size=10, color="#000"),
    ))

    # Draw flow lines with animation effect
    for src in sources:
        # Source to Data Lake
        fig.add_trace(go.Scatter(
            x=[0.12, 0.28], y=[src["y"], 0.5],
            mode="lines",
            line=dict(color=src["color"], width=2, dash="dot"),
            hoverinfo="none"
        ))

    # Data Lake to Processing
    fig.add_trace(go.Scatter(
        x=[0.42, 0.48], y=[0.5, 0.5],
        mode="lines",
        line=dict(color="#00d4ff", width=3),
        hoverinfo="none"
    ))

    # Processing to Star Schema
    fig.add_trace(go.Scatter(
        x=[0.62, 0.68], y=[0.5, 0.5],
        mode="lines",
        line=dict(color="#ffd93d", width=3),
        hoverinfo="none"
    ))

    # Star Schema to AI/ML
    fig.add_trace(go.Scatter(
        x=[0.82, 0.87], y=[0.5, 0.5],
        mode="lines",
        line=dict(color="#00ff88", width=3),
        hoverinfo="none"
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#fff"},
        height=500,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.05]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.05]),
        title="Data Flow: Source Systems to AI/ML Models"
    )

    # Data source cards with animation indicators
    source_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-database fa-2x me-3 data-source", style={"color": "#00d4ff"}),
                        html.Div([
                            html.H6("AVEVA Historian", className="mb-0"),
                            html.Small("Real-time sensor data", className="text-muted")
                        ])
                    ], className="d-flex align-items-center"),
                    html.Hr(style={"borderColor": "#444"}),
                    html.Div([
                        html.Span("● ", style={"color": "#00ff88"}),
                        html.Span("Streaming: ", className="text-muted"),
                        html.Span("12,847 pts/sec", className="text-info")
                    ], className="small"),
                    html.Div(id="historian-flow-indicator", className="mt-2", children=[
                        html.Div([
                            html.I(className="fas fa-arrow-right data-flow-arrow me-1", style={"color": "#00d4ff"}),
                            html.Span("Vibration, Temperature, Pressure", className="small text-muted")
                        ])
                    ])
                ])
            ], className="bg-dark mcp-server-card h-100")
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-building fa-2x me-3 data-source", style={"color": "#ffd93d"}),
                        html.Div([
                            html.H6("SAP ERP", className="mb-0"),
                            html.Small("Work orders & inventory", className="text-muted")
                        ])
                    ], className="d-flex align-items-center"),
                    html.Hr(style={"borderColor": "#444"}),
                    html.Div([
                        html.Span("● ", style={"color": "#00ff88"}),
                        html.Span("Batch: ", className="text-muted"),
                        html.Span("Every 15 min", className="text-info")
                    ], className="small"),
                    html.Div(className="mt-2", children=[
                        html.Div([
                            html.I(className="fas fa-arrow-right data-flow-arrow me-1", style={"color": "#ffd93d"}),
                            html.Span("Parts, Costs, Procurement", className="small text-muted")
                        ])
                    ])
                ])
            ], className="bg-dark mcp-server-card h-100")
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-users fa-2x me-3 data-source", style={"color": "#00ff88"}),
                        html.Div([
                            html.H6("Salesforce CRM", className="mb-0"),
                            html.Small("Customer & contract data", className="text-muted")
                        ])
                    ], className="d-flex align-items-center"),
                    html.Hr(style={"borderColor": "#444"}),
                    html.Div([
                        html.Span("● ", style={"color": "#00ff88"}),
                        html.Span("CDC: ", className="text-muted"),
                        html.Span("Real-time sync", className="text-info")
                    ], className="small"),
                    html.Div(className="mt-2", children=[
                        html.Div([
                            html.I(className="fas fa-arrow-right data-flow-arrow me-1", style={"color": "#00ff88"}),
                            html.Span("SLAs, Contracts, Contacts", className="small text-muted")
                        ])
                    ])
                ])
            ], className="bg-dark mcp-server-card h-100")
        ], width=4),
    ], className="mb-4")

    return html.Div([
        dbc.Card([
            dbc.CardBody([
                dcc.Graph(figure=fig, config={"displayModeBar": False})
            ])
        ], className="bg-dark mb-4"),

        html.H5("Source System Connections", className="mb-3"),
        source_cards,

        # Data pipeline stats
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-chart-bar me-2"),
                "Real-Time Data Pipeline Statistics"
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3("2.4M", className="mb-0", style={"color": "#00d4ff"}),
                            html.Small("Records/Hour", className="text-muted")
                        ], className="text-center")
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H3("99.7%", className="mb-0", style={"color": "#00ff88"}),
                            html.Small("Data Quality", className="text-muted")
                        ], className="text-center")
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H3("<500ms", className="mb-0", style={"color": "#ffd93d"}),
                            html.Small("Latency", className="text-muted")
                        ], className="text-center")
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H3("6", className="mb-0", style={"color": "#ff6b6b"}),
                            html.Small("Source Systems", className="text-muted")
                        ], className="text-center")
                    ], width=3),
                ])
            ])
        ], className="bg-dark")
    ])


def create_star_schema_view():
    """Create star schema visualization"""

    # Create star schema diagram
    fig = go.Figure()

    # Fact table (center)
    fig.add_trace(go.Scatter(
        x=[0.5], y=[0.5],
        mode="markers+text",
        marker=dict(size=100, color="#ff6b6b", symbol="square"),
        text=["FACT:<br>Maintenance<br>Events"],
        textposition="middle center",
        textfont=dict(size=12, color="#fff"),
    ))

    # Dimension tables (around the fact)
    dimensions = [
        {"name": "DIM:\nEquipment", "x": 0.5, "y": 0.9, "color": "#00d4ff"},
        {"name": "DIM:\nWorker", "x": 0.85, "y": 0.7, "color": "#00ff88"},
        {"name": "DIM:\nTime", "x": 0.85, "y": 0.3, "color": "#ffd93d"},
        {"name": "DIM:\nLocation", "x": 0.5, "y": 0.1, "color": "#9b59b6"},
        {"name": "DIM:\nParts", "x": 0.15, "y": 0.3, "color": "#e67e22"},
        {"name": "DIM:\nSafety", "x": 0.15, "y": 0.7, "color": "#1abc9c"},
    ]

    for dim in dimensions:
        # Draw connection line
        fig.add_trace(go.Scatter(
            x=[0.5, dim["x"]], y=[0.5, dim["y"]],
            mode="lines",
            line=dict(color="#444", width=3),
            hoverinfo="none"
        ))
        # Draw dimension node
        fig.add_trace(go.Scatter(
            x=[dim["x"]], y=[dim["y"]],
            mode="markers+text",
            marker=dict(size=70, color=dim["color"], symbol="square"),
            text=[dim["name"]],
            textposition="middle center",
            textfont=dict(size=10, color="#fff"),
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#fff"},
        height=600,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
        title="Star Schema: Maintenance Analytics Data Model"
    )

    # Dimension details
    dim_details = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("FACT: Maintenance Events", style={"backgroundColor": "#ff6b6b"}),
                dbc.CardBody([
                    html.H6("Measures:", className="text-info"),
                    html.Ul([
                        html.Li("event_id (PK)", className="small"),
                        html.Li("duration_hours", className="small"),
                        html.Li("labor_cost", className="small"),
                        html.Li("parts_cost", className="small"),
                        html.Li("downtime_minutes", className="small"),
                        html.Li("failure_prevented (bool)", className="small"),
                    ]),
                    html.H6("Foreign Keys:", className="text-info mt-3"),
                    html.Ul([
                        html.Li("equipment_key", className="small"),
                        html.Li("worker_key", className="small"),
                        html.Li("time_key", className="small"),
                        html.Li("location_key", className="small"),
                        html.Li("parts_key", className="small"),
                        html.Li("safety_key", className="small"),
                    ])
                ])
            ], className="bg-dark dimension-table h-100")
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("DIM: Equipment", style={"backgroundColor": "#00d4ff", "color": "#000"}),
                dbc.CardBody([
                    html.Ul([
                        html.Li("equipment_key (PK)", className="small"),
                        html.Li("equipment_id", className="small"),
                        html.Li("equipment_name", className="small"),
                        html.Li("equipment_type", className="small"),
                        html.Li("manufacturer", className="small"),
                        html.Li("install_date", className="small"),
                        html.Li("criticality_rating", className="small"),
                        html.Li("last_maintenance_date", className="small"),
                    ])
                ])
            ], className="bg-dark dimension-table mb-3"),
            dbc.Card([
                dbc.CardHeader("DIM: Worker", style={"backgroundColor": "#00ff88", "color": "#000"}),
                dbc.CardBody([
                    html.Ul([
                        html.Li("worker_key (PK)", className="small"),
                        html.Li("worker_id", className="small"),
                        html.Li("name", className="small"),
                        html.Li("certification_level", className="small"),
                        html.Li("skills (array)", className="small"),
                        html.Li("shift", className="small"),
                        html.Li("department", className="small"),
                    ])
                ])
            ], className="bg-dark dimension-table")
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("DIM: Time", style={"backgroundColor": "#ffd93d", "color": "#000"}),
                dbc.CardBody([
                    html.Ul([
                        html.Li("time_key (PK)", className="small"),
                        html.Li("date", className="small"),
                        html.Li("hour", className="small"),
                        html.Li("day_of_week", className="small"),
                        html.Li("month", className="small"),
                        html.Li("quarter", className="small"),
                        html.Li("year", className="small"),
                        html.Li("is_outage_window", className="small"),
                    ])
                ])
            ], className="bg-dark dimension-table mb-3"),
            dbc.Card([
                dbc.CardHeader("DIM: Parts", style={"backgroundColor": "#e67e22"}),
                dbc.CardBody([
                    html.Ul([
                        html.Li("parts_key (PK)", className="small"),
                        html.Li("part_number", className="small"),
                        html.Li("description", className="small"),
                        html.Li("supplier", className="small"),
                        html.Li("unit_cost", className="small"),
                        html.Li("lead_time_days", className="small"),
                    ])
                ])
            ], className="bg-dark dimension-table")
        ], width=4),
    ])

    return html.Div([
        dbc.Card([
            dbc.CardBody([
                dcc.Graph(figure=fig, config={"displayModeBar": False})
            ])
        ], className="bg-dark mb-4"),

        html.H5("Schema Details", className="mb-3"),
        dim_details
    ])


# =============================================================================
# AI AGENT ARCHITECTURE PAGE
# =============================================================================

def create_agents_page():
    """Create AI agent architecture visualization page with animation"""

    # View selector for different agent views
    view_selector = dbc.ButtonGroup([
        dbc.Button("Agent Overview", id="btn-agent-overview", color="primary", outline=True, className="me-1"),
        dbc.Button("MCP Servers", id="btn-mcp-servers", color="primary", outline=True, className="me-1"),
        dbc.Button("Agent Communication", id="btn-agent-comm", color="primary", outline=True),
    ], className="mb-4")

    return html.Div([
        html.H2("AI Agent Architecture", className="mb-4"),
        html.P("Multi-agent supervision system with MCP server integration for intelligent scheduling", className="text-muted mb-4"),

        view_selector,

        html.Div(id="agent-view-content", children=[
            create_agent_overview()
        ])
    ])


def create_agent_overview():
    """Create the agent overview with animated architecture"""

    # Multi-agent architecture diagram using Plotly
    fig_architecture = go.Figure()

    # Define node positions for architecture diagram
    nodes = {
        "supervisor": {"x": 0.5, "y": 0.92, "label": "Supervisor Agent", "color": "#ff6b6b", "size": 55},
        "predictive": {"x": 0.15, "y": 0.68, "label": "Predictive\nMaintenance", "color": "#00d4ff", "size": 45},
        "scheduling": {"x": 0.38, "y": 0.68, "label": "Scheduling\nOptimization", "color": "#00d4ff", "size": 45},
        "resource": {"x": 0.62, "y": 0.68, "label": "Resource\nAllocation", "color": "#00d4ff", "size": 45},
        "safety": {"x": 0.85, "y": 0.68, "label": "Safety &\nCompliance", "color": "#00d4ff", "size": 45},
    }

    # Define connections
    connections = [
        ("supervisor", "predictive"), ("supervisor", "scheduling"),
        ("supervisor", "resource"), ("supervisor", "safety"),
    ]

    # Draw connections
    for start, end in connections:
        fig_architecture.add_trace(go.Scatter(
            x=[nodes[start]["x"], nodes[end]["x"]],
            y=[nodes[start]["y"], nodes[end]["y"]],
            mode="lines",
            line=dict(color="#444", width=3),
            hoverinfo="none"
        ))

    # Draw nodes
    for node_id, node in nodes.items():
        fig_architecture.add_trace(go.Scatter(
            x=[node["x"]], y=[node["y"]],
            mode="markers+text",
            marker=dict(size=node["size"], color=node["color"], line=dict(color="#fff", width=2)),
            text=[node["label"]],
            textposition="bottom center",
            textfont=dict(size=11, color="#fff"),
        ))

    fig_architecture.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#fff"},
        height=350,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.05]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.4, 1.05]),
        margin=dict(l=20, r=20, t=20, b=20)
    )

    # Agent cards with detailed descriptions
    agent_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-crown me-2", style={"color": "#ff6b6b"}),
                    "Supervisor Agent"
                ], className="bg-dark"),
                dbc.CardBody([
                    html.P("Orchestrates all sub-agents and makes final scheduling decisions", className="small"),
                    html.Hr(style={"borderColor": "#444"}),
                    html.Strong("Responsibilities:", className="small"),
                    html.Ul([
                        html.Li("Coordinate agent communication", className="small"),
                        html.Li("Resolve scheduling conflicts", className="small"),
                        html.Li("Prioritize based on safety criticality", className="small"),
                        html.Li("Human-in-the-loop approval workflow", className="small"),
                    ], className="mb-0")
                ])
            ], className="bg-dark h-100 agent-active")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-chart-line me-2", style={"color": "#00d4ff"}),
                    "Predictive Maintenance"
                ], className="bg-dark"),
                dbc.CardBody([
                    html.P("Analyzes sensor data to predict equipment failures", className="small"),
                    html.Hr(style={"borderColor": "#444"}),
                    html.Strong("Capabilities:", className="small"),
                    html.Ul([
                        html.Li("Time-series anomaly detection", className="small"),
                        html.Li("Failure probability estimation", className="small"),
                        html.Li("RUL prediction", className="small"),
                        html.Li("Maintenance recommendations", className="small"),
                    ], className="mb-0")
                ])
            ], className="bg-dark h-100")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-calendar-check me-2", style={"color": "#00d4ff"}),
                    "Scheduling Optimization"
                ], className="bg-dark"),
                dbc.CardBody([
                    html.P("Creates optimal maintenance schedules", className="small"),
                    html.Hr(style={"borderColor": "#444"}),
                    html.Strong("Optimization Factors:", className="small"),
                    html.Ul([
                        html.Li("Equipment criticality ranking", className="small"),
                        html.Li("Worker skill matching", className="small"),
                        html.Li("Regulatory compliance", className="small"),
                        html.Li("Parts availability", className="small"),
                    ], className="mb-0")
                ])
            ], className="bg-dark h-100")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-shield-alt me-2", style={"color": "#00d4ff"}),
                    "Safety & Compliance"
                ], className="bg-dark"),
                dbc.CardBody([
                    html.P("Ensures nuclear safety regulations are met", className="small"),
                    html.Hr(style={"borderColor": "#444"}),
                    html.Strong("Compliance Checks:", className="small"),
                    html.Ul([
                        html.Li("NRC regulation validation", className="small"),
                        html.Li("Radiation exposure limits", className="small"),
                        html.Li("Required certifications", className="small"),
                        html.Li("Work permit requirements", className="small"),
                    ], className="mb-0")
                ])
            ], className="bg-dark h-100")
        ], width=3),
    ], className="mb-4")

    return html.Div([
        dbc.Card([
            dbc.CardBody([
                dcc.Graph(figure=fig_architecture, config={"displayModeBar": False})
            ])
        ], className="bg-dark mb-4"),

        html.H5("Agent Responsibilities", className="mb-3"),
        agent_cards
    ])


def create_mcp_servers_view():
    """Create MCP server integration view with Databricks components"""

    # MCP Server diagram
    fig = go.Figure()

    # Central Agent Hub
    fig.add_trace(go.Scatter(
        x=[0.5], y=[0.85],
        mode="markers+text",
        marker=dict(size=70, color="#ff6b6b"),
        text=["AI Agent<br>Hub"],
        textposition="middle center",
        textfont=dict(size=12, color="#fff"),
    ))

    # MCP Servers
    mcp_servers = [
        {"name": "Databricks\nGenie Spaces", "x": 0.1, "y": 0.5, "color": "#FF3621"},
        {"name": "Databricks\nVector MCP", "x": 0.3, "y": 0.5, "color": "#FF3621"},
        {"name": "AVEVA\nHistorian MCP", "x": 0.5, "y": 0.5, "color": "#00d4ff"},
        {"name": "Unity Catalog\nMCP", "x": 0.7, "y": 0.5, "color": "#FF3621"},
        {"name": "External\nAPI MCP", "x": 0.9, "y": 0.5, "color": "#00ff88"},
    ]

    for mcp in mcp_servers:
        # Connection line
        fig.add_trace(go.Scatter(
            x=[0.5, mcp["x"]], y=[0.78, 0.58],
            mode="lines",
            line=dict(color="#444", width=2, dash="dot"),
            hoverinfo="none"
        ))
        # MCP node
        fig.add_trace(go.Scatter(
            x=[mcp["x"]], y=[mcp["y"]],
            mode="markers+text",
            marker=dict(size=50, color=mcp["color"], symbol="square"),
            text=[mcp["name"]],
            textposition="bottom center",
            textfont=dict(size=10, color="#fff"),
        ))

    # External services
    external = [
        {"name": "OpenAI", "x": 0.2, "y": 0.15, "color": "#10a37f"},
        {"name": "Weather API", "x": 0.4, "y": 0.15, "color": "#ffd93d"},
        {"name": "NRC Database", "x": 0.6, "y": 0.15, "color": "#9b59b6"},
        {"name": "Parts Supplier", "x": 0.8, "y": 0.15, "color": "#e67e22"},
    ]

    for ext in external:
        fig.add_trace(go.Scatter(
            x=[ext["x"]], y=[ext["y"]],
            mode="markers+text",
            marker=dict(size=35, color=ext["color"], symbol="circle"),
            text=[ext["name"]],
            textposition="bottom center",
            textfont=dict(size=9, color="#fff"),
        ))
        # Connection to External API MCP
        fig.add_trace(go.Scatter(
            x=[0.9, ext["x"]], y=[0.42, 0.22],
            mode="lines",
            line=dict(color="#444", width=1),
            hoverinfo="none"
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#fff"},
        height=500,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.05]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.05]),
        title="MCP Server Integration Architecture"
    )

    # MCP Server detail cards
    mcp_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Img(src="https://www.databricks.com/wp-content/uploads/2022/06/db-nav-logo.svg",
                            height="20", className="me-2", style={"filter": "brightness(0) invert(1)"}),
                    "Genie Spaces MCP"
                ], style={"backgroundColor": "#FF3621"}),
                dbc.CardBody([
                    html.P("Natural language interface for data exploration", className="small"),
                    html.Hr(style={"borderColor": "#444"}),
                    html.Strong("Capabilities:", className="small text-info"),
                    html.Ul([
                        html.Li("Text-to-SQL generation", className="small"),
                        html.Li("Data discovery", className="small"),
                        html.Li("Semantic search", className="small"),
                        html.Li("Auto-visualization", className="small"),
                    ], className="mb-0"),
                    html.Div([
                        dbc.Badge("Active", color="success", className="me-1"),
                        dbc.Badge("15 queries/min", color="info"),
                    ], className="mt-2")
                ])
            ], className="bg-dark mcp-server-card h-100")
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-vector-square me-2"),
                    "Vector MCP Server"
                ], style={"backgroundColor": "#FF3621"}),
                dbc.CardBody([
                    html.P("Semantic search and RAG capabilities", className="small"),
                    html.Hr(style={"borderColor": "#444"}),
                    html.Strong("Features:", className="small text-info"),
                    html.Ul([
                        html.Li("Document embeddings", className="small"),
                        html.Li("Similarity search", className="small"),
                        html.Li("Knowledge retrieval", className="small"),
                        html.Li("Context augmentation", className="small"),
                    ], className="mb-0"),
                    html.Div([
                        dbc.Badge("Active", color="success", className="me-1"),
                        dbc.Badge("1.2M vectors", color="info"),
                    ], className="mt-2")
                ])
            ], className="bg-dark mcp-server-card h-100")
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-globe me-2"),
                    "External API Gateway"
                ], style={"backgroundColor": "#00ff88", "color": "#000"}),
                dbc.CardBody([
                    html.P("Secure access to external services", className="small"),
                    html.Hr(style={"borderColor": "#444"}),
                    html.Strong("Connected Services:", className="small text-info"),
                    html.Ul([
                        html.Li("OpenAI GPT-4", className="small"),
                        html.Li("Weather Service", className="small"),
                        html.Li("NRC Compliance DB", className="small"),
                        html.Li("Parts Suppliers API", className="small"),
                    ], className="mb-0"),
                    html.Div([
                        dbc.Badge("Active", color="success", className="me-1"),
                        dbc.Badge("4 endpoints", color="info"),
                    ], className="mt-2")
                ])
            ], className="bg-dark mcp-server-card h-100")
        ], width=4),
    ], className="mb-4")

    # Additional MCP details
    mcp_row2 = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-database me-2", style={"color": "#00d4ff"}),
                    "AVEVA Historian MCP"
                ]),
                dbc.CardBody([
                    html.Div([
                        html.Span("● ", style={"color": "#00ff88"}),
                        html.Span("Status: ", className="text-muted"),
                        html.Span("Connected", style={"color": "#00ff88"})
                    ], className="small mb-1"),
                    html.Div([
                        html.Span("● ", style={"color": "#00ff88"}),
                        html.Span("Data Rate: ", className="text-muted"),
                        html.Span("12,847 pts/sec", className="text-info")
                    ], className="small mb-1"),
                    html.Div([
                        html.Span("● ", style={"color": "#00ff88"}),
                        html.Span("Tags: ", className="text-muted"),
                        html.Span("2,456 monitored", className="text-info")
                    ], className="small"),
                ])
            ], className="bg-dark")
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-layer-group me-2", style={"color": "#FF3621"}),
                    "Unity Catalog MCP"
                ]),
                dbc.CardBody([
                    html.Div([
                        html.Span("● ", style={"color": "#00ff88"}),
                        html.Span("Status: ", className="text-muted"),
                        html.Span("Connected", style={"color": "#00ff88"})
                    ], className="small mb-1"),
                    html.Div([
                        html.Span("● ", style={"color": "#00ff88"}),
                        html.Span("Tables: ", className="text-muted"),
                        html.Span("847 registered", className="text-info")
                    ], className="small mb-1"),
                    html.Div([
                        html.Span("● ", style={"color": "#00ff88"}),
                        html.Span("Policies: ", className="text-muted"),
                        html.Span("156 active", className="text-info")
                    ], className="small"),
                ])
            ], className="bg-dark")
        ], width=6),
    ])

    return html.Div([
        dbc.Card([
            dbc.CardBody([
                dcc.Graph(figure=fig, config={"displayModeBar": False})
            ])
        ], className="bg-dark mb-4"),

        html.H5("MCP Server Details", className="mb-3"),
        mcp_cards,
        mcp_row2
    ])


def create_agent_communication_view():
    """Create animated agent communication visualization"""

    # This will be updated by callback for animation
    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-comments me-2"),
                "Live Agent Communication"
            ]),
            dbc.CardBody([
                html.Div(id="agent-comm-diagram", children=[
                    create_comm_diagram_static()
                ])
            ])
        ], className="bg-dark mb-4"),

        # Message log
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-stream me-2"),
                "Agent Message Stream"
            ]),
            dbc.CardBody([
                html.Div(id="agent-message-log", style={"maxHeight": "400px", "overflowY": "auto"}, children=[
                    create_message_log()
                ])
            ])
        ], className="bg-dark")
    ])


def create_comm_diagram_static():
    """Create static communication diagram with CSS animations"""

    # Agent communication flow
    agents = [
        {"id": "supervisor", "name": "Supervisor", "x": 50, "y": 10, "color": "#ff6b6b"},
        {"id": "predictive", "name": "Predictive", "x": 15, "y": 50, "color": "#00d4ff"},
        {"id": "scheduling", "name": "Scheduling", "x": 40, "y": 50, "color": "#00d4ff"},
        {"id": "resource", "name": "Resource", "x": 60, "y": 50, "color": "#00d4ff"},
        {"id": "safety", "name": "Safety", "x": 85, "y": 50, "color": "#00d4ff"},
    ]

    # MCP servers at bottom
    mcps = [
        {"id": "genie", "name": "Genie Spaces", "x": 20, "y": 85, "color": "#FF3621"},
        {"id": "vector", "name": "Vector DB", "x": 40, "y": 85, "color": "#FF3621"},
        {"id": "historian", "name": "Historian", "x": 60, "y": 85, "color": "#00d4ff"},
        {"id": "external", "name": "External APIs", "x": 80, "y": 85, "color": "#00ff88"},
    ]

    agent_elements = []

    # Create SVG for animated lines
    svg_lines = html.Div([
        html.Div(
            style={
                "position": "absolute",
                "top": "0",
                "left": "0",
                "width": "100%",
                "height": "100%",
                "pointerEvents": "none"
            },
            children=[
                # Animated connection lines using CSS
                html.Div(style={
                    "position": "absolute",
                    "top": "25%", "left": "50%",
                    "width": "35%", "height": "2px",
                    "background": "linear-gradient(90deg, transparent, #00d4ff, transparent)",
                    "transform": "rotate(30deg)",
                    "transformOrigin": "left center",
                    "animation": "pulse 2s infinite"
                }),
            ]
        )
    ])

    # Create agent nodes
    for agent in agents:
        agent_elements.append(
            html.Div([
                html.Div([
                    html.I(className="fas fa-robot fa-2x", style={"color": agent["color"]}),
                ], className="text-center mb-2 agent-node agent-active"),
                html.Div(agent["name"], className="small text-center")
            ], style={
                "position": "absolute",
                "left": f"{agent['x']}%",
                "top": f"{agent['y']}%",
                "transform": "translate(-50%, -50%)",
                "zIndex": "10"
            })
        )

    # Create MCP server nodes
    for mcp in mcps:
        agent_elements.append(
            html.Div([
                html.Div([
                    html.I(className="fas fa-server fa-lg", style={"color": mcp["color"]}),
                ], className="text-center mb-1"),
                html.Div(mcp["name"], className="small text-center", style={"fontSize": "10px"})
            ], style={
                "position": "absolute",
                "left": f"{mcp['x']}%",
                "top": f"{mcp['y']}%",
                "transform": "translate(-50%, -50%)",
                "zIndex": "10"
            })
        )

    # Add animated message bubbles
    message_bubbles = [
        {"from": "predictive", "to": "supervisor", "msg": "High failure probability detected", "delay": "0s"},
        {"from": "supervisor", "to": "scheduling", "msg": "Create maintenance task", "delay": "1s"},
        {"from": "scheduling", "to": "resource", "msg": "Find available worker", "delay": "2s"},
        {"from": "resource", "to": "safety", "msg": "Verify certifications", "delay": "3s"},
    ]

    for i, bubble in enumerate(message_bubbles):
        agent_elements.append(
            html.Div([
                html.Div(bubble["msg"], className="small px-2 py-1 rounded", style={
                    "backgroundColor": "#00d4ff",
                    "color": "#000",
                    "fontSize": "10px",
                    "whiteSpace": "nowrap"
                })
            ], className="message-bubble", style={
                "position": "absolute",
                "left": f"{30 + i * 15}%",
                "top": "35%",
                "animationDelay": bubble["delay"],
                "zIndex": "20"
            })
        )

    return html.Div([
        html.Div(agent_elements, style={
            "position": "relative",
            "height": "350px",
            "backgroundColor": "rgba(0,0,0,0.3)",
            "borderRadius": "10px"
        })
    ])


def create_message_log():
    """Create simulated agent message log"""

    messages = [
        {"time": "14:32:15.123", "from": "Supervisor", "to": "All Agents", "type": "broadcast",
         "msg": "Initiating schedule optimization cycle", "color": "#ff6b6b"},
        {"time": "14:32:15.456", "from": "Predictive", "to": "Genie Spaces MCP", "type": "query",
         "msg": "SELECT equipment_id, failure_prob FROM sensor_analysis WHERE failure_prob > 0.7", "color": "#00d4ff"},
        {"time": "14:32:15.892", "from": "Genie Spaces MCP", "to": "Predictive", "type": "response",
         "msg": "Returned 3 equipment records with high failure probability", "color": "#FF3621"},
        {"time": "14:32:16.234", "from": "Predictive", "to": "Supervisor", "type": "alert",
         "msg": "CRITICAL: Reactor Coolant Pump A - 85% failure probability", "color": "#00d4ff"},
        {"time": "14:32:16.567", "from": "Supervisor", "to": "Vector MCP", "type": "query",
         "msg": "Retrieve similar historical maintenance procedures", "color": "#ff6b6b"},
        {"time": "14:32:17.123", "from": "Vector MCP", "to": "Supervisor", "type": "response",
         "msg": "Found 5 similar procedures with 92% relevance score", "color": "#FF3621"},
        {"time": "14:32:17.456", "from": "Supervisor", "to": "Scheduling", "type": "task",
         "msg": "Create urgent maintenance task for RCP-A bearing replacement", "color": "#ff6b6b"},
        {"time": "14:32:17.789", "from": "Scheduling", "to": "Resource", "type": "request",
         "msg": "Find available senior technician with Pumps certification", "color": "#00d4ff"},
        {"time": "14:32:18.123", "from": "Resource", "to": "Safety", "type": "check",
         "msg": "Verify radiation exposure limits for W003 (Michael Rodriguez)", "color": "#00d4ff"},
        {"time": "14:32:18.456", "from": "Safety", "to": "External API", "type": "query",
         "msg": "Check NRC compliance requirements for RCP maintenance", "color": "#00d4ff"},
        {"time": "14:32:18.892", "from": "External API", "to": "Safety", "type": "response",
         "msg": "Work permit required, 2-person rule applies, max 4hr exposure", "color": "#00ff88"},
        {"time": "14:32:19.234", "from": "Safety", "to": "Supervisor", "type": "approval",
         "msg": "Schedule APPROVED with safety constraints applied", "color": "#00d4ff"},
        {"time": "14:32:19.567", "from": "Supervisor", "to": "All Agents", "type": "complete",
         "msg": "Schedule optimization complete - 8 tasks scheduled for next 7 days", "color": "#ff6b6b"},
    ]

    log_items = []
    for msg in messages:
        type_colors = {
            "broadcast": "secondary",
            "query": "info",
            "response": "success",
            "alert": "danger",
            "task": "warning",
            "request": "info",
            "check": "primary",
            "approval": "success",
            "complete": "success"
        }

        log_items.append(
            html.Div([
                html.Div([
                    html.Span(msg["time"], className="text-muted", style={"fontFamily": "monospace", "fontSize": "11px"}),
                    dbc.Badge(msg["type"], color=type_colors.get(msg["type"], "secondary"), className="mx-2"),
                    html.Span(msg["from"], style={"color": msg["color"], "fontWeight": "bold"}),
                    html.I(className="fas fa-arrow-right mx-2", style={"color": "#666", "fontSize": "10px"}),
                    html.Span(msg["to"], style={"color": "#888"}),
                ], className="mb-1"),
                html.Div(msg["msg"], className="small text-muted ps-4", style={"fontSize": "12px"})
            ], className="mb-3 pb-2 border-bottom border-secondary")
        )

    return html.Div(log_items)


# =============================================================================
# VALUE & ROI PAGE
# =============================================================================

def create_value_page():
    """Create value proposition and ROI page"""

    # Before/After comparison metrics
    fig_comparison = go.Figure()
    categories = ["Unplanned Downtime", "Maintenance Costs", "Schedule Efficiency", "Worker Utilization"]

    fig_comparison.add_trace(go.Bar(
        name="Before AI Scheduling",
        x=categories,
        y=[120, 2.8, 65, 58],
        marker_color="#ff6b6b",
        text=["120 hrs/yr", "$2.8M", "65%", "58%"],
        textposition="outside"
    ))
    fig_comparison.add_trace(go.Bar(
        name="After AI Scheduling",
        x=categories,
        y=[25, 1.9, 94, 87],
        marker_color="#00ff88",
        text=["25 hrs/yr", "$1.9M", "94%", "87%"],
        textposition="outside"
    ))

    fig_comparison.update_layout(
        title="Performance Improvement Comparison",
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#fff"},
        height=400,
        yaxis={"gridcolor": "#333", "title": "Value"},
        legend={"orientation": "h", "y": -0.15}
    )

    # ROI Timeline
    months = ["Month 1", "Month 3", "Month 6", "Month 12", "Month 18", "Month 24"]
    roi_values = [-500, -200, 150, 800, 1500, 2400]

    fig_roi = go.Figure()
    fig_roi.add_trace(go.Scatter(
        x=months, y=roi_values,
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(0, 212, 255, 0.2)",
        line=dict(color="#00d4ff", width=3),
        marker=dict(size=10)
    ))
    fig_roi.add_hline(y=0, line_dash="dash", line_color="#888")
    fig_roi.add_annotation(x="Month 6", y=150, text="Break-even Point", showarrow=True,
                          arrowhead=2, arrowcolor="#ffd93d", font=dict(color="#ffd93d"))

    fig_roi.update_layout(
        title="Projected ROI Timeline ($K)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#fff"},
        height=350,
        yaxis={"gridcolor": "#333", "title": "Cumulative ROI ($K)"},
        xaxis={"gridcolor": "#333"}
    )

    # Value cards
    value_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-clock fa-3x", style={"color": "#00ff88"}),
                    ], className="text-center mb-3"),
                    html.H3("79%", className="text-center", style={"color": "#00ff88"}),
                    html.P("Reduction in Unplanned Downtime", className="text-center text-muted"),
                    html.Hr(style={"borderColor": "#444"}),
                    html.P("From 120 hours to 25 hours annually", className="text-center small")
                ])
            ], className="bg-dark h-100")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-dollar-sign fa-3x", style={"color": "#00ff88"}),
                    ], className="text-center mb-3"),
                    html.H3("$900K", className="text-center", style={"color": "#00ff88"}),
                    html.P("Annual Cost Savings", className="text-center text-muted"),
                    html.Hr(style={"borderColor": "#444"}),
                    html.P("32% reduction in maintenance costs", className="text-center small")
                ])
            ], className="bg-dark h-100")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-chart-line fa-3x", style={"color": "#00ff88"}),
                    ], className="text-center mb-3"),
                    html.H3("45%", className="text-center", style={"color": "#00ff88"}),
                    html.P("Improvement in Schedule Efficiency", className="text-center text-muted"),
                    html.Hr(style={"borderColor": "#444"}),
                    html.P("From 65% to 94% optimal scheduling", className="text-center small")
                ])
            ], className="bg-dark h-100")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-users fa-3x", style={"color": "#00ff88"}),
                    ], className="text-center mb-3"),
                    html.H3("50%", className="text-center", style={"color": "#00ff88"}),
                    html.P("Better Worker Utilization", className="text-center text-muted"),
                    html.Hr(style={"borderColor": "#444"}),
                    html.P("From 58% to 87% productive time", className="text-center small")
                ])
            ], className="bg-dark h-100")
        ], width=3),
    ], className="mb-4")

    # Key benefits list
    benefits = dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-star me-2", style={"color": "#ffd93d"}),
            "Key Business Benefits"
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Operational Excellence", className="text-info mb-3"),
                    html.Ul([
                        html.Li("Proactive maintenance reduces emergency repairs"),
                        html.Li("Optimized scheduling minimizes equipment downtime"),
                        html.Li("Real-time visibility into plant maintenance status"),
                        html.Li("Data-driven decision making"),
                    ])
                ], width=4),
                dbc.Col([
                    html.H6("Safety & Compliance", className="text-info mb-3"),
                    html.Ul([
                        html.Li("Automated NRC regulation compliance checks"),
                        html.Li("Proper certification matching for all tasks"),
                        html.Li("Radiation exposure optimization"),
                        html.Li("Complete audit trail for all decisions"),
                    ])
                ], width=4),
                dbc.Col([
                    html.H6("Workforce Optimization", className="text-info mb-3"),
                    html.Ul([
                        html.Li("Skill-based task assignment"),
                        html.Li("Fair workload distribution"),
                        html.Li("Reduced overtime costs"),
                        html.Li("Better training needs identification"),
                    ])
                ], width=4),
            ])
        ])
    ], className="bg-dark mb-4")

    # Risk mitigation
    risk_card = dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-shield-alt me-2", style={"color": "#ff6b6b"}),
            "Risk Mitigation"
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-times-circle fa-2x me-3", style={"color": "#ff6b6b"}),
                        html.Div([
                            html.H5("Without AI Scheduling", className="mb-1"),
                            html.Ul([
                                html.Li("Reactive maintenance leads to failures", className="text-muted"),
                                html.Li("Manual scheduling prone to errors", className="text-muted"),
                                html.Li("Skill mismatches cause delays", className="text-muted"),
                                html.Li("Compliance gaps risk penalties", className="text-muted"),
                            ])
                        ])
                    ], className="d-flex")
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-check-circle fa-2x me-3", style={"color": "#00ff88"}),
                        html.Div([
                            html.H5("With AI Scheduling", className="mb-1"),
                            html.Ul([
                                html.Li("Predictive alerts prevent failures", className="text-muted"),
                                html.Li("Optimized schedules maximize uptime", className="text-muted"),
                                html.Li("Perfect skill matching every time", className="text-muted"),
                                html.Li("100% compliance assurance", className="text-muted"),
                            ])
                        ])
                    ], className="d-flex")
                ], width=6),
            ])
        ])
    ], className="bg-dark")

    return html.Div([
        html.H2("Value Proposition & ROI", className="mb-4"),
        html.P("Demonstrating the business impact of AI-driven maintenance scheduling", className="text-muted mb-4"),

        value_cards,

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=fig_comparison, config={"displayModeBar": False})
                    ])
                ], className="bg-dark")
            ], width=7),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=fig_roi, config={"displayModeBar": False})
                    ])
                ], className="bg-dark")
            ], width=5),
        ], className="mb-4"),

        benefits,
        risk_card
    ])


# =============================================================================
# CALLBACKS
# =============================================================================

@callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    """Route to appropriate page based on URL"""
    if pathname == "/" or pathname == "":
        return create_dashboard()
    elif pathname == "/predictive":
        return create_predictive_page()
    elif pathname == "/workforce":
        return create_workforce_page()
    elif pathname == "/schedule":
        return create_schedule_page()
    elif pathname == "/data-model":
        return create_data_model_page()
    elif pathname == "/agents":
        return create_agents_page()
    elif pathname == "/value":
        return create_value_page()
    else:
        return create_dashboard()


@callback(
    Output("data-model-content", "children"),
    [Input("btn-data-flow", "n_clicks"),
     Input("btn-star-schema", "n_clicks"),
     Input("btn-source-systems", "n_clicks")],
    prevent_initial_call=True
)
def update_data_model_view(data_flow, star_schema, source_systems):
    """Update data model view based on button clicks"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return create_data_flow_view()

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "btn-data-flow":
        return create_data_flow_view()
    elif button_id == "btn-star-schema":
        return create_star_schema_view()
    elif button_id == "btn-source-systems":
        return create_data_flow_view()  # Same as data flow for now
    else:
        return create_data_flow_view()


@callback(
    Output("agent-view-content", "children"),
    [Input("btn-agent-overview", "n_clicks"),
     Input("btn-mcp-servers", "n_clicks"),
     Input("btn-agent-comm", "n_clicks")],
    prevent_initial_call=True
)
def update_agent_view(overview, mcp, comm):
    """Update agent view based on button clicks"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return create_agent_overview()

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "btn-agent-overview":
        return create_agent_overview()
    elif button_id == "btn-mcp-servers":
        return create_mcp_servers_view()
    elif button_id == "btn-agent-comm":
        return create_agent_communication_view()
    else:
        return create_agent_overview()


# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
