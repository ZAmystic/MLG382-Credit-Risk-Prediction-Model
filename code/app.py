import dash
from dash import dcc, html, Input, Output, ctx
import pandas as pd
import plotly.express as px
import joblib

# ---------------- LOAD DATA ----------------
df = pd.read_csv("cleaned_german_credit_data.csv")

# ---------------- LOAD MODEL ----------------
model = joblib.load("credit_model.pkl")

# ---------------- APP INIT ----------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# ---------------- OPTIONS ----------------
sex_options = [{"label": x.title(), "value": x} for x in sorted(df["Sex"].dropna().unique())]
housing_options = [{"label": x.title(), "value": x} for x in sorted(df["Housing"].dropna().unique())]
saving_options = [{"label": x.title(), "value": x} for x in sorted(df["Saving accounts"].dropna().unique())]
checking_options = [{"label": x.title(), "value": x} for x in sorted(df["Checking account"].dropna().unique())]
purpose_options = [{"label": x.title(), "value": x} for x in sorted(df["Purpose"].dropna().unique())]

# ---------------- LAYOUT ----------------
app.layout = html.Div([

    html.Div([
        html.H2(" Credit Risk Dashboard", className="sidebar-title"),
        html.Hr(),
        html.P("Navigation", className="sidebar-text"),

        dcc.Tabs(
            id="tabs",
            value="tab-1",
            children=[
                dcc.Tab(label="Prediction", value="tab-1"),
                dcc.Tab(label="EDA", value="tab-2"),
            ]
        )
    ], className="sidebar"),

    html.Div(id="tab-content", className="main-content")
])

# ---------------- TAB CONTENT ----------------
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value")
)
def render_content(tab):

    if tab == "tab-1":
        return html.Div([
            html.H2("Credit Risk Prediction"),
            html.P("Enter applicant details below to estimate credit risk."),

            html.Div([
                html.Div([
                    html.Label("Age"),
                    dcc.Input(id="age", type="number", value=30, className="input-box"),

                    html.Label("Sex"),
                    dcc.Dropdown(id="sex", options=sex_options, value="male", className="dropdown-box"),

                    html.Label("Job"),
                    dcc.Input(id="job", type="number", value=2, min=0, max=3, className="input-box"),

                    html.Label("Housing"),
                    dcc.Dropdown(id="housing", options=housing_options, value="own", className="dropdown-box"),

                    html.Label("Saving Accounts"),
                    dcc.Dropdown(id="saving_accounts", options=saving_options, value="little", className="dropdown-box"),
                ], className="form-column"),

                html.Div([
                    html.Label("Checking Account"),
                    dcc.Dropdown(id="checking_account", options=checking_options, value="moderate", className="dropdown-box"),

                    html.Label("Credit Amount"),
                    dcc.Input(id="credit_amount", type="number", value=2000, className="input-box"),

                    html.Label("Duration (months)"),
                    dcc.Input(id="duration", type="number", value=12, className="input-box"),

                    html.Label("Purpose"),
                    dcc.Dropdown(id="purpose", options=purpose_options, value="radio/TV", className="dropdown-box"),

                    html.Br(),
                    html.Button("Predict", id="predict-btn", className="predict-button"),
                ], className="form-column"),
            ], className="form-grid"),

            html.Div(id="prediction-output", className="result-card")
        ])

    return html.Div([
        html.H2("Exploratory Data Analysis"),
        html.P("Interactive analysis of the cleaned credit dataset."),

        html.Div([
            dcc.Graph(id="age-distribution"),
            dcc.Graph(id="credit-box"),
        ], className="graph-grid"),

        html.Div([
            dcc.Graph(id="risk-pie"),
            dcc.Graph(id="housing-risk-bar"),
        ], className="graph-grid")
    ])

# ---------------- PREDICTION CALLBACK ----------------
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    Input("age", "value"),
    Input("sex", "value"),
    Input("job", "value"),
    Input("housing", "value"),
    Input("saving_accounts", "value"),
    Input("checking_account", "value"),
    Input("credit_amount", "value"),
    Input("duration", "value"),
    Input("purpose", "value"),
    prevent_initial_call=True
)
def predict_risk(n_clicks, age, sex, job, housing, saving_accounts,
                 checking_account, credit_amount, duration, purpose):

    if not ctx.triggered:
        return ""

    input_df = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "Saving accounts": saving_accounts,
        "Checking account": checking_account,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose": purpose
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    low_risk_prob = probability[1] * 100
    high_risk_prob = probability[0] * 100

    if prediction == 1:
        label = "Low Risk ✅"
        message = "This applicant is more likely to be creditworthy."
    else:
        label = "High Risk ⚠️"
        message = "This applicant may have a higher chance of default."

    return html.Div([
        html.H3(f"Prediction: {label}"),
        html.P(message),
        html.P(f"Low Risk Probability: {low_risk_prob:.2f}%"),
        html.P(f"High Risk Probability: {high_risk_prob:.2f}%")
    ])

# ---------------- EDA CALLBACKS ----------------
@app.callback(
    Output("age-distribution", "figure"),
    Input("age-distribution", "id")
)
def age_graph(_):
    return px.histogram(
        df,
        x="Age",
        color="Risk",
        barmode="overlay",
        title="Age Distribution by Risk"
    )

@app.callback(
    Output("credit-box", "figure"),
    Input("credit-box", "id")
)
def credit_graph(_):
    return px.box(
        df,
        x="Risk",
        y="Credit amount",
        color="Risk",
        title="Credit Amount by Risk"
    )

@app.callback(
    Output("risk-pie", "figure"),
    Input("risk-pie", "id")
)
def risk_graph(_):
    return px.pie(
        df,
        names="Risk",
        title="Risk Distribution"
    )

@app.callback(
    Output("housing-risk-bar", "figure"),
    Input("housing-risk-bar", "id")
)
def housing_risk_graph(_):
    grouped = df.groupby(["Housing", "Risk"]).size().reset_index(name="Count")
    return px.bar(
        grouped,
        x="Housing",
        y="Count",
        color="Risk",
        barmode="group",
        title="Housing Type by Risk"
    )

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)