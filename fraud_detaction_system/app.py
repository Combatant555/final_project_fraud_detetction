import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import joblib
from src.visualization.plot_utils import plot_feature_importances
from sklearn.metrics import roc_curve, auc  
from sklearn.preprocessing import StandardScaler

# Load processed data
df = pd.read_csv('data/processed/processed_data.csv')

models = {
    'RandomForest': joblib.load('models/RandomForest_model.pkl'),
    'LogisticRegression': joblib.load('models/LogisticRegression_model.pkl'),
    'SVC': joblib.load('models/SVC_model.pkl')
}


# Define feature columns (all columns except 'Class')
feature_columns = df.columns[:-1].tolist()  # Assuming 'Class' is the last column

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Application layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Fraud Detection Dashboard", className="text-center mb-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='model-dropdown',
                options=[{'label': k, 'value': k} for k in models.keys()],
                value='RandomForest',
                clearable=False,
                className='mb-4'
            ),
        ], width=4),
        
        dbc.Col([
            dcc.Graph(id='scatter-plot', config={'displayModeBar': False}),
        ], width=8)
    ]),
    
    dbc.Row([
        dbc.Col(html.H2("Check a Transaction", className="text-center mb-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("Enter Transaction Details"),
            *[dbc.Input(id=f"input-{feature}", type="number", placeholder=feature, className="mb-2") 
              for feature in feature_columns],
            dbc.Button('Check Transaction', id='check-button', color='primary', className='mb-4', n_clicks=0),
            html.Div(id='prediction-output', className='text-center mt-4')
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col(html.H2("Model Performance", className="text-center mb-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='feature-importances'), width=6),
        dbc.Col(dcc.Graph(id='roc-curve'), width=6)
    ]),
    
], fluid=True)

# Callback to update the scatter plot based on selected model
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('model-dropdown', 'value')
)
def update_scatter_plot(model_name):
    fig = px.scatter(df, x='V1', y='V2', color='Class', title=f'Scatter Plot of V1 vs V2 by Class using {model_name}')
    fig.update_layout(transition_duration=500)
    return fig

# Callback to check a transaction and update feature importance and prediction result
@app.callback(
    Output('prediction-output', 'children'),
    Output('feature-importances', 'figure'),
    Output('roc-curve', 'figure'),
    Input('check-button', 'n_clicks'),
    State('model-dropdown', 'value'),
    [State(f"input-{feature}", "value") for feature in feature_columns]
)
def update_output(n_clicks, model_name, *feature_values):
    if n_clicks > 0:
        model = models[model_name]
        
        # Ensure feature values are properly formatted
        input_df = pd.DataFrame([feature_values], columns=feature_columns)
        
        # Standardize the input
        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        message = f"The transaction is predicted to be {'FRAUDULENT' if prediction else 'NOT fraudulent'} with a probability of {probability:.2f}"
        
        # Plot feature importances
        feature_importance_fig = plot_feature_importances(model, feature_columns)
        
        # Plot ROC curve
        y_test = pd.read_csv(r'C:\Users\sajaw\fraud_detaction_system\data\processed\y_test.csv').values.ravel()
        X_test = pd.read_csv(r'C:\Users\sajaw\fraud_detaction_system\data\processed\X_test.csv')
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        roc_curve_fig = go.Figure()
        roc_curve_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.2f})'))
        roc_curve_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(dash='dash')))
        roc_curve_fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        
        return message, feature_importance_fig, roc_curve_fig
    
    return "", {}, {}

if __name__ == "__main__":
    app.run(debug=True)

