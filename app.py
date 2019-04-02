# Dash and plotly
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

# Other
import os
import sys
import pickle
import json
import pandas as pd
import numpy as np

# Math
from scipy import stats

# Add shap to pythonpath
sys.path.append(os.getcwd() + '/fpdash')
from shapley import shap

"""
INITIALIZE STUFF
"""
# Import classifier
with open(os.getcwd() + '/data/clf.pickle', 'rb') as f:
    clf = pickle.load(f)
# Load data
X_test = pd.read_csv(os.getcwd() + '/data/X_test.csv')
X_train = pd.read_csv(os.getcwd() + '/data/X_train.csv')
X_test_decoded = pd.read_csv(os.getcwd() + '/data/X_test_decoded.csv')
# Initialize SHAP explainer
explainer = shap.Explainer(X=X_train, f=clf, mc='training')

"""
COMPUTE DATA
"""

# PROTOTYPE DATAFRAME
d = {'Feature' : ['personal_status', 'credit_amount', 'checking_status'],
     'Value' : ['male div/sep', '15000', '<0']}
df = pd.DataFrame(d)

# SHAP VALUES
def compute_shap(instance, nr_samples, top):
    # Compute SHAP values
    shap_values = explainer.standard(X_test.iloc[instance], m = nr_samples, return_samples=True)
    # Retrieve most important features
    df = pd.DataFrame.from_dict(shap_values.shap_values, orient = 'index').reset_index(level=0)
    df = df.reindex(df[0].abs().sort_values(ascending = False).index)
    features = list(df['index'].iloc[0:top])
    importances = list(df[0].iloc[0:top])
    # Retrieve feature value
    values = [X_test_decoded.iloc[instance][f] for f in features]
    # Retrieve errors
    errors = []
    alpha = 0.05
    for f in features: 
        index = list(X_train).index(f)
        samples = shap_values.samples[index]
        n = len(samples)
        t = stats.t.ppf(1-alpha/2, n)
        s = samples['c'].std()
        errors.append(t*s/np.sqrt(n))
    return importances, features, values, errors

"""
PLOT/DISPLAY FUNCTIONS
"""

def generate_options():
    return [{'label' : 'Instance %s' % nr, 'value' : nr} for nr in range(1,11)]

def prototype_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +
        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))],
        style={'font-size': '1.3rem',
               'margin' : '0 auto'}
    )

def feature_importance_bar(shap_value, error, lim):
    if shap_value >= 0:
        color = '#0DB5E6'
    else:
        color = '#ffa54c'
    # Trace definition
    hoverlabel = {
        'bordercolor' : 'white',
        'font' : {'size' : 10},
    }
    trace = go.Bar(x = [shap_value] , 
               y = [''],
               orientation = 'h',
               hoverinfo = 'x',
               hoverlabel = hoverlabel,
               marker = {'color' : color},
               error_x= {'type' : 'data', 
                         'array' : [error],
                         'visible' : True,
                         'thickness': 1,
                         'width' : 2, 
                         'color' : '#727272'}
    )
    # Layout definition
    xaxis = {
        'range' : [-lim, lim],
        'fixedrange' : True,
        'showgrid' : False,
        'zeroline' : False,
        'showline' : False,
        'showticklabels' : False,
        'hoverformat': '.2f'
    }
    yaxis = {
        'fixedrange' : True,
        'showgrid' : False,
        'zeroline' : False,
        'showline' : False,
        'showticklabels' : False
    }
    margin=go.layout.Margin(l=0, r=0, t=0, b=0, pad=0)
    layout = go.Layout(yaxis = yaxis,
                       xaxis = xaxis,
                       margin = margin,
                       bargap = 0)
    
    # Config definition
    config={'displayModeBar': False,
            'showLink' : False}
    
    return dcc.Graph(figure = {'data' : [trace],
                               'layout' : layout}, 
                     config = config,
                     style = {'height' : '18px',
                              'width' : '170px'})

def feature_importance_table(importances, features, values, errors):
    # Add header
    table = [html.Tr([html.Th(col) for col in ['Importance', 'Feature', 'Value']])]
    # Add body
    lim = np.abs(importances[0]) + 0.2
    for importance, feature, value, error in zip(importances, features, values, errors):
        table.append(html.Tr([
            html.Td(feature_importance_bar(importance, error, lim)),
            html.Td(feature),
            html.Td(value),
        ]))
        
    return html.Table(table,
                      style={'font-size': '1.5rem',
                             'marginTop' : '10px'}
                     )
"""
STYLING
"""

colors = {
    'background': '#f6f6f6',
    'text-gray' : '#727272'
}

# DIV STYLES
columnStyle = {'marginLeft': 5,
               'marginRight' : 5,
                'backgroundColor': colors['background'],
                'paddingLeft' : 20,
                'paddingRight' : 20,
                'paddingBottom' : 20,
                'height' : '93vh',
                'overflow': 'auto'}

middleColumnStyle = {'marginLeft': 20,
                'paddingLeft' : 20,
                'paddingRight' : 20,
                'paddingBottom' : 20}

radioStyle  = {
    'margin-right': 10
}

labelStyle = {
    'font-size' : '1.4rem',
    'color' : colors['text-gray']
}

iconStyle = {'font-size' : '1.5rem',
             'color' : colors['text-gray']}


"""
APPLICATION
"""

external_stylesheets = [os.getcwd() + '/assets/font-awesome/css/all.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
        html.Div([
            # LEFT COLUMN: ALERT SELECTION
            html.Div([html.H1('Instances'),
                      dcc.RadioItems(
                          id='alertlist',
                          inputStyle = radioStyle,
                          labelStyle = labelStyle,
                          value=1,
                          options=generate_options())
                     ], 
                     className="two columns",
                     style = columnStyle),
            
            # MIDDLE COLUMN: ALERT EXPLANATION
            html.Div(
                children = [
                    # Title
                    html.H1('Alert ID', id='alert_id'),
                    # Prediction Probability Row
                    html.Div(className = 'row', 
                             children = [
                                 # Title
                                 html.H2('Prediction'),
                                 # Subtitle
                                 html.P("""The prediction probability indicates the 
                                           likelihood of the positive class, as estimated by the classifier. """),
                                 html.H4(id = 'load-probability'),
                                 html.H4(id = 'prediction-probability')
                             ]),
                    
                    # Number of Samples Row
                    html.Div(className = 'row',
                             children = [
                                 # Title
                                 html.H2('Explanation'),
                                 # Paragraph
                                 html.P('Feature values and their approximated importances are displayed, sorted on absolute importance.'),
                                 # Slider
                                 html.H3(['Number of Samples ',
                                          html.Div([html.I(className="fas fa-info-circle", style=iconStyle),
                                                     html.Span( """Increasing the number of samples per feature results in
                                                                   more accurate approximations but increases the 
                                                                   computation time.""",
                                                               className='tooltiptext')], 
                                                    className = "tooltip")]),
                                 html.Div(className="nine columns",
                                          children = [dcc.Slider(
                                              id='samples-slider',
                                              min=0,
                                              max=1000,
                                              step=50,
                                              value=100,                            
                                              marks={str(n): {'label' : str(n), 
                                                              'style' : {'font-size' : 10}} for n in range(0,1100,100)})]),
                             ]),
                    html.Div(className = 'row', 
                             children = [html.Button('Compute', 
                                                     id='compute-button',
                                                     style = {'marginTop' : '2.5rem'})]),
                    html.Div(id='load-importances'),
                    html.Div(id='feature-importance-table')
                ],
                className="seven columns",
                id = 'explanation',
                style = middleColumnStyle),
            
            # RIGHT COLUMN: PROTOTYPE
            html.Div(className="three columns",
                     style = columnStyle,
                     children = [html.H1('Cluster Information'),
                                 html.H3('Prototype'),
                                 html.P('This instance is similar to a group of instances with the following properties.'),
                                 html.Div(prototype_table(df), 
                                          style={'padding-bottom' : 30,
                                                 'margin-left' : 0},
                                          className = 'column'),
                                 html.H3('Local Performance'),
                                 html.H5("""Local accuracy is the proportion of similar instances classified 
                                 correctly by the algorithm."""),
                                 html.H4(['Local Accuracy: 0.73 \u00B1 0.1 ']),
                                 html.H5("""Local precision is the proportion of similar instances classified as 
                                 positive by the classifier that were truly positive."""),
                                 html.H4(['Local Precision: 0.73 \u00B1 0.1 '])
                                ]),
            # HIDDEN DIVS WITH INTERMEDIATE VALUES
            html.Div(id='selected-instance', style={'display': 'none'}, children=0),
                      ])
], style={'paddingTop' : 5})

"""
-------------------------
CALLBACKS
-------------------------
"""

"""
SELECT INSTANCE
"""
# Select instance
@app.callback(Output('selected-instance', 'children'), 
              [Input('alertlist', 'value')])
def select_instance(value):
    instance = value
    return instance # or, more generally, json.dumps(cleaned_df)

# Update title
@app.callback(Output('alert_id', 'children'),
              [Input('alertlist', 'value')])
def update_title(value):
    title = 'Instance ID: %s' % value
    return title

"""
UPDATE PROBABILITY
"""
@app.callback(Output('load-probability', 'children'),
              [Input('alertlist', 'value')])
def update_probability(instance):
    if instance:
        return html.Div([html.P('Loading...')], 
                            id='prediction-probability',
                            style = {'marginTop' : '10px'})
    
@app.callback(Output('prediction-probability', 'children'),
              [Input('alertlist', 'value')])

def update_probability(instance):
    if instance:
        probability = clf.predict_proba([X_test.iloc[instance]])[0][1]
        return ['Prediction Probability: %.2f ' % probability,
                html.Div([html.I(className="fas fa-exclamation-circle", style=iconStyle),
                          html.Span([html.B("""WARNING: """),
                                       """the classifier might have made an inaccurate estimation
                                       (e.g. based on too few samples)!"""],
                                     className='tooltiptext')],
                          className = "tooltip")]
    
"""
COMPUTE SHAP VALUE
"""

@app.callback(Output('load-importances', 'children'),
              [Input('compute-button', 'n_clicks'),
               Input('selected-instance', 'children')])
def update_importances(n_clicks, children):
    if (n_clicks or children):
        return html.Div([html.P('Loading...')], 
                        id='feature-importance-table',
                        style = {'marginTop' : '10px'})

@app.callback(
    Output('feature-importance-table', 'children'),
    [Input('compute-button', 'n_clicks'), 
     Input('selected-instance', 'children')],
    [State('samples-slider', 'value')])
def update_importances(n_clicks, instance, nr_samples):
    if (n_clicks or instance):
        top = 10
        importances, features, values, errors = compute_shap(int(instance), nr_samples, top)
        return feature_importance_table(importances, features, values, errors)
    
if __name__ == '__main__':
    app.run_server(debug=True, processes=4)