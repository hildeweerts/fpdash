# General
import os, sys, pickle, json
import pandas as pd
import numpy as np
# Dash and plotly
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
# colors
import matplotlib 
from matplotlib import cm
# Math
from scipy import stats
# sklearn
from sklearn.manifold import MDS

# add to pythonpath
sys.path.append(os.getcwd() + '/fpdash')
import shapley.shap as shap

"""
INITIALIZE GLOBAL STUFF
"""

# Import classifier
with open(os.getcwd() + '/data/clf.pickle', 'rb') as f:
    clf = pickle.load(f)
# Import NN
with open(os.getcwd() + '/data/nn.pickle', 'rb') as f:
    nn = pickle.load(f)

# load case base data
X_base = pd.read_csv(os.getcwd() + '/data/X_base.csv')
X_base_decoded = pd.read_csv(os.getcwd() + '/data/X_base_decoded.csv')
meta_base = pd.read_csv(os.getcwd() + '/data/meta_base.csv')
SHAP_base = pd.read_csv(os.getcwd() + '/data/SHAP_base.csv')

# load alert data
X_alert = pd.read_csv(os.getcwd() + '/data/X_alert.csv')
X_alert_decoded = pd.read_csv(os.getcwd() + '/data/X_alert_decoded.csv')
meta_alert = pd.read_csv(os.getcwd() + '/data/meta_alert.csv')
SHAP_alert = pd.read_csv(os.getcwd() + '/data/SHAP_alert.csv')

# load separate train data
X_train = pd.read_csv(os.getcwd() + '/data/X_train.csv')

# Initialize SHAP explainer (must use TRAIN data!)
explainer = shap.Explainer(X=X_train, f=clf, mc='training')

# Spectral colormap
spectral_cmap = matplotlib.cm.get_cmap('Spectral')
spectral_rgb = []
norm = matplotlib.colors.Normalize(vmin=0, vmax=255)
for i in range(0, 255):
    k = matplotlib.colors.colorConverter.to_rgb(spectral_cmap(norm(i)))
    spectral_rgb.append(k)
spectral = []

n_entries = 255
for k in [x / n_entries for x in range(0, n_entries+1, 1)]:
    C = spectral_rgb[int(np.round(255*k))-1]
    spectral.append([k, 'rgb'+str((C[0], C[1], C[2]))])
    
# Border colors
opacity = 0.5
cat_colors = {'TP' : 'rgba(159, 211, 86, %s)' % opacity,
             'TN' : 'rgba(13, 181, 230, %s)' % opacity,
             'FP' : 'rgba(177, 15, 46, %s)' % opacity,
             'FN' : 'rgba(255, 165, 76, %s)' % opacity}

"""
COMPUTE SHAP WITH SAMPLES
"""

def retrieve_shap(instance, top):
    # Retrieve SHAP values    
    shap_values = SHAP_alert.iloc[instance].to_dict()
    # Retrieve most important features
    df = pd.DataFrame.from_dict(shap_values, orient = 'index').reset_index(level=0)
    df = df.reindex(df[0].abs().sort_values(ascending = False).index)
    features = list(df['index'].iloc[0:top])
    importances = list(df[0].iloc[0:top])
    # Retrieve feature value
    values = [X_alert_decoded.iloc[instance][f] for f in features]
    # Retrieve errors
    alpha = 0.05
    return importances, features, values


"""
COMPUTATIONS FOR NEIGHBOR PLOT
"""
def retrieve_neighbors(i, n_neighbors = 10):
    if n_neighbors == 0:
        distances, neighbors = [None], [None]
    else:
        distances, neighbors = nn.kneighbors(SHAP_alert.iloc[[i]], n_neighbors=n_neighbors)
    return distances[0], neighbors[0]

def compute_mds(i, neighbors, space):
    """Compute x and y for multi-dimensional scaling plot.
    
    Parameters
    ----------
    i : int
        index of instance in X_test
    neighbors : np array [n_neighbors]
        array with indices of neighbors in X_train
    space : str, one from ['shap', 'feature']
        distances computed based on shap value space or feature value space
    """
    if space == 'shap':
        alert = SHAP_alert
        base = SHAP_base
    elif space == 'feature':
        alert = X_alert
        base = X_base
    else:
        raise ValueError("space not in ['shap', 'feature']")
    mds_values = np.vstack((np.array(alert.iloc[i]), np.array(base.iloc[neighbors])))
    mds = MDS(random_state=1, dissimilarity ='euclidean', metric=True)
    mds.fit(mds_values.astype(np.float64))
    x, y = mds.embedding_.transpose()
    return x, y

"""
PLOT FUNCTIONS
"""

"""
Feature importances
"""
def generate_options():
    return [{'label' : 'Case %s' % nr, 'value' : nr} for nr in range(1,11)]

def feature_importance_bar_exact(shap_value, lim):
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

def feature_importance_table_exact(importances, features, values):
    # Add header
    table = [html.Tr([html.Th(col) for col in ['Contribution', 'Feature', 'Value']])]
    # Add body
    lim = np.abs(importances[0]) + 0.2
    for importance, feature, value in zip(importances, features, values):
        table.append(html.Tr([
            html.Td(feature_importance_bar_exact(importance, lim)),
            html.Td(feature),
            html.Td(value),
        ]))
        
    return html.Table(table,
                      style={'font-size': '1.5rem',
                             'marginTop' : '10px'}
                     )

"""
Neighbors plot
"""
def scatter_neighbors(x, y, neighbors, view, instance, border_width=4):
    """
    Parameters
    ----------
        x : array
            mds x with x[0] alert and x[1:] neighbors
        y : array
            mds y, with y[0] being alert and y[1:] neighbors
        neighbors : array
            array with indexes of neighbors
        view : str, one from ['perf', 'pred']
            which view to plot
        instance : int
            index of the current alert
        border_width : int
            border width 
        
    """
    global spectral
    global cat_colors
    global meta_base
    global meta_alert
    
    if view == 'perf':
        showscale = False
        colorscale = [[0,'rgba(75, 75, 75, 1)'], [1, 'rgba(75, 75, 75, 1)']]
        color_alert = 'rgba(255, 255, 0, 0.3)'
        showlegend = True
    elif view == 'pred':
        border_width = 0
        showscale = True
        colorscale = spectral
        color_alert = spectral[int(meta_alert.iloc[instance]['score']*len(spectral))-1][1]
        showlegend = False
    else:
        raise ValueError("view must be one of ['pred', 'perf']")
    
    """
    PREP
    """
    # Retrieve meta information
    meta_neighbors = pd.DataFrame({'x' : x[1:], 'y' : y[1:], 
                  'performance' : meta_base['performance'].iloc[neighbors], 
                  'score' : meta_base['score'].iloc[neighbors], 
                  'index' : neighbors})
    """
    ADD TRACES
    """
    traces = []
    # Add neighbors
    for perf in ['TP', 'TN', 'FP', 'FN']:
        group = meta_neighbors[meta_neighbors['performance'] == perf]
        scatter = go.Scatter(
            x = group['x'],
            y = group['y'],
            mode = 'markers',
            marker = {'line' : {'width' : border_width, 'color' : cat_colors[perf]},
                      'color' : group['score'],
                      'colorscale' : colorscale,
                      'cmin' : 0,
                      'cmax' : 1,
                      'size' : 10},
            showlegend = showlegend,
            name=perf,
            hoverinfo = 'text',
            hoveron = 'points',
            text = ['p=%.2f' % i for i in group['score']])
        traces.append(scatter)
    #Add alert
    traces.append(go.Scatter(
        x = [x[0]],
        y = [y[0]],
        mode = 'markers',
        marker = {'line' : {'width' : 3, 'color' : 'rgba(50, 50, 50, 1)'},
                  'size' : 14,
                  'color' : color_alert,
                  'cmin' : 0,
                  'cmax' : 1},
        name = 'Current alert',
        showlegend = True,
        hoverinfo = 'text',
        hoveron = 'points',
        text = 'Current Alert (p=%.2f)' % meta_alert['score'].iloc[instance]))
    # Add dummy colorbar
    traces.append(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            colorscale=spectral, 
            showscale=showscale,
            cmin=0,
            cmax=1,
            colorbar=dict(thickness=5, ticklen=8, outlinewidth=0, title="""Prediction""", tickfont = {'size' : 8}, titlefont={'size' : 10})),
        showlegend = False,
        hoverinfo='none'))

    """
    Define layout
    """
    xaxis = {'fixedrange' : False,
             'showgrid' : True,
            'zeroline' : False,
            'showline' : False,
            'showticklabels' : False,
        }
    yaxis = {'fixedrange' : False,
            'showgrid' : True,
            'zeroline' : False,
            'showline' : False,
            'showticklabels' : False
        }
    margin = go.layout.Margin(l=0, r=0, t=0, b=0, pad=0)
    layout = go.Layout(yaxis = yaxis, xaxis = xaxis, margin = margin, height = 400,
                       hovermode = 'closest', legend = dict(y=-0.05, orientation='h'),
                       paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',title='Hoi')
    
    """
    Define config
    """
    # Config definition
    config={'displayModeBar': False,
            'showLink' : False}
    
    return dcc.Graph(id='neighbors-scatter',
                     figure = {'data' : traces,
                               'layout' : layout},
                     config = config,
                     #style = {'height' : '18px',
                     #  'width' : '170px'}
                    )

def update_performance(fig, instance, view, border_width=4):
    global spectral, meta_alert
    current_width = fig['data'][0]['marker']['line']['width']
    if ((current_width == 0) and (view == 'perf')):
        #alert
        fig['data'][4]['marker']['color'] = 'rgba(255, 255, 0, 0.3)' 
        #scale
        fig['data'][4]['showlegend'] = True
        fig['data'][5]['marker']['showscale'] = False
        #neighbors
        for i in range(4):
            fig['data'][i]['marker']['line']['width'] = border_width
            fig['data'][i]['marker']['colorscale'] = [[0,'rgba(75, 75, 75, 1)'], [1, 'rgba(75, 75, 75, 1)']]
            fig['data'][i]['showlegend'] = True
    elif ((current_width != 0) and (view == 'pred')):
        #alert
        fig['data'][4]['marker']['color'] = spectral[int(meta_alert.iloc[instance]['score']*len(spectral))-1][1]
        #scale
        fig['data'][4]['showlegend'] = True
        fig['data'][5]['marker']['showscale'] = True
        #neighbors
        for i in range(4):
            fig['data'][i]['marker']['line']['width'] = 0
            fig['data'][i]['marker']['colorscale'] = spectral
            fig['data'][i]['showlegend'] = False
    return fig

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
app.config.supress_callback_exceptions = True

app.layout = html.Div([
        html.Div([
            # LEFT COLUMN: ALERT SELECTION
            html.Div([html.H1('Alerts'),
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
                    html.H1('Case ID', id='alert_id'),
                    # Prediction Probability Row
                    html.Div(className = 'row', 
                             children = [
                                 # Title
                                 html.H2('Prediction'),
                                 # Subtitle
                                 html.P("""The model's confidence indicates the 
                                           likelihood of the positive class, as estimated by the model. """),
                                 html.H4(id = 'load-probability'),
                                 html.H4(id = 'prediction-probability')
                             ]),
                    
                    # Number of Samples Row
                    html.Div(className = 'row',
                             children = [
                                 # Title
                                 html.H2('Explanation'),
                                 # Paragraph
                                 html.P("""The displayed feature contributions indicate how much each feature value contributed to the
                                           algorithm's prediction."""),
                                 html.Ul([
                                     html.Li(html.P([
                                         html.B('Negative contribution: '), 
                                         'feature value makes it ', html.B('less'), ' likely that the case is fraudulent.']
                                     )),
                                     html.Li(html.P([
                                         html.B('Positive contribution: '), 
                                         'feature value makes it ', html.B('more'), ' likely that the case is fraudulent.']
                                     ))
                                 ])
                             ]),

                    html.Div(id='load-importances'),
                    html.Div(id='feature-importance-table')
                ],
                className="five columns",
                id = 'explanation',
                style = middleColumnStyle),
            
            # RIGHT COLUMN: NEIGHBORS
            html.Div(className="five columns",
                     style = columnStyle,
                     children = [html.H1('Case-Based Performance'),
                                 html.H2('Most Similar Cases'),
                                 html.P('The cases most similar to the alert are retrieved.'),
                                 html.Div(className='row',
                                          children = [
                                              html.H6('Number of Cases'),
                                          ]),
                                 html.Div(className = 'row', 
                                          children = [
                                              html.Div(className="eleven columns",
                                                      children = [dcc.Slider(
                                                          id='neighbors-slider',
                                                          min=10,
                                                          max=100,
                                                          step=5,
                                                          value=20,                            
                                                          marks={str(n): {'label' : str(n), 
                                                                          'style' : {'font-size' : 10}} for n in range(10,110,10)})])
                                          ]),
                                 html.Div(className='row',
                                          children = [
                                              html.H6('View', style={'marginTop' : '2rem'}),
                                          ]),
                                 html.Div(className = 'row',
                                          id='perf-buttons',
                                          children = [html.Button('Performance', 
                                                                  id ='color-button-perf',
                                                                  style = {'marginRight' : '0.5rem',
                                                                           'background-color' : 'transparent', 
                                                                           'color' : '#555'}),
                                                      html.Button('Predictions', 
                                                                  id ='color-button-pred',
                                                                  style = {'marginRight' : '0.5rem'})]),
                                 html.Div(className='row',
                                          children = [
                                              html.H6('Similarity'),
                                          ]),
                                 html.Div(className = 'row',
                                          id='space-buttons',
                                          children = [html.Button('Feature Values', 
                                                                  id ='space-button-val',
                                                                  style = {'marginRight' : '0.5rem'}),
                                                      html.Button('Feature Contributions', 
                                                                  id ='space-button-contr',
                                                                  style = {'marginRight' : '0.5rem'})]),

                                 html.Div(className ='row',
                                          children = [
                                              html.Div(id='neighbors-plot', style= {'marginTop' : '1.5rem', 'marginBottom' : '1.5rem'})
                                          ])
                                ]),
            # HIDDEN DIVS WITH INTERMEDIATE VALUES
            html.Div(id='selected-instance', style={'display': 'none'}, children=0),
            html.Div(id='neighbor-dummydiv', style={'display': 'none'}, children=None),
            html.Div(id='color-state', children='perf:0 pred:0 last:pred', style={'display': 'none'}),
            html.Div(id='space-state', children='val:0 contr:0 last:val', style={'display': 'none'}),
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
    global X_alert
    title = 'Case ID: %s' % value
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
        probability = clf.predict_proba([X_alert.iloc[instance]])[0][1]
        return ["Model's confidence: %.2f " % probability,
                html.Div([html.I(className="fas fa-exclamation-circle", style=iconStyle),
                          html.Span([html.B("""WARNING: """),
                                       """the model's confidence might be inaccurate!"""],
                                     className='tooltiptext')],
                          className = "tooltip")]
    
"""
COMPUTE SHAP VALUE
"""

@app.callback(
    Output('feature-importance-table', 'children'),
    [Input('selected-instance', 'children')])
def update_importances(instance):
    if instance:
        top = 15
        nr_samples=10
        importances, features, values = retrieve_shap(int(instance), top)
        return feature_importance_table_exact(importances, features, values)
    
"""
RETRIEVE NEIGHBORS
"""

"""
save neighbors
"""
@app.callback(
    Output('neighbor-dummydiv', 'children'),
    [Input('neighbors-slider', 'value'),
     Input('selected-instance', 'children')])
def update_dummy_div(n_neighbors, instance):
    global meta_base
    distances, neighbors = retrieve_neighbors(instance, n_neighbors)
    return neighbors

"""
scatterplot
"""

@app.callback(
    Output('color-state', 'children'), 
    [Input('color-button-perf', 'n_clicks'),
     Input('color-button-pred', 'n_clicks')],
    [State('color-state', 'children')])
def update_state(perf_clicks, pred_clicks, prev_clicks):
    prev_clicks = dict([i.split(':') for i in prev_clicks.split(' ')])
    
    # Replace None by 0
    if perf_clicks is None:
        perf_clicks = 0
    if pred_clicks is None:
        pred_clicks = 0
    # Check value
    if perf_clicks > int(prev_clicks['perf']):
        last_clicked = 'perf'
    elif pred_clicks > int(prev_clicks['pred']):
        last_clicked = 'pred'
    else:
        last_clicked ='pred'
    # Check changed
    prev_clicked = prev_clicks['last']
    if prev_clicked == last_clicked:
        changed = 'no'
    else:
        changed = 'yes'
    # Update current state
    cur_clicks = 'perf:{} pred:{} last:{} changed:{}'.format(perf_clicks, pred_clicks, last_clicked, changed)
    return cur_clicks

@app.callback(
    Output('space-state', 'children'), 
    [Input('space-button-val', 'n_clicks'),
     Input('space-button-contr', 'n_clicks')],
    [State('space-state', 'children')])
def update_state(val_clicks, contr_clicks, prev_clicks):
    prev_clicks = dict([i.split(':') for i in prev_clicks.split(' ')])
    # Replace None by 0
    if val_clicks is None:
        val_clicks = 0
    if contr_clicks is None:
        contr_clicks = 0
    # Check value
    if val_clicks > int(prev_clicks['val']):
        last_clicked = 'feature'
    elif contr_clicks > int(prev_clicks['contr']):
        last_clicked = 'shap'
    else:
        last_clicked ='feature'
    # Check changed
    prev_clicked = prev_clicks['last']
    if prev_clicked == last_clicked:
        changed = 'no'
    else:
        changed = 'yes'
    cur_clicks = 'val:{} contr:{} last:{} changed:{}'.format(val_clicks, contr_clicks, last_clicked, changed)
    return cur_clicks

@app.callback(
    Output('neighbors-plot', 'children'),
    [Input('neighbor-dummydiv', 'children'),
     Input('selected-instance', 'children'),
     Input('space-state', 'children')],
    [State('color-state', 'children'),
     State('neighbors-plot', 'children')])
def update_neighbors(neighbors, instance, prev_clicks_space, prev_clicks_color, current_graph):
    global meta_base
    prev_clicks_space = dict([i.split(':') for i in prev_clicks_space.split(' ')])
    # Check whether it has changed
    if prev_clicks_space['changed'] == 'no':
        graph = current_graph
    else:
        # Determine space
        last_clicked_space = prev_clicks_space['last']
        # compute distances
        x, y = compute_mds(instance, neighbors, space=last_clicked_space)
        # Determine color
        last_clicked_color = dict([i.split(':') for i in prev_clicks_color.split(' ')])['last']
        # create graph
        graph = scatter_neighbors(x, y, neighbors, view=last_clicked_color, instance=instance)
    return graph

@app.callback(
    Output('neighbors-scatter', 'figure'),
    [Input('color-state', 'children')],
    [State('neighbors-scatter', 'figure'), 
     State('selected-instance', 'children')])
def update_scatter(prev_clicks_color, figure, instance):
    prev_clicks_color = dict([i.split(':') for i in prev_clicks_color.split(' ')])
    if prev_clicks_color['changed'] == 'no':
        figure = figure
    else:
        last_clicked_color = prev_clicks_color['last']
        print(last_clicked_color)
        figure = update_performance(figure, instance, view=last_clicked_color)
    return figure


"""
update buttons
"""

@app.callback(
    Output('color-button-perf', 'style'),
    [Input('color-state', 'children')],
    [State('color-button-perf', 'style')])
def update_perf_button(prev_clicks_color, current_style):
    prev_clicks_color = dict([i.split(':') for i in prev_clicks_color.split(' ')])
    new_background = current_style['background-color']
    new_color = current_style['color']
    if prev_clicks_color['changed'] == 'yes':
        if prev_clicks_color['last'] == 'perf':
            new_background = '#888'
            new_color = '#fff'
        elif prev_clicks_color['last'] == 'pred':
            new_background = 'transparent'
            new_color = '#555'
    current_style['background-color'] = new_background
    current_style['color'] = new_color
    return current_style


# @app.callback(
#     Output('perf-button', 'children'),
#     [Input('perf-button', 'n_clicks')])
# def update_perf_button(n_clicks):
#     if (n_clicks is None) or (n_clicks % 2 == 0):
#         children = 'View Performance'
#     else:
#         children = 'View Predictions'
#     return children

# @app.callback(
#     Output('space-button', 'children'),
#     [Input('space-button', 'n_clicks')])
# def update_space_button(n_clicks):
#     if (n_clicks is None) or (n_clicks % 2 == 0):
#         children = 'View in explanation space'
#     else:
#         children = 'View in feature space'
#     return children

if __name__ == '__main__':
    app.run_server(debug=True, processes=4)
    