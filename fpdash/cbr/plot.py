import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

def manifold(df, c='prediction', figsize=(8,6), s=15, title='nMDS'):
    """
    Plot points using a 2D manifold embedding.

        Parameters
        ----------
        df : pandas dataframe
            'x' : mds embedding 0
            'y' : mds embedding 1
            optional:
                'prediction' : predicted probability by classifier
                'label' : label assigned by cluster
        c : string
            - prediction : plot manifold scatterplot colored by prediction probability
            - label : plot manifold scatterplot colored by cluster label
        s : int
            scatterplot node size
    """
    f, ax = plt.subplots(figsize=figsize)
    if c == 'prediction':
        points = ax.scatter(x='x', y='y', c=c, s=s, cmap='Spectral', data=df)
        cbar = f.colorbar(points)
        cbar.set_label('prediction probability')
        plt.title("%s in SHAP space" % title)
    else:
        for label, group in df.groupby([c]):
            points = ax.scatter(group['x'], group['y'], s=s, label=label, cmap='Spectral')
    return