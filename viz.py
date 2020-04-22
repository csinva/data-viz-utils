from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D

def heatmap_extended(data, cond1, cond2, show_cbar=True, annot=False, 
              cmap=sns.color_palette("viridis", n_colors=1000),
              fontsize_small=10):

    plt.figure(figsize=(6, 6)) 
    gs = gridspec.GridSpec(20, 20) # split up the space into a grid
    topheight = 4  # height of top plot (based on grid)
    rightwidth = 5 # width of right plot (based on grid)

    # heatmap
    ax2 = plt.subplot(gs[topheight:, :-rightwidth])
    im = sns.heatmap(data.T, cmap=cmap, cbar=False, annot=annot)
    plt.xlabel('X1')
    plt.ylabel('X2')
    
    # top plot
    ax1 = plt.subplot(gs[:topheight, :-rightwidth])
    plt.xticks([])
    plt.plot(cond1)
    plt.ylabel('Conditional 1', fontsize=fontsize_small)

    # right plot
    ax3 = plt.subplot(gs[topheight:, -rightwidth:])
    plt.yticks([])
    plt.plot(cond2, range(len(cond2))[::-1])
    plt.xlabel('Conditional 2', fontsize=fontsize_small)

    plt.tight_layout()
    
    # colorbar
    if show_cbar:

        # create a dummy plot and then remove it
        plt.subplot(gs[0, -1])
        cm = LinearSegmentedColormap.from_list('x', cmap)
        im_c = plt.imshow(data, cmap=cm)
        plt.cla()
        plt.axis('off')

        # partition the right space to add colorbar
        plt.subplot(gs[topheight:, -rightwidth:])
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="20%", pad=0.1)

        # make colorbar based on dummy plot
        cb = plt.colorbar(im_c, cax=cax)
        cb.ax.tick_params(labelsize=8) 
        cb.outline.set_visible(False)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#     fig, ax = plt.subplots()
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    ax = plt.gca()
#     ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
#            title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return ax


def plot_pcs(pca, feat_names=None):
    '''Pretty plot of pcs with explained var bars
    Params
    ------
    pca: sklearn PCA class after being fitted
    '''
    # plt.figure(figsize=(6, 9), dpi=200)
    
    # extract out relevant pars
    comps = pca.components_.transpose()
    var_norm = pca.explained_variance_ / np.sum(pca.explained_variance_) * 100
    
    
    # create a 2 X 2 grid 
    gs = gridspec.GridSpec(2, 2, height_ratios=[2,10], 
                      width_ratios=[12, 1], wspace=0.1, hspace=0)

    
    # plot explained variance
    ax2 = plt.subplot(gs[0])
    ax2.bar(np.arange(0, comps.shape[1]), var_norm, 
            color='gray', width=0.8)
    plt.title('Explained variance (%)')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.yaxis.set_ticks_position('left')
    ax2.set_yticks([0, max(var_norm)])
    plt.xlim((-0.5, comps.shape[1]-0.5))
    
    # plot pcs
    ax = plt.subplot(gs[2])
    vmaxabs = np.max(np.abs(comps))
    p = ax.imshow(comps, interpolation='None', aspect='auto',
                  cmap=sns.diverging_palette(10, 240, as_cmap=True, center='light'),
                  vmin=-vmaxabs, vmax=vmaxabs) # center at 0
    plt.xlabel('PCA component number')
    
    if feat_names is not None:
        ax.set_yticklabels(list(feat_names))
        ax.set_yticks(range(len(list(feat_names))))
    

    # make colorbar
    colorAx = plt.subplot(gs[3])
    cb = plt.colorbar(p, cax=colorAx)
    
def jointplot_grouped(col_x: str, col_y: str, col_k: str, df, 
                      k_is_color=False, scatter_alpha=.5, add_global_hists: bool=True):
    '''Jointplot of hists + densities
    Params
    ------
    col_x
        name of X var
    col_y
        name of Y var
    col_k
        name of variable to group/color by
    add_global_hists
        whether to plot the global hist as well
    '''
    def colored_scatter(x, y, c=None):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            plt.scatter(*args, **kwargs)

        return scatter

    g = sns.JointGrid(
        x=col_x,
        y=col_y,
        data=df
    )
    color = None
    legends=[]
    for name, df_group in df.groupby(col_k):
        legends.append(name)
        if k_is_color:
            color=name
        g.plot_joint(
            colored_scatter(df_group[col_x],df_group[col_y],color),
        )
        sns.distplot(
            df_group[col_x].values,
            ax=g.ax_marg_x,
            color=color,
        )
        sns.distplot(
            df_group[col_y].values,
            ax=g.ax_marg_y,
            color=color,            
            vertical=True
        )
    if add_global_hists:
        sns.distplot(
            df[col_x].values,
            ax=g.ax_marg_x,
            color='grey'
        )
        sns.distplot(
            df[col_y].values.ravel(),
            ax=g.ax_marg_y,
            color='grey',
            vertical=True
        )
    plt.legend(legends)

def scatter_2_legends(x, y, c, s, xlab: str, ylab: str, colorlab: str,
                 sizelab: str, markersize_rescaling: int, figsize=(7, 3)):
    '''
    Params
    ------
    markersize_rescaling: 
    '''
    # Unique category labels: 'D', 'F', 'G', ...
    color_labels = c

    # List of RGB triplets
    rgb_values = sns.color_palette("Set2", len(color_labels))

    # Map label to RGB
    color_map = dict(zip(color_labels, rgb_values))

    # Finally use the mapped values
    colors = c.map(color_map)
#     plt.scatter(df['carat'], df['price'], c=df['color'].map(color_map))
    rgb_values = sns.color_palette("Set2", 8)

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(x, y, c=colors, s=s, alpha=1)
    plt.yscale('symlog')
    plt.xscale('symlog')

    # produce a legend with the unique colors from the scatter
    leg_els = []
    for k in color_map:
        leg_els.append(Line2D([0], [0], marker='o', color='w', label=k, markerfacecolor=color_map[k], markersize=6))

    legend1 = ax.legend(handles=leg_els, loc="upper left", title=colorlab, fontsize=9)
    ax.add_artist(legend1)

    # produce a legend with a cross section of sizes from the scatter
    handles, labels = scatter.legend_elements(prop="sizes", alpha=1)
    l2 = []
    for i in range(len(labels)):
        s = labels[i]
        num = markersize_rescaling * round(float(s[s.index('{') + 1: s.index('}')]), 2)
        l2.append('$\\mathdefault{' + str(num) + '}$')
    legend2 = ax.legend(handles, l2, loc="lower right", title=sizelab)
    plt.xlabel(xlab)
    plt.ylabel(ylab)