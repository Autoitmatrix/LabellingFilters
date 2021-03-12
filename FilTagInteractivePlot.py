import networkx as nx
import plotly.graph_objects as go
import numpy as np
import pickle
import textwrap
from tqdm import tqdm

# Input
k = 3
filTags_file = r'C:\Users\Daniel\PycharmProjects\dashboards\FilTags\2021-01-30-1722-VGG16-filTags-hr'
save_dir = r'C:\Users\Daniel\PycharmProjects\dashboards\FilTags'

network = filTags_file.split('-')[-3]
with open(filTags_file, 'rb') as data:
    filTags = pickle.load(data)
filTags = filTags[k]
layersize = []
assert isinstance(filTags, list)
for i in range(len(filTags)):
    layersize.append(len(filTags[i]))
layersize_cum = list(np.cumsum(layersize))
layersize_cum.append(0)
# create graph
print("computing graph")
G = nx.complete_multipartite_graph(*layersize)
G = nx.create_empty_copy(G)
print("graph ready")

# create node_trace
print("annotate graph")
node_trace = go.Scatter(x=[], y=[], text=[],
                        textposition="top center",
                        textfont_size=10,
                        mode='markers+text',
                        hoverinfo='text',
                        hovertext=[],
                        marker=dict(color='orange',
                                    size=5,
                                    line=None))
pos = nx.multipartite_layout(G)
layer = -1
fil = 0
for count, node in enumerate(tqdm(G.nodes())):
    if count in layersize_cum:
        layer += 1
        fil = 0
        node_trace['text'] += tuple(['<b>Layer ' + str(layer) + '</b>'])

    else:
        node_trace['text'] += tuple(' ')
        fil += 1
    x, y = pos[node]
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])

    num_Tags = len(filTags[layer][fil])
    if num_Tags > 50:
        filTags_temp = str(filTags[layer][fil][0:50])
        node_trace['hovertext'] += tuple(['<b>layer ' + str(layer) + ',fil ' + str(fil) + '</b><br>' + '<br>'.join(
            textwrap.wrap(filTags_temp, width=100)) + "<i> (and " + str(num_Tags - 50) + " more</i>)"])
    else:
        filTags_temp = str(filTags[layer][fil])
        node_trace['hovertext'] += tuple(['<b>layer ' + str(layer) + ',fil ' + str(fil) + '</b><br>' + '<br>'.join(
            textwrap.wrap(filTags_temp, width=100))])

print("annotation ready")

# plot with plotly
print("plotting graph")
layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)',  # transparent background
                   plot_bgcolor='rgba(0,0,0,0)',  # transparent 2nd background
                   xaxis={'showgrid': False, 'zeroline': False},  # no gridlines
                   yaxis={'showgrid': False, 'zeroline': False},  # no gridlines
                   title="FilTags of " + network + " with k=" + str(k))
config = dict({'modeBarButtonsToRemove': ['toggleSpikelines', 'hoverCompareCartesian', 'select2d', 'lasso2d',
                                          'autoScale2d']})  # 'displaylogo': False,'scrollZoom': True
fig = go.Figure(layout=layout)
fig.add_trace(node_trace)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show(config=config)
fig.write_html(save_dir+'/filTags-k'+ str(k) +'-hr-'+network+'.html')

print("run successful")
