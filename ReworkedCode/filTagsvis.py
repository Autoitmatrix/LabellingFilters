"""# Visualization of filTags"""

# imports
import pickle
import networkx as nx
import plotly.graph_objects as go
import textwrap
from tqdm import tqdm
from keras import applications

# input
filTags_file="/content/Results/2021-01-01-0000-TF-VGG16-vars"
k=5
q=5
show_last_layers=7

# preparation
network=filTags_file.split('-')[-2]
cnn = getattr(applications, network)()
network_module = getattr(applications, network_mod[network])
layer_name={}
convlayer=0
for i, layer in enumerate(cnn.layers):
  if 'conv' in layer.name:
    layer_name[convlayer]=layer.name
    convlayer+=1
    
# get filTags (from Step2)
with open(filTags_file, "rb") as p:
  filTags_k=pickle.load(p)
  filTags_q=pickle.load(p)

# create plots
def plotfilTags(filTags,scope,k_q,show):
  # prepare filTags
  num_layers =len(filTags)
  if show>num_layers:
    show=num_layers
  for i in range(num_layers-show,num_layers):
      layersize.append(len(filTags[i]))
  layersize_cum = list(np.cumsum(layersize))
  layersize_cum.insert(0,0)

  # create graph
  print("computing graph")
  G = nx.Graph()
  node_dict=[]
  num_nodes=layersize_cum[-1]
  layer=0
  for i in tqdm(range(num_nodes)):
    if i in layersize_cum:
      layer+=1
    node_dict.append((i,{'subset':layer}))
  G.add_nodes_from(node_dict)
  print("graph ready")

  # create annotation (node trace)
  print("annotate graph")
  node_trace = go.Scatter(x=[], y=[], text=[],
                          textposition="top center",
                          textfont_size=10,
                          mode='markers+text',
                          hoverinfo='text',
                          hovertext=[],
                          marker=dict(color='orange',
                                      size=10,
                                      line=None))
  pos = nx.multipartite_layout(G)
  layer = (num_layers)-(show)-1
  fil = 0
  for count, node in enumerate(tqdm(G.nodes())):
      if count in layersize_cum:
          layer += 1
          fil = 0
          node_trace['text'] += tuple(['<b>' + layer_name[layer] + '</b>'])
      else:
          node_trace['text'] += tuple(' ')
          fil += 1
      x, y = pos[node]
      node_trace['x'] += tuple([x])
      node_trace['y'] += tuple([y])
      num_Tags = len(filTags[layer][fil])
      if num_Tags > 50:
          filTags_temp = str(filTags[layer][fil][0:50])
          node_trace['hovertext'] += tuple(['<b>' + layer_name[layer] + ',fil ' + str(fil) + '</b><br>' + '<br>'.join(
              textwrap.wrap(filTags_temp, width=100)) + "<i> (and " + str(num_Tags - 50) + " more</i>)"])
      else:
          filTags_temp = str(filTags[layer][fil])
          node_trace['hovertext'] += tuple(['<b>' + layer_name[layer] + ',fil ' + str(fil) + '</b><br>' + '<br>'.join(
              textwrap.wrap(filTags_temp, width=100))])
  print("annotation ready")

  # plot with plotly
  print("plotting graph")
  layout = go.Layout(paper_bgcolor='rgba(0,38,77,1)',  # dark blue frame
                    plot_bgcolor='rgba(255,255,255,1)',  # white background
                    xaxis={'showgrid': False, 'zeroline': False},  # no gridlines
                    yaxis={'showgrid': False, 'zeroline': False},  # no gridlines
                    title={'text':"FilTags of " + network + " with k=" + str(k),
                            'font':  dict(family="Arial",size=18, color="white"),
                            'xanchor': 'center',
                            'x':0.5})
  config = dict({'modeBarButtonsToRemove': ['toggleSpikelines', 
                                            'hoverCompareCartesian', 
                                            'select2d', 'lasso2d',
                                            'autoScale2d']})
  fig = go.Figure(layout=layout)
  fig.add_trace(node_trace)
  fig.update_xaxes(showticklabels=False)
  fig.update_yaxes(showticklabels=False)
  fig.show(config=config)
  fig.write_html('filTags-'+scope+ str(k_q) +network+'.html')

plotfilTags(filTags_k[k],'k',k,show_last_layers)
plotfilTags(filTags_q[q],'q',q,show_last_layers)