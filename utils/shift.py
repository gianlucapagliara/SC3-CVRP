from datetime import timedelta

import pandas as pd

'''
Work shift abstraction.
'''
class Shift:
  def __init__(self, data, depot, cluster_class, graph_class, start_date=None):
    self.data = data
    self.data.sort_index(inplace=True)
    self.remove_duplicates()
    self.depot = depot
    self.start_date = data.index[-1] if start_date is None else pd.to_datetime(start_date)
    self.cluster_class = cluster_class
    self.graph_class = graph_class
    self.graphs = []
    self.routes = []
  
  def __del__(self):
  	for g in self.graphs:
  		del g
  	del (self.graphs)
  	del (self.data)
  	del (self.routes)

  def get_data(self):
    return self.data

  def remove_duplicates(self, keep='last'):
    self.data = self.data.drop_duplicates(keep=keep)
    return self.data

  '''
  Method to create a group of bins, a cluster.
  It doesn't make clusters if no clustering class was given on initialization.
  '''
  def clusterize(self, columns, max_size=175, kwargs={}):
    if(self.cluster_class is None):
      self.data.loc[:, 'cluster'] = None
    else:
      X = self.data[columns]
      n_clusters = self.data.shape[0] // max_size
      n_clusters += 0 if(self.data.shape[0]%max_size==0) else 1
      cluster_method = self.cluster_class(n_clusters=n_clusters, **kwargs)
      self.data.loc[:, 'cluster'] = cluster_method.fit_predict(X.values.tolist())
    return self.data

  def get_centroids(self):
    if('cluster' not in self.data.columns):
      self.clusterize()
    centroids = self.data.groupby('cluster').mean()
    return centroids
  
  def get_labels(self):
    if('cluster' not in self.data.columns):
      self.clusterize()
    return self.data.loc[:, 'cluster']

  '''
  Get the graphs.
  It returns a graph for every cluster, if clustering class was given on intialization.
  Otherwise, it returns only one graph.
  '''
  def build_graphs(self, columns, kwargs={}):
    if(self.cluster_class is not None):
      clusters = sorted(self.data.cluster.unique())
      for c in clusters:
        current_data = self.data[self.data.cluster == c]
        nodes = [[x, y] for x, y in zip(current_data['latitude'], current_data['longitude'])]
        self.graphs.append(self.graph_class(nodes, start_node=self.depot, end_node=self.depot, **kwargs))
    else:
      nodes = [[x, y] for x, y in zip(self.data['latitude'], self.data['longitude'])]
      self.graphs.append(self.graph_class(nodes, 
                                    start_node=self.depot, end_node=self.depot, **kwargs))
    return self.graphs

  '''
  Get the routes.
  It returns a route for every cluster, if clustering class was given on intialization.
  Otherwise, the routing algorithm is used to make a route for each vehicle.
  '''
  def get_routes(self):
    if(not self.routes):
      if(self.cluster_class is not None):
        clusters = sorted(self.data.cluster.unique())
        for c in clusters:
          current_data = self.data[self.data.cluster == c]
          nodes = [(x, y) for x, y in zip(current_data['latitude'], current_data['longitude'])]
          nodes.insert(0, self.depot)
          nodes.insert(len(nodes), self.depot)
          route = self.graphs[c].get_shortest_path()
          route = route[0] if (isinstance(route, list)) else route
          self.routes.append([nodes[node] for node in route])
      else:
        nodes = [(x, y) for x, y in zip(self.data['latitude'], self.data['longitude'])]
        nodes.insert(0, self.depot)
        nodes.insert(len(nodes), self.depot)
        routes = self.graphs[0].get_shortest_path()
        for route in routes:
          self.routes.append([nodes[node] for node in route])
    return self.routes

  '''
  Get the total distance of the simulated shifts.
  '''
  def get_distance(self):
    total_distance = 0
    for g in self.graphs:
      total_distance += g.get_shortest_path_length()
    return total_distance

  '''
  Get the timetables of the empting of every bin.
  '''
  def get_timetables(self, distance_function, speed=30, emp_time=60):
    timetables = []
    for route in self.routes:
      timetable = []
      timetable.append(self.start_date)
      timetable.append(self.start_date + timedelta(seconds=distance_function(route[0], route[1])/speed*3600))
      for i in range(2, len(route)):
          d = distance_function(route[i], route[i-1])
          current_time = (d/speed)*3600 + emp_time
          timetable.append(timetable[i-1]+timedelta(seconds=current_time))
      timetables.append(timetable)
    return timetables
