import networkx as nx
from networkx.classes import DiGraph
import matplotlib.pyplot as plt
import random
import numpy as np
import math
from scipy import stats
from scipy import spatial
from gensim.models import Word2Vec
import copy


def read_graph(edgeListFile, weighted=False, directed=False):
	"""
	Reads the input network in networkx.
	"""
	if weighted:
		G = nx.read_edgelist(
			edgeListFile,
			nodetype=str,
			data=(('type', int), ('weight', float), ('id', int)),
			create_using=nx.DiGraph()
		)
	else:
		G = nx.read_edgelist(
			edgeListFile,
			nodetype=str,
			data=(('type', int), ('id', int)),
			create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1.0

	if not directed:
		G = G.to_undirected()

	return G

class Edge2Vec:
	def __init__(self, nxGraph=nx.Graph(), em_iteration=5,
							 e_step=3, dimensions=10, walk_length=3, num_walks=2,
							 window_size=5, iter=10, workers=8, p=1, q=1, directed=False):
		self.graph = nxGraph
		self.em_iteration = em_iteration
		self.evaluation_metric = e_step
		self.dimensions = dimensions
		self.walk_length = walk_length
		self.num_walks = num_walks
		self.window_size = window_size
		self.iter = iter
		self.workers = workers
		self.p = p
		self.q = q
		self.directed = directed

		# get the number of edge types
		edge_type_size = len({x['type'] for _, _, x in self.graph.edges(data=True)})
		self.edge_type_size = edge_type_size

		# initialize the transition matrix
		initialized_val = 1.0 / (edge_type_size * edge_type_size)
		self.matrix = [[initialized_val for i in range(edge_type_size)] for j in range(edge_type_size)]

	def edge2vec_walk(self, start_link):
		"""
		return a random walk path
		"""
		# todo remove
		print(f'edge2vec_walk: link = {start_link}')
		walk = [start_link]
		result = [str(start_link[2]['type'])]
		while len(walk) < self.walk_length:
			cur = walk[-1]
			start_node = cur[0]
			end_node = cur[1]
			cur_edge_type = cur[2]['type']
			if self.directed:
				direction_node = end_node
				left_node = start_node
			else:
				start_direction = 1.0 / self.graph.degree(start_node)
				end_direction = 1.0 / self.graph.degree(end_node)
				prob = start_direction / (start_direction + end_direction)
				rand = np.random.rand()
				if prob >= rand:
					direction_node = start_node
					left_node = end_node
				else:
					direction_node = end_node
					left_node = start_node
			neighbors = self.graph.neighbors(direction_node)
			distance_sum = 0
			for neighbor in neighbors:
				neighbor_link = self.graph[direction_node][neighbor]
				neighbor_link_type = neighbor_link['type']
				neighbor_link_weight = neighbor_link['weight']
				trans_weight = self.matrix[cur_edge_type - 1][neighbor_link_type - 1]
				if self.graph.has_edge(neighbor, left_node) or self.graph.has_edge(left_node, neighbor):
					distance_sum += trans_weight * neighbor_link_weight / self.p
				elif neighbor == left_node:
					distance_sum += trans_weight * neighbor_link_weight
				else:
					distance_sum += trans_weight * neighbor_link_weight / self.q
			rand = np.random.rand() * distance_sum
			threshold = 0
			neighbors2 = self.graph.neighbors(direction_node)
			for neighbor in neighbors2:
				neighbor_link = self.graph[direction_node][neighbor]
				neighbor_link_type = neighbor_link['type']
				neighbor_link_weight = neighbor_link['weight']
				trans_weight = self.matrix[cur_edge_type - 1][neighbor_link_type - 1]
				if self.graph.has_edge(neighbor, left_node) or self.graph.has_edge(left_node, neighbor):
					threshold += trans_weight * neighbor_link_weight / self.p
					if threshold >= rand:
						next_link_end_node = neighbor
						break
					elif neighbor == left_node:
						threshold += trans_weight * neighbor_link_weight
					if threshold >= rand:
						next_link_end_node = neighbor
						break
				else:
					threshold += trans_weight * neighbor_link_weight / self.q
					if threshold >= rand:
						next_link_end_node = neighbor
						break
			if distance_sum > 0:
				next_link = self.graph[direction_node][next_link_end_node]
				next_link_tuple = tuple()
				next_link_tuple += (direction_node,)
				next_link_tuple += (next_link_end_node,)
				next_link_tuple += (next_link,)
				walk.append(next_link_tuple)
				result.append(str(next_link_tuple[2]['type']))
			else:
				break
		return result

	def simulate_walks(self):
		"""
		generate random walk paths constrained by transition matrix
		"""
		walks = []
		links = list(self.graph.edges(data=True))
		print('Walk iteration:')
		for walk_iter in range(self.num_walks):
			print(f'{str(walk_iter + 1)} / {str(self.num_walks)}')
			random.shuffle(links)
			count = 1000
			for link in links:
				# todo remove print
				print(f'simulate_walks: link = {link}')
				# walk = self.edge2vec_walk(self, walk_length, matrix, directed, p, q)
				walk = self.edge2vec_walk(link)
				walks.append(walk)
				count = count - 1
				if count == 0 and len(links) > 1000:
					break
		return walks

	def update_trans_matrix(self, walks):
		"""
		E step, update transition matrix
		"""
		type_size = int(self.edge_type_size)
		matrix = [[0 for i in range(type_size)] for j in range(type_size)]
		repo = dict()
		for i in range(type_size):
			repo[i] = []
		for walk in walks:
			print(f'walk = {walk}')
			curr_repo = dict()
			for edge in walk:
				edge_id = int(edge) - 1
				if edge_id in curr_repo:
					curr_repo[edge_id] = curr_repo[edge_id] + 1
				else:
					curr_repo[edge_id] = 1
			for i in range(type_size):
				if i in curr_repo:
					repo[i].append(curr_repo[i])
				else:
					repo[i].append(0)
		for i in range(type_size):
			for j in range(type_size):
				if self.evaluation_metric == 1:
					sim_score = Edge2Vec.wilcoxon_test(repo[i], repo[j])
					matrix[i][j] = sim_score
				elif self.evaluation_metric == 2:
					sim_score = Edge2Vec.entroy_test(repo[i], repo[j])
					matrix[i][j] = sim_score
				elif self.evaluation_metric == 3:
					sim_score = Edge2Vec.spearmanr_test(repo[i], repo[j])
					matrix[i][j] = sim_score
				elif self.evaluation_metric == 4:
					sim_score = Edge2Vec.pearsonr_test(repo[i], repo[j])
					matrix[i][j] = sim_score
				else:
					raise ValueError('not correct evaluation metric! You need to choose from 1-4')
		return matrix

	@staticmethod
	def wilcoxon_test(v1, v2):
		if v1 == v2:
			return 1 / (math.sqrt(0) + 1)
		result = stats.wilcoxon(v1, v2).statistic
		if result != result:
			result = 0
		return 1 / (math.sqrt(result) + 1)

	@staticmethod
	def entroy_test(v1, v2):
		if v1 == v2:
			return 0
		result = stats.entropy(v1, v2)
		result = stats.wilcoxon(v1, v2).statistic
		if result != result:
			result = 0
		return result

	@staticmethod
	def spearmanr_test(v1, v2):
		if v1 == v2:
			return Edge2Vec.sigmoid(-1)
		result = stats.mstats.spearmanr(v1, v2).correlation
		result = stats.wilcoxon(v1, v2).statistic
		if result != result:
			result = -1
		return Edge2Vec.sigmoid(result)

	@staticmethod
	def pearsonr_test(v1, v2):
		if v1 == v2:
			return Edge2Vec.sigmoid(-1)
		result = stats.mstats.pearsonr(v1, v2)[0]
		result = stats.wilcoxon(v1, v2).statistic
		if result != result:
			result = -1
		return Edge2Vec.sigmoid(result)

	@staticmethod
	def sigmoid(x):
		return 1 / (1 + math.exp(-x))

	@staticmethod
	def standardization(x):
		return (x + 1) / 2

	@staticmethod
	def relu(x):
		return (abs(x) + x) / 2

	def gen_transition_matrix(self):
		print("------begin to simulate walk---------")
		for i in range(self.em_iteration):
			walks = self.simulate_walks()
			print(f'{str(i)}, "th iteration for updating transition matrix!"')
			trans_matrix = self.update_trans_matrix(walks)
		print("------finish!---------")
		return trans_matrix
