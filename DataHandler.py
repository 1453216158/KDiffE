import pickle
import numpy as np
from scipy.sparse import coo_matrix
from Params import args
import scipy.sparse as sp
import torch
import torch.utils.data as data
import torch.utils.data as dataloader
from collections import defaultdict
from tqdm import tqdm
import random
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(4)}")
else:
    device = torch.device("cpu")
    print("Using CPU")
class DataHandler:
	def __init__(self):
		if args.data == 'tiktok':
			predir = './Datasets/tiktok/'
		elif args.data == 'baby':
			predir = './Datasets/baby/'
		elif args.data == 'sports':
			predir = './Datasets/sports/'
		self.predir = predir
		self.trnfile = predir + 'trnMat.pkl'
		self.tstfile = predir + 'tstMat.pkl'
		self.kgfile = predir + 'kg.txt'
		self.imagefile = predir + 'image_feat.npy'
		self.textfile = predir + 'text_feat.npy'
		if args.data == 'tiktok':
			self.audiofile = predir + 'audio_feat.npy'
		self.S_file = predir + 'Similarity_' + args.similarity + '.pkl'

	def loadOneFile(self, filename,num_rows_to_read):
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
			ret = ret[:num_rows_to_read, :]
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
			# print(ret.shape)
		return ret

	def readTriplets(self, file_name):
		can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
		can_triplets_np = np.unique(can_triplets_np, axis=0)

		inv_triplets_np = can_triplets_np.copy()
		inv_triplets_np[:, 0] = can_triplets_np[:, 2]
		inv_triplets_np[:, 2] = can_triplets_np[:, 0]
		inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
		triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
		n_relations = max(triplets[:, 1]) + 1
		args.relation_num = n_relations
		args.entity_n = max(max(triplets[:, 0]), max(triplets[:, 1])) + 1
		return triplets
	
	def buildGraphs(self, triplets):
		kg_dict = defaultdict(list)
		kg_edges = list()

		print("Begin to load knowledge graph triples ...")

		kg_counter_dict = {}

		for h_id, r_id, t_id in tqdm(triplets, ascii=True):
			if h_id not in kg_counter_dict.keys():
				kg_counter_dict[h_id] = set()
			if t_id not in kg_counter_dict[h_id]:
				kg_counter_dict[h_id].add(t_id)
			else:
				continue
			kg_edges.append([h_id, t_id, r_id])
			kg_dict[h_id].append((r_id, t_id))

		return kg_edges, kg_dict
	
	def buildKGMatrix(self, kg_edges):
		edge_list = []
		for h_id, t_id, r_id in kg_edges:
			edge_list.append((h_id, t_id))
		edge_list = np.array(edge_list)

		kgMatrix = sp.csr_matrix((np.ones_like(edge_list[:,0]), (edge_list[:,0], edge_list[:,1])), dtype='float64', shape=(args.entity_n, args.entity_n))

		return kgMatrix

	def normalizeAdj(self, mat, S_values):
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		mat = mat + S_values[:mat.shape[0], :mat.shape[0]]
		return dInvSqrtMat.transpose().dot(mat).dot(dInvSqrtMat).tocoo()

	def random_walk(self, adj_matrix, start_node, steps):
			# 变量初始化
			current_node = start_node
			walk_sequence = []
			previous_node = start_node
			restart = 0

			for _ in range(steps):
				neighbors = adj_matrix[
					current_node].indices
				if restart > random.random():
					current_node = previous_node
					walk_sequence.pop()
					restart = 0
					_ -= 1
					continue
				else:
					if len(neighbors) > 0:
						previous_node = current_node
						current_node = np.random.choice(neighbors)
						walk_sequence.append(current_node)
					else:
						break
				restart += 0.2

			return walk_sequence


	def compute_jaccard_similarity(self, set_a, set_b):
			intersection = len(set_a & set_b)
			union = len(set_a | set_b)
			return intersection / union if union > 0 else 0

	def compute_attention(self, adj_matrix, M, R, epsilon):
			print("开始构建相似度矩阵")
			S_values = np.zeros(adj_matrix.shape)
			node_visits = defaultdict(list)
			for u in range(adj_matrix.shape[0]):
				for v in range(adj_matrix.shape[1]):
					if v > u:
						break
					if adj_matrix[u, v] == 0:
						continue
					for _ in range(R):
						start_node_u = u
						start_node_v = v
						u_walk_sequence = self.random_walk(adj_matrix, start_node_u,M)
						v_walk_sequence = self.random_walk(adj_matrix, start_node_v,M)
						node_visits[u].append(u_walk_sequence)
						node_visits[v].append(v_walk_sequence)
						total_visits_u = set().union(*node_visits[u])
						total_visits_v = set().union(*node_visits[v])
						similarity = self.compute_jaccard_similarity(total_visits_u, total_visits_v)
						S_values[u, v] = similarity
						S_values[v, u] = similarity
						S_uv = epsilon * similarity
						S_values[u][v] = S_uv
						S_values[v][u] = S_uv
					print('相似度 %d/%d' % (u, v), S_uv)
			S_values = sp.csr_matrix(S_values)
			with open(self.predir + 'Similarity_130_6.pkl', 'wb') as file:
				pickle.dump(S_values, file)
			print(type(S_values))
			return S_values

	def makeTorchAdj(self, mat):
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		# self.compute_attention(mat, args.M, args.R, args.epsilon)
		with open(self.S_file, 'rb') as file:
			 S_values= pickle.load(file)
		print("S_values-----",self.S_file,args.epsilon)
		S_values = args.epsilon * S_values
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat, S_values)

		# make cuda tensor
		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)
		return torch.sparse.FloatTensor(idxs, vals, shape).to(device)

	def RelationDictBuild(self):
		relation_dict = {}
		for head in self.kg_dict:
			relation_dict[head] = {}
			for (relation, tail) in self.kg_dict[head]:
				relation_dict[head][tail] = relation
		return relation_dict

	def buildUIMatrix(self, mat):
		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)
		return torch.sparse.FloatTensor(idxs, vals, shape).to(device)

	def LoadData(self):
		trnMat = self.loadOneFile(self.trnfile,35598)
		tstMat = self.loadOneFile(self.tstfile,35598)
		self.trnMat = trnMat
		args.user, args.item = trnMat.shape
		# print("user, item: ", args.user, args.item)
		self.torchBiAdj = self.makeTorchAdj(trnMat)

		self.ui_matrix = self.buildUIMatrix(trnMat)
		trnData = TrnData(trnMat)
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

		self.diffusionData1 = DiffusionData(torch.FloatTensor(self.trnMat.A))
		self.diffusionLoader1 = dataloader.DataLoader(self.diffusionData1, batch_size=args.batch, shuffle=True,num_workers=0)

		kg_triplets = self.readTriplets(self.kgfile)
		self.kg_edges, self.kg_dict = self.buildGraphs(kg_triplets)

		self.kg_matrix = self.buildKGMatrix(self.kg_edges)
		print("kg shape: ", self.kg_matrix.shape)
		print("number of edges in KG: ", len(self.kg_edges))

		self.diffusionData = DiffusionData(torch.FloatTensor(self.kg_matrix.A))
		self.diffusionLoader = dataloader.DataLoader(self.diffusionData, batch_size=args.batch, shuffle=True,num_workers=0)
		self.relation_dict = self.RelationDictBuild()


		self.image_feats, args.image_feat_dim = self.loadFeatures(self.imagefile)
		self.text_feats, args.text_feat_dim = self.loadFeatures(self.textfile)
		if args.data == 'tiktok':
			self.audio_feats, args.audio_feat_dim = self.loadFeatures(self.audiofile)


		self.diffusionData1 = DiffusionData(torch.FloatTensor(self.trnMat.A))
		self.diffusionLoader1 = dataloader.DataLoader(self.diffusionData1, batch_size=args.batch, shuffle=True,num_workers=0)
	def loadFeatures(self, filename):
		feats = np.load(filename)
		return torch.tensor(feats).float().to(device), np.shape(feats)[1]

class TrnData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)

	def negSampling(self):
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
	def __init__(self, coomat, trnMat):
		self.csrmat = (trnMat.tocsr() != 0) * 1.0

		tstLocs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = tstLocs

	def __len__(self):
		return len(self.tstUsrs)

	def __getitem__(self, idx):
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])
	
class DiffusionData(data.Dataset):
	def __init__(self, data):
		self.data = data
	
	def __getitem__(self, index):
		item = self.data[index]
		return item, index
	
	def __len__(self):
		return len(self.data)