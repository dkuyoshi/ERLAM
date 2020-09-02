import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from sklearn.neighbors import KDTree


class AssociativeMemory(object):
    def __init__(self, capacity, xp, dim=4):
        self.capacity = capacity
        self.key_error_threshold = 10**-7  # (このくらいでいいのか？)
        self.converge_threshold = 10**-6  # (これも変えていく可能性あり)
        self.xp = xp
        self.dim = dim

        # simpleな有向グラフとする (持たせる情報は ht, at, rt, t, Rt)
        # ノードのkeyをどのようにしようか? keyはただの追加順の番号にしてノードの属性に全部入れちゃうか
        self.graph = nx.DiGraph()

        # 埋め込み情報を格納しておくバッファ
        self.keys = self.xp.empty((capacity, dim), dtype=self.xp.float32)

        # LRUの戦略のための色々な情報の初期化
        self.lru_timestamp = self.xp.empty(self.capacity, dtype=self.xp.int32)
        self.current_timestamp = 0
        self.current_size = 0

        # self.queue = deque()

        # エッジ追加用のバッファ
        self.index_list = []
        self.action_list = []

        # update用のインデックス確保
        self.update_index = None
        self.action_number = None

    def append(self, hidden_vector, action, reward, t, Rt):
        if self._search_node(hidden_vector, action):
            self._update_qg(Rt)
        else:
            self._lru_strategy(hidden_vector, action, reward, t, Rt)

        # self.add_edge()

    def append_collectively(self, hidden_vectors, actions, rewards, ids, Rts):
        # 一気にやるパターン
        for hidden_vector, action, reward, t, qg in zip(hidden_vectors, actions, rewards, ids, Rts):
            if self._search_node(hidden_vector, action):
                self._update_qg(qg)
            else:
                self._lru_strategy(hidden_vector, action, reward, t, qg)

        # self.add_edge()

    def get_q(self, hidden_vectors, actions):
        # Qgの獲得
        value = []
        for hidden_vector, action in zip(hidden_vectors, actions):
            nodes = self._lookup_kd(hidden_vector, action)
            qg = []
            for node in nodes:
                self.lru_timestamp[node] = self.current_timestamp
                self.current_timestamp += 1
                qg.append(self.graph.nodes[node]['qg'])
            value.append(self.xp.mean(qg))
        return self.xp.asarray(value, dtype=self.xp.float32)

    def value_propagation(self, gamma):
        # Algorithm1　
        converge = False
        prev_qg = 10**7
        nodes = self._node_sort()

        while not converge:
            update_amount = 0
            for node in nodes:
                neighbor_qg = self._adjacent_qg(node)
                update_qg = self.graph.nodes[node]['reward'] + gamma * max(neighbor_qg)
                prev_qg = self.graph.nodes[node]['qg']
                self.graph.nodes[node]['qg'] = update_qg

                diff = abs(update_qg - prev_qg)
                update_amount += diff

            converge = self._converge_judgement(update_amount)

        print('Converge!')

    def _lru_strategy(self, hidden_vector, action, reward, t, Rt):
        # LRUによるノードの追加
        if self.current_size < self.capacity:
            index = self.current_size
            self.graph.add_node(index, hidden_vector=hidden_vector, action=action, reward=reward, id=t, qg=Rt)
            self.index_list.append(index)
            self.action_list.append(action)
            self.lru_timestamp[index] = self.current_timestamp
            self.current_size += 1
            self.current_timestamp += 1
        else:
            index = np.argmin(self.lru_timestamp)
            self.graph.remove_node(index)
            self.graph.add_node(index, hidden_vector=hidden_vector, action=action, reward=reward, id=t, qg=Rt)
            self.index_list.append(index)
            self.action_list.append(action)
            self.lru_timestamp[index] = self.current_timestamp
            self.current_timestamp += 1

    def _update_qg(self, Rt):
        # 同じノードあれば更新
        self.index_list.append(self.update_index)
        self.action_list.append(self.action_number)
        self.graph.nodes[self.update_index]['qg'] = max(Rt, self.graph.nodes[self.update_index]['qg'])

    def _search_node(self, hidden_vector, action):
        # 同じノードを探す -> Boolで返す(なければFalse) <- 2乗誤差全探索
        for node in self.graph.nodes():
            if self.graph.nodes[node]['action'] == action:
                diff = self.xp.sum((self.graph.nodes[node]['hidden_vector'] - hidden_vector) ** 2, axis=1)
                if diff <= self.key_error_threshold:
                    self.update_index = node
                    self.action_number = action
                    return True
        return False

    def _search_node_kd(self, hidden_vector, action):
        # 同じノードを探す -> Boolで返す(なければFalse) <- kd-treeによる距離計算
        dict_action = nx.get_node_attributes(self.graph, 'action')
        not_keys = [key for key, val in dict_action.items() if val != action]
        dict_hidden = nx.get_node_attributes(self.graph, 'hidden_vector')
        for no in not_keys:
            del dict_hidden[no]
        embeddings_data = np.squeeze(list(dict_hidden.values()))
        embeddings_id = np.asarray(list(dict_hidden.keys()), dtype=np.int32)

        if len(embeddings_data) == 0:
            return False

        tree = KDTree(embeddings_data)
        dist, ind = tree.query(hidden_vector, k=1)

        if dist[0][0] <= self.key_error_threshold:
            self.update_index = embeddings_id[ind[0][0]]
            self.action_number = action
            return True

        return False

    def _lookup(self, hidden_vector, action):
        # もっとも近い状態のやつを取ってくる(2乗誤差からの全探索)
        minimum = 10**7
        get_node = None
        for node in self.graph.nodes():
            if self.graph.nodes[node]['action'] == action:
                distances = self.xp.sum((self.xp.asarray(hidden_vector) - self.graph.nodes[node]['hidden_vector']) ** 2,
                                        axis=1)
                if distances <= minimum:
                    minimum = distances
                    get_node = node

        return get_node

    def _lookup_kd(self, hidden_vector, action):
        # KD-Treeによる最近傍
        dict_action = nx.get_node_attributes(self.graph, 'action')
        not_keys = [key for key, val in dict_action.items() if val != action]
        dict_hidden = nx.get_node_attributes(self.graph, 'hidden_vector')
        for no in not_keys:
            del dict_hidden[no]
        embeddings_data = np.squeeze(list(dict_hidden.values()))
        embeddings_id = np.asarray(list(dict_hidden.keys()), dtype=np.int32)

        tree = KDTree(embeddings_data)
        dist, ind = tree.query(hidden_vector, k=5)

        if len(ind[0]) == 1:
            return embeddings_id[ind].flatten()

        return np.squeeze(embeddings_id[ind])

    def _node_sort(self):
        # nodeをID降順でソートする 
        sorted_node_list = []
        dict_id = nx.get_node_attributes(self.graph, 'id')
        for node, _ in sorted(dict_id.items(), key=lambda x: -x[1]):
            sorted_node_list.append(node)

        return sorted_node_list

    def _converge_judgement(self, diff):
        # convergeしたかどうかのjudge とりあえず更新量 10**-6
        return diff <= self.converge_threshold

    def _adjacent_qg(self, node):
        qg_list = []
        for neighbor in self.graph.adj[node]:
            qg_list.append(self.graph.nodes[neighbor]['qg'])

        # 隣接ノードがない場合
        if len(qg_list) == 0:
            qg_list = [0]

        return qg_list

    def add_edge(self):
        # エッジの追加
        for i in range(len(self.index_list) - 1):
            self.graph.add_edge(self.index_list[i + 1], self.index_list[i], action=self.action_list[i+1])

        self.index_list = []
        self.action_list = []

    def visualize_graph(self):
        # AssociativeMemoryの可視化用
        nx.draw_networkx(self.graph)
        plt.show()

