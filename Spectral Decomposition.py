import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import copy
if __name__ == "__main__":

    def import_facebook_data(path):                     #importing the FB data
        X = np.loadtxt(path, dtype=int)
        X = sorted(X, key=lambda x: (x[0], x[1]))       #To avoid duplicacies, sorted each element and returned unique
        return np.unique(X, axis=0)

    def spectralDecomp_OneIter(nodes_connectivity_list_fb):
        h_map = dict()                                  #Forward Map initialized
        inv_hmap = dict()                               #Reverse Map initialized
        nodes = sorted(list(set(nodes_connectivity_list_fb.ravel())))       #get to total set of unique nodes
        length = len(nodes)
        for i in range(len(nodes)):
            h_map[nodes[i]] = i
            inv_hmap[i] = nodes[i]                      #Created the forward and reverse Mappings so that the func can be used iteratively
        adj_mat_fb = np.zeros([length, length])         #Adjacency matrix
        for i in range(len(nodes_connectivity_list_fb)):
            adj_mat_fb[h_map[nodes_connectivity_list_fb[i, 0]],
                       h_map[nodes_connectivity_list_fb[i, 1]]] = 1
            adj_mat_fb[h_map[nodes_connectivity_list_fb[i, 1]],
                       h_map[nodes_connectivity_list_fb[i, 0]]] = 1

        D_array = np.array(np.sum(adj_mat_fb, axis=0))
        D = np.diag(D_array.squeeze())
        L = D - adj_mat_fb                              #Laplacian

        inv_D_root = np.sqrt(np.linalg.inv(D))
        new_L = inv_D_root @ L @ inv_D_root


        eig, eig_vec = np.linalg.eigh(new_L)
        fielder_vec_fb = eig_vec[:, 1]
        positive = np.argwhere(fielder_vec_fb >= 0).squeeze() #Split the indexes of fiedler vector based on "Sign"
        negative = np.argwhere(fielder_vec_fb < 0).squeeze()

        P = []
        N = []
        if len(positive) != 0:

            for i in range(len(positive)):              #With reverse mapping found the original identities of node indexes
                P.append(inv_hmap[positive.tolist()[i]])

            p_id = min(P)
            P = np.column_stack((P, np.ones_like(P) * p_id))
        if len(negative) != 0:
            for i in range(len(negative)):
                N.append(inv_hmap[negative.tolist()[i]])
            n_id = min(N)
            N = np.column_stack((N, np.ones_like(N) * n_id))

        if len(P) != 0 and len(N) != 0:                 #Created the if-else just in case all elements of Fiedler vector are of same sign
            graph_partition_fb = np.vstack((P, N))
        elif len(P) != 0:
            graph_partition_fb = P
        elif len(N) != 0:
            graph_partition_fb = N
        else:
            graph_partition_fb = None

        return fielder_vec_fb, adj_mat_fb, graph_partition_fb

    def spectralDecomposition(nodes_connectivity_list_fb):
        fielder_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(
            nodes_connectivity_list_fb)                 #Drew the first 2 partitions using the 1st function
        graph_dict = {}                                 #Dictionary that contains the community ids of each element
        nodes = graph_partition_fb[:, 0]
        ids = graph_partition_fb[:, 1]
        for A, B in zip(nodes, ids):
            graph_dict[A] = B
        count = 0
        while (True):
            ids = list(graph_dict.values())
            unique_ids = np.unique(ids).tolist()        #Unique IDs in the current iteration
            length = []                                         #contains the cluster size of each id

            for index in range(len(unique_ids)):
                length.append(np.sum(np.array(ids) == unique_ids[index]))
                longest = max(length)
            ################################################   CONVERGENCE CRITERION
            if longest < 300:                           #for FB dataset
                break
            ################################################

            longest_id = unique_ids[np.argwhere(
                np.array(length) == longest)[0].squeeze().tolist()]  #located the nodes of the largest cluster
            if longest_id is int:                                   #checking if the cluster contains a single node
                nodes_in_cluster = list(
                    np.array([
                        k for k, v in graph_dict.items()
                        if int(v) == longest_id
                    ]))
            else:
                nodes_in_cluster = list(
                    np.array([
                        k for k, v in graph_dict.items()
                        if int(v) == longest_id
                    ]))

            ################################################    NEW NODELIST METHOD
            new_nodelist = []                               #Creating a new nodelist based on the nodes of largest cluster
            new_nodelist_cand = list(
                itertools.permutations(nodes_in_cluster, 2))
            set_totallist = set(tuple(i) for i in nodes_connectivity_list_fb)
            set_newlist = set(tuple(i) for i in new_nodelist_cand)

            new_nodelist = np.array(list(set_totallist
                                         & set_newlist)).squeeze()

            ################################################

            fielder_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(
                new_nodelist)

            graph_dict_updated = {}
            nodes_new = graph_partition_fb[:, 0]
            ids_new = graph_partition_fb[:, 1]
            for A, B in zip(nodes_new, ids_new):
                graph_dict_updated[A] = B
            graph_dict_prior = copy.deepcopy(graph_dict)            #creating a copy to check if the graph has been updated
            graph_dict.update(graph_dict_updated)
            if graph_dict_prior == graph_dict:
                break

        #         count = count + 1
        graph_partition_final = np.array(list(graph_dict.items()))
        return graph_partition_final

    def createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb):
        adj_mat = np.zeros([
            nodes_connectivity_list_fb.max() + 1,
            nodes_connectivity_list_fb.max() + 1
        ])
        clustered_adj_mat_fb = np.zeros_like(adj_mat)
        for i in range(len(nodes_connectivity_list_fb)):
            adj_mat[nodes_connectivity_list_fb[i, 0],
                    nodes_connectivity_list_fb[i, 1]] = 1
            adj_mat[nodes_connectivity_list_fb[i, 1],
                    nodes_connectivity_list_fb[i, 0]] = 1

        unique_ids = np.unique(graph_partition_fb[:, 1]).tolist()
        length = []
        ids = graph_partition_fb[:, 1]
        for index in range(len(unique_ids)):
            length.append(np.sum(np.array(ids) == unique_ids[index])) #Found the size of each cluster

        unique_ids = [
            x for _, x in sorted(zip(length, unique_ids),
                                 key=lambda pair: pair[0])
        ]
        node_order = np.array([])                                      #Created node order based on unique ids
        for index in range(len(unique_ids)):
            node_order = np.hstack((node_order, graph_partition_fb[:, 0][
                graph_partition_fb[:, 1] == unique_ids[index]]))
        node_order = list(map(int, node_order))
        clustered_adj_mat_fb = adj_mat[node_order, :][:, node_order]

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(clustered_adj_mat_fb, cmap="viridis", interpolation="none")
        return clustered_adj_mat_fb

    def louvain_one_iter(nodes_connectivity_list_fb):

        adj_mat = np.zeros([
            nodes_connectivity_list_fb.max() + 1,
            nodes_connectivity_list_fb.max() + 1
        ])
        mod_mat = np.zeros_like(adj_mat)
        D_mat = np.zeros_like(adj_mat)
        for i in range(len(nodes_connectivity_list_fb)):
            adj_mat[nodes_connectivity_list_fb[i, 0],
                    nodes_connectivity_list_fb[i, 1]] = 1
            adj_mat[nodes_connectivity_list_fb[i, 1],
                    nodes_connectivity_list_fb[i, 0]] = 1

        D_array = np.array(np.sum(adj_mat, axis=0))
        for i in range(len(D_array)):
            for j in range(len(D_array)):
                mod_mat[i, j] = adj_mat[i, j] - D_array[i] * D_array[j] / (
                    2 * len(nodes_connectivity_list_fb))

        nodes = sorted(list(set(nodes_connectivity_list_fb.ravel().tolist())))
        dict_clust = dict()
        for A, B in zip(nodes, nodes):
            dict_clust[A] = B
        del_Q = np.zeros([len(nodes), len(nodes)])
        Q_val_init = np.trace(mod_mat) / (2 * len(nodes_connectivity_list_fb))
        DICT_association = {k: [] for k in nodes}

        for i in nodes_connectivity_list_fb:
            DICT_association[i[1]].append(i[0])
            DICT_association[i[0]].append(i[1])

        unique_ids = sorted(list(set(list(dict_clust.values()))))
        Q_max = Q_val_init
        ###FOR 1st Iteration, we can simply traverse the nodelist provided and find the pair for which we obtain the highest modularity
        for i in range(len(nodes_connectivity_list_fb)):
            node = nodes_connectivity_list_fb[i, :].ravel().tolist()
            mod_mat_pair = mod_mat[node, :][:, node]
            Q_val_pair = np.sum(mod_mat_pair) / (2 * len(nodes_connectivity_list_fb))

            # node_rem = list(set(nodes) - set(node))
            # mod_mat_rem = mod_mat[node_rem, :][:, node_rem]
            # Q_val_rem = np.trace(mod_mat_rem) / (
            #     2 * len(nodes_connectivity_list_fb))

            Q_obtained =  Q_val_pair #+ Q_val_rem
            if Q_obtained > Q_max:
                Q_max = Q_obtained
                best_pair = nodes_connectivity_list_fb[i, :]


        p = min(best_pair)
        q = max(best_pair)
        print(p, q)
        node_rem = list(set(nodes) - set(best_pair))
        mod_mat_rem = mod_mat[node_rem, :][:, node_rem]
        Q_val_rem = np.trace(mod_mat_rem) / (2 * len(nodes_connectivity_list_fb))
        print(Q_max + Q_val_rem)
        dict_clust[q] = p
        graph_partition_louvian_fb = np.array(list(dict_clust.items()))
        return graph_partition_louvian_fb

    ############ Answer qn 1-4 for facebook data #################################################
    # Import facebook_combined.txt
    # nodes_connectivity_list is a nx2 numpy array, where every row
    # is a edge connecting i<->j (entry in the first column is node i,
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    nodes_connectivity_list_fb = import_facebook_data(
        "../data/facebook_combined.txt")

    # This is for question no. 1
    # fielder_vec    : n-length numpy array. (n being number of nodes in the network)
    # adj_mat        : nxn adjacency matrix of the graph
    # graph_partition: graph_partition is a nx2 numpy array where the first column consists of all
    #                  nodes in the network and the second column lists their community id (starting from 0)
    #                  Follow the convention that the community id is equal to the lowest nodeID in that community.
    fielder_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(
        nodes_connectivity_list_fb)

    # This is for question no. 2. Use the function
    # written for question no.1 iteratively within this function.
    # graph_partition is a nx2 numpy array, as before. It now contains all the community id's that you have
    # identified as part of question 2. The naming convention for the community id is as before.
    graph_partition_fb = spectralDecomposition(nodes_connectivity_list_fb)

    # This is for question no. 3
    # Create the sorted adjacency matrix of the entire graph. You will need the identified communities from
    # question 3 (in the form of the nx2 numpy array graph_partition) and the nodes_connectivity_list. The
    # adjacency matrix is to be sorted in an increasing order of communities.
    clustered_adj_mat_fb = createSortedAdjMat(graph_partition_fb,
                                              nodes_connectivity_list_fb)

    # This is for question no. 4
    # run one iteration of louvain algorithm and return the resulting graph_partition. The description of
    # graph_partition vector is as before.
    graph_partition_louvain_fb = louvain_one_iter(nodes_connectivity_list_fb)
