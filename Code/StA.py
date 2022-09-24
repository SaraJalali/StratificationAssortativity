from collections import defaultdict
import networkx as nx
import numpy as np

def stratificationAssortativity(G, class_num=4):
    nodes = list(G.nodes)
    edges = list(G.edges())
    h_score = nx.get_node_attributes(G, 'att')
    vals = list(set(list(h_score.values())))
    interv = max(vals) - min(vals)
    class_dic = nx.get_node_attributes(G, 'groups')
    class_vals = list(set(list(class_dic.values())))
    class_indices = dict(zip(class_vals, range(class_num)))

    class_values_dic = {}
    for item in class_vals:
        class_values_dic[item] = []
    for (u, v) in class_dic.items():
        class_values_dic[v].append(u)

    intra_score = {}
    M = np.zeros((class_num, class_num))
    MM = np.zeros((class_num, class_num))

    if interv == 0:
        return 0

    for i in range(min(vals), max(vals) + 1):
        for j in range(i, max(vals) + 1):
            sij = 1 - (abs(i - j) / interv)
            intra_score[(i, j)] = sij
            intra_score[(j, i)] = sij

    node_scores = dict(zip(nodes, [0] * len(nodes)))
    sum1 = 0
    sum2 = 0

    for (u, v) in edges:
        if class_dic[u] == class_dic[v]:
            M[class_indices[class_dic[u]]][class_indices[class_dic[v]]] += 2 * intra_score[(h_score[u], h_score[v])]
            node_scores[u] += intra_score[(h_score[u], h_score[v])]
            node_scores[v] += intra_score[(h_score[u], h_score[v])]
            sum2 += 2* intra_score[(h_score[u], h_score[v])]
        else:
            M[class_indices[class_dic[u]]][class_indices[class_dic[v]]] += (1 - intra_score[(h_score[u], h_score[v])])
            M[class_indices[class_dic[v]]][class_indices[class_dic[u]]] += (1 - intra_score[(h_score[u], h_score[v])])
            node_scores[u] += intra_score[(h_score[u], h_score[v])]
            node_scores[v] += intra_score[(h_score[u], h_score[v])]
            sum2 += 2 * intra_score[(h_score[u], h_score[v])]

    A = sum(M)

    for i in range(class_num):
        if (2 * A[i] - M[i, i])!=0:
            M[i, i] = M[i, i] / (2 * A[i] - M[i, i])
        else:
            M[i, i] = 0

    '''
    ###fast estimation of rand score
    n = len(nodes)
    for c, nodes1 in class_values_dic.items():
        m = len(nodes1)
        o = n - m
        class_scores = sum([node_scores[u] for u in nodes1])
        num = class_scores * class_scores
        den = 2 *(m*o*sum2 - (class_scores * (sum2 - class_scores))) + num
        if den !=0:
            MM[c][c] += num / den
    s = MM.trace()
    '''

    class_scores={}
    for i in class_vals:
        class_scores[i] = [0,0]
    ww_ = {}
    for i in range(len(nodes)):
        u = nodes[i]
        c1 = class_dic[u]
        for j in range(i, len(nodes)):
            c2 = class_dic[v]
            v = nodes[j]
            ww_[(u,v)] = (node_scores[u]*node_scores[v])/ sum2
            if c1 == c2:
                class_scores[c1][0] += ww_[(u, v)]
                class_scores[c1][1] += ww_[(u, v)]
            else:
                class_scores[c1][1] += 1 - ww_[(u, v)]
                class_scores[c2][1] += 1 - ww_[(u, v)]

    rand_score = 0
    for k,v in class_scores.items():
        if v[1] != 0:
            rand_score += v[0]/v[1]
    s= rand_score


    t = M.trace()

    r = (t - s) / (class_num - s)
    return r


def maxStrat(G, numberOfClasses=4):
    intra_score = compute_intrascore(G)
    edges = list(G.edges())
    h_score = nx.get_node_attributes(G, 'att')
    scores = list(set(list(h_score.values())))
    h_score_map = dict(zip(scores, range(len(scores))))
    h_score_map_rev = dict(zip(range(len(scores)), scores))

    A = np.zeros((len(scores), (len(scores))))
    A_ = np.zeros((len(scores), (len(scores))))
    TS ={}

    T1 = {}

    for (u, v) in edges:
        A[h_score_map[h_score[u]]][h_score_map[h_score[v]]] +=  intra_score[(h_score[u], h_score[v])]
        A[h_score_map[h_score[v]]][h_score_map[h_score[u]]] +=  intra_score[(h_score[u], h_score[v])]

        A_[h_score_map[h_score[u]]][h_score_map[h_score[v]]] += 1-intra_score[(h_score[u], h_score[v])]
        A_[h_score_map[h_score[v]]][h_score_map[h_score[u]]] += 1-intra_score[(h_score[u], h_score[v])]

    for i in range(len(scores)):
        for j in range(i,len(scores)):
            s1 = 0
            s2 = 0
            for l in range(i,j+1):
                for k in range(len(scores)):
                    if k in range(i, j+1):
                        s1 += A[l,k]
                    else:
                        s2 += A_[l,k]
            if s1+s2 == 0:
                T1[(i, j)] = [0, [(i, j)]]
            else:
                T1[(i,j)] = [s1/ (s1+2*s2), [(i,j)]]

    TS[1]= T1

    for l in range(2, numberOfClasses+1):
        TS[l] = {}
        for i in range(len(scores)):
            for j in range(i+l-1, len(scores)):
                max_val = -1
                for k in range(i,j):
                    if j - k >=l-1:
                        left = TS[1][i,k]
                        right = TS[l-1][k+1,j]
                        if left[0]+ right[0]>max_val:
                            max_val = left[0]+ right[0]
                            max_groups = left[1].copy()
                            max_groups.extend(right[1].copy())
                            TS[l][(i,j)]= [max_val, max_groups]
    k = numberOfClasses
    while len(TS[k])==0:
        k-=1

    groups1 = TS[k][(0, len(scores) - 1)]
    groups1 = [(h_score_map_rev[a], h_score_map_rev[b]) for (a, b) in groups1[1]]


    group_dic = {}
    mappp = {}
    j = 0
    item = sorted(groups1)
    for groups in item:
        for i in range(groups[0], groups[1] + 1):
            mappp[i] = j
        j += 1
    for u, v in h_score.items():
        try:
            group_dic[u] = mappp[v]
        except:
            group_dic[u] = max(list(mappp.values()))
    nx.set_node_attributes(G, group_dic, 'groups')
    group_list=[[a,b] for (a,b) in item]
    for i in range(1,len(group_list)):
        group_list[i][0]= group_list[i-1][1]+1
    result = scan(G, numberOfClasses, group_list,h_score)
    return result#,group_dic, group_list

def scan(G,class_num, interval, h_score):
    seen = [interval]
    max_val = stratificationAssortativity(G, class_num)
    bool1 = True
    while bool1:
        bool1 = False
        max_current = max_val
        interval_current = interval.copy()
        for i in range(len(interval) - 1):
            (l1, l2) = interval[i]
            (r1, r2) = interval[i + 1]
            score_val1 = 0
            score_val2 = 0
            if l2 - l1 > 0:
                groups1 = interval.copy()
                groups1[i] = [l1, l2 - 1]
                groups1[i + 1] = [l2, r2]
                if groups1 not in seen:
                    score_val1 = score_comp(groups1, G, class_num)
                    seen.append(groups1)
                    if score_val1 > max_val and score_val1 > score_val2:
                        max_current = score_val1
                        interval_current = groups1
            if r2 - r1 > 0:
                groups2 = interval.copy()
                groups2[i] = [l1, r1]
                groups2[i + 1] = [r1 + 1, r2]
                if groups2 not in seen:
                    score_val2 = score_comp(groups2, G, class_num)
                    seen.append(groups2)
                    if score_val2 > max_val and score_val2 > score_val1:
                        max_current = score_val2
                        interval_current = groups2
        if max_current>max_val:
            bool1 = True
            max_val = max_current
            interval = interval_current

    mappp = {}
    j = 0
    group_dic = {}
    for groups in interval:
        for i in range(groups[0], groups[1] + 1):
            mappp[i] = j
        j += 1
    for u, v in h_score.items():
        try:
            group_dic[u] = mappp[v]
        except:
            group_dic[u] = max(list(mappp.values()))
    nx.set_node_attributes(G, group_dic, 'groups')

    result = stratificationAssortativity(G, len(interval))
    return result



def score_comp(groups1, G, numberOfClasses):
    G1 = G.copy()
    h_score = nx.get_node_attributes(G, "att")
    group_dic = {}
    mappp = {}
    j = 0
    item = sorted(groups1)
    for groups in item:
        for i in range(groups[0], groups[1] + 1):
            mappp[i] = j
        j += 1
    for u, v in h_score.items():
        try:
            group_dic[u] = mappp[v]
        except:
            group_dic[u] = max(list(mappp.values()))
    nx.set_node_attributes(G1, group_dic, 'groups')

    return stratificationAssortativity(G1, numberOfClasses)


def compute_intrascore(G):
    intra_score = {}
    h_score = nx.get_node_attributes(G, 'att')
    vals = list(set(list(h_score.values())))
    interv = max(vals) - min(vals)
    h_score = nx.get_node_attributes(G, 'att')
    vals = list(set(list(h_score.values())))
    for i in range(min(vals), max(vals) + 1):
        for j in range(i, max(vals) + 1):
            if interv == 0:
                intra_score[(min(vals), min(vals))] = 0
            else:
                sij = 1 - (abs(i - j) / interv)
                intra_score[(i, j)] = sij
                intra_score[(j, i)] = sij
    return intra_score




def example():
    G = nx.Graph()
    edges = []
    for i in range(1, 5):
        for j in range(i + 1, 5):
            edges.append((i, j))
    for i in range(5, 9):
        for j in range(i + 1, 9):
            edges.append((i, j))
    for i in range(9, 13):
        for j in range(i + 1, 13):
            edges.append((i, j))
    for i in range(13, 17):
        for j in range(i + 1, 17):
            edges.append((i, j))
    att_dic = {1: 1, 2: 2, 3: 3, 4: 4,
               5: 5, 6: 6, 7: 7, 8: 8,
               9: 9, 10: 10, 11: 11, 12: 12,
               13: 13, 14: 14, 15: 15, 16: 16}
    group_dic = {1: 1, 2: 1, 3: 1, 4: 1,
                 5: 2, 6: 2, 7: 2, 8: 2,
                 9: 3, 10: 3, 11: 3, 12: 3,
                 13: 0, 14: 0, 15: 0, 16: 0}
    G.add_edges_from(edges)
    nx.set_node_attributes(G, att_dic, 'att')
    nx.set_node_attributes(G, group_dic, 'groups')
    groups = defaultdict(list)
    for k, v in group_dic.items():
        groups[v].append(k)

    nx.set_node_attributes(G, group_dic, 'groups')
    print("G1")
    print(stratificationAssortativity(G))
    print(nx.attribute_assortativity_coefficient(G, "groups"))
    print(nx.numeric_assortativity_coefficient(G, "att"))


example()