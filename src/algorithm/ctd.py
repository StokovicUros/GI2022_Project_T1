"""

    Implements probability diffusion algorithm in both recursive and iterative form.

"""
THRESHOLD_DIFF = 0.01


def DIFFUSE_PROB_RECURSIVE(p1, sn, G, vN, adj_mat, alpha=0.5):
    u_vN = G.keys() - vN
    UNsn = set(filter(lambda x: adj_mat.iloc[sn, x] > 0, u_vN))

    if len(UNsn) == 0:
        if len(u_vN) == 0:
            return
        probability_for_all = p1 / len(u_vN)
        for node in u_vN:
            G[node] += probability_for_all
        return

    sum_weights = 0
    for node in UNsn:
        sum_weights += adj_mat.iloc[sn, node]

    multiplier = p1 / sum_weights
    for node in UNsn:
        inherited_prob = multiplier * adj_mat.iloc[sn, node]
        G[node] += inherited_prob
        if inherited_prob * alpha > THRESHOLD_DIFF:
            G[node] -= inherited_prob * alpha
            DIFFUSE_PROB_RECURSIVE(inherited_prob * alpha, node, G, vN.union({node}), adj_mat)


def DIFFUSE_PROB_ITERATIVE(p1, sn, G, adj_mat, alpha=0.5):
    queue = [(sn, p1, set())]

    while len(queue) > 0:
        current_node, current_probability, vN = queue.pop(0)

        u_vN = G.keys() - vN
        UNsn = set(filter(lambda x: adj_mat.iloc[current_node, x] > 0, u_vN))

        if len(UNsn) == 0:
            if len(u_vN) == 0:
                continue
            to_add = current_probability / len(u_vN)
            for node in u_vN:
                G[node] += to_add
            continue

        sum_weights = 0
        for node in UNsn:
            sum_weights += adj_mat.iloc[current_node, node]
        for node in UNsn:
            inherited_prob = current_probability * (adj_mat.iloc[current_node, node] / sum_weights)
            G[node] += inherited_prob
            if inherited_prob * alpha > THRESHOLD_DIFF:
                G[node] -= inherited_prob * alpha
                queue.append((node, inherited_prob * alpha, vN.union({node})))
