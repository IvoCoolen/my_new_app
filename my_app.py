import streamlit as st
import time
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
# import folium
# from branca.element import Figure
import pickle
import math
from simpful import *

st.set_option('deprecation.showPyplotGlobalUse', False)

# Read file
case_file = pd.read_csv("case_study.txt", header=0, encoding='latin1')
a_file = open("cluster_1_array.pkl", "rb")
cluster1_array = pickle.load(a_file, encoding='bytes')
a_file.close()
a_file = open("cluster_2_array.pkl", "rb")
cluster2_array = pickle.load(a_file, encoding='bytes')
a_file.close()
a_file = open("cluster_3_array.pkl", "rb")
cluster3_array = pickle.load(a_file, encoding='bytes')
a_file.close()
a_file = open("cluster_4_array.pkl", "rb")
cluster4_array = pickle.load(a_file, encoding='bytes')
a_file.close()

cluster_files = [cluster1_array, cluster2_array, cluster3_array, cluster4_array]
# Set configurations
st.set_page_config(layout="wide", page_title='Bridge prioritization tool', page_icon=':bridge_at_night')

# Make sidebar
st.sidebar.header("Choose your page")
side_rad = st.sidebar.radio("", ["Home", "Prioritization model", "Info bridges", "Future condition calculator", "Fuzzy logic model"])


def make_networkx():
    G1 = nx.Graph()
    G1.add_edge(1, 13, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(1, 19, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(1, 4, weight=1-0.039, weight2=0.039, name="bridge")
    G1.add_edge(1, 21, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(1, 7, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(2, 13, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(2, 31, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(2, 28, weight=1-0.269, weight2=0.269, name="bridge")
    G1.add_edge(3, 12, weight=1-0.218, weight2=0.218, name="bridge")
    G1.add_edge(3, 7, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(3, 21, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(4, 5, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(4, 6, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(4, 17, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(5, 16, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(6, 11, weight=1-0.01, weight2=0.01, name="bridge")
    G1.add_edge(6, 16, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(8, 13, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(8, 30, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(8, 31, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(9, 26, weight=1-0.512, weight2=0.512, name="bridge")
    G1.add_edge(9, 25, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(9, 29, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(9, 27, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(10, 17, weight=1-0.296, weight2=0.296, name="bridge")
    G1.add_edge(10, 16, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(11, 21, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(11, 12, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(11, 16, weight=1-0.775, weight2=0.775, name="bridge")
    G1.add_edge(12, 24, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(14, 30, weight=1-0.865, weight2=0.865, name="bridge")
    G1.add_edge(14, 20, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(15, 18, weight=1-0.658, weight2=0.658, name="bridge")
    G1.add_edge(15, 32, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(15, 22, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(17, 19, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(19, 30, weight=1-0.641, weight2=0.641, name="bridge")
    G1.add_edge(20, 22, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(23, 29, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(24, 25, weight=1-0.320, weight2=0.320, name="bridge")
    G1.add_edge(25, 28, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(25, 27, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(26, 29, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(27, 31, weight=1.0, weight2=0.5, name="bridge")
    G1.add_edge(28, 31, weight=0.5, weight2=0.5, name="road")
    G1.add_edge(29, 32, weight=1-0.660, weight2=0.660, name="bridge")
    G1.add_edge(30, 32, weight=0.5, weight2=0.5, name="road")

    # explicitly set positions
    pos1 = {0: ([0.35315709, 0.07477746]),
           1: ([0.20029783, 0.06232065]),
           2: ([0.04269488, -0.06656593]),
           3: ([0.31353677, -0.18207499]),
           4: ([0.60853845, 0.25382299]),
           5: ([0.75031606, 0.24013921]),
           6: ([0.76197827, 0.14639941]),
           7: ([0.23735862, -0.09027926]),
           8: ([-0.09847993,  0.05632589]),
           9: ([-0.45262878, -0.2806743 ]),
           10: ([0.66330427, 0.37380372]),
           11: ([0.68281404, -0.08764636]),
           12: ([0.52825229, -0.30698211]),
           13: ([0.08304266, 0.01548455]),
           14: ([-0.45853563,  0.28689702]),
           15: ([-0.58139359,  0.04143607]),
           16: ([1, 0.24879184]),
           17: ([0.39197481, 0.40251167]),
           18: ([-0.76844051,  0.03021643]),
           19: ([0.10763779, 0.26509003]),
           20: ([-0.61951835,  0.29871758]),
           21: ([0.5240987, 0.01526191]),
           22: ([-0.60106666,  0.16596107]),
           23: ([-0.4257711, -0.16187317]),
           24: ([-0.28127787, -0.50665595]),
           25: ([-0.31411275, -0.34304471]),
           26: ([-0.59480741, -0.34183487]),
           27: ([-0.27137732, -0.24990636]),
           28: ([-0.09172537, -0.29080525]),
           29: ([-0.58920114, -0.14373311]),
           30: ([-0.1846268,  0.16329371]),
           31: ([-0.17766676, -0.16416388]),
           32: ([-0.39204571,  0.04649739]),
           33: ([-0.34632686, -0.02150833])}

    options1 = {
        "font_size": 10,
        "node_size": 150,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 1,
        "width": 1,
    }
    emergency1 = [5, 21, 27, 32]
    bridge1 = [(u, v) for (u, v, d) in G1.edges(data=True) if d["name"] == "bridge"]
    road1 = [(u, v) for (u, v, d) in G1.edges(data=True) if d["name"] == "road"]
    return G1, pos1, options1, bridge1, road1, emergency1


def plot_networkx(bridge_id, X, bridge_list, road_list, options_, pos_, col, values, emergency, vis_path=False,
                  shortest_path_list=None):
    X.add_edge(bridge_list[bridge_id][0], bridge_list[bridge_id][1], weight=1, name="selected")
    selected = [(u, v) for (u, v, d) in X.edges(data=True) if d["name"] == "selected"]
    nx.draw_networkx(X, pos_, **options_)
    if vis_path:
        nx.draw_networkx_edges(
            X, pos_, edgelist=shortest_path_list, width=4, alpha=0.7, edge_color="green")
    nx.draw_networkx_edges(
        X, pos_, edgelist=bridge_list, width=3, alpha=0.6, edge_color="r")
    nx.draw_networkx_edges(
        X, pos_, edgelist=road_list, width=2, alpha=1, edge_color="black")
    nx.draw_networkx_edges(
        X, pos_, edgelist=selected, width=4, alpha=0.7, edge_color="blue")
    for fac in emergency:
        pos_[fac][0] = pos_[fac][0] - 0.03
        pos_[fac][1] = pos_[fac][1] + 0.04
    nx.draw_networkx_nodes(X, pos_, nodelist=emergency, node_color="Red", alpha=0.7, node_shape="x", node_size=30)
    # Set margins for the axes so that nodes aren't clipped
    # fig, ax = plt.subplots()
    ax = plt.gca()
    ax.margins(0.20)
    if not vis_path:
        color = ["green", "green", "orange", "orange", "red", "red"]
        color2 = ["red", "red", "red", "red", "orange", "orange", "orange", "green", "green", "green"]
        text1 = "Age: " + str(values[0])
        text2 = "Structural evaluation: " + str(values[4])
        text3 = "Deck condition: " + str(values[1])
        text4 = "Superstructure condition: " + str(values[2])
        text5 = "Substructure condition: " + str(values[3])
        plt.figtext(0.14, 0.18, text1, size=9, color="black")
        plt.figtext(0.25, 0.18, text2, size=9, color=color2[values[4]])
        plt.figtext(0.14, 0.13, text3, size=9, color=color[values[1]-1])
        plt.figtext(0.34, 0.13, text4, size=9, color=color[values[2] - 1])
        plt.figtext(0.63, 0.13, text5, size=9, color=color[values[3] - 1])
    plt.figtext(0.7, 0.82, "X", size=7, color="red")
    plt.figtext(0.713, 0.82, ": Emergency facility", size=7)
    col.pyplot(plt)


def make_table(bridge_id, file, col, cond=True):
    row = file.loc[file["Bridge ID"] == bridge_id]
    # row = row.astype(int)
    row = row.astype(str)
    row_transpose = row.transpose()
    row_transpose.set_axis(["Values"], axis="columns", inplace=True)
    col.table(row_transpose)
    if cond:
        age = int(row["Age"].tolist()[0])
        deck = int(row["Deck condition"].tolist()[0])
        sup = int(row["Superstructure condition"].tolist()[0])
        sub = int(row["Substructure condition"].tolist()[0])
        struc_eval = int(row["Structural evaluation"].tolist()[0])
        return age, deck, sup, sub, struc_eval


def calc_expected_condition(cluster, condition_term, begin_age, age_range1, curr_cond, col, num_year_fut):
    save_data = [curr_cond]
    age = begin_age+2
    array_2 = cluster[condition_term][age_range1]
    save_data.append(array_2[curr_cond - 1, 0] * 1 +
                     array_2[curr_cond - 1, 1] * 2 +
                     array_2[curr_cond - 1, 2] * 3 +
                     array_2[curr_cond - 1, 3] * 4 +
                     array_2[curr_cond - 1, 4] * 5 +
                     array_2[curr_cond - 1, 5] * 6)
    while age <= 180:
        ages1 = ["30", "60", "90", "120", "150", "180"]
        age_range1 = ages1[math.ceil(age / 30) - 1]
        if age == 0:
            age_range1 = "30"
        array1 = cluster[condition_term][age_range1]
        array_2 = np.matmul(array_2, array1)
        save_data.append(array_2[curr_cond - 1, 0] * 1 +
                         array_2[curr_cond - 1, 1] * 2 +
                         array_2[curr_cond - 1, 2] * 3 +
                         array_2[curr_cond - 1, 3] * 4 +
                         array_2[curr_cond - 1, 4] * 5 +
                         array_2[curr_cond - 1, 5] * 6)
        age += 1
    y = range(begin_age, begin_age+len(save_data))
    if num_year_fut != 0:
        plt.scatter(begin_age+num_year_fut, save_data[num_year_fut], edgecolors='red')
        plt.figtext(0.15, 0.8, "Expected condition: " + str(round(save_data[num_year_fut],3)))
    plt.plot(y, save_data)
    plt.title("Expected condition score of bridge at certain age")
    plt.ylim(1, 6.1)
    plt.xlim(begin_age-1, 185)
    plt.xlabel("Age")
    plt.ylabel("Condition score")
    col.pyplot(plt)
    return save_data[num_year_fut], save_data


def fuzzy_logic(duration, detour_length, vehicle_speed, distance_parc, distance_emergency):
    FS = FuzzySystem()

    # Duration (hours)
    DU_1 = FuzzySet(function=Triangular_MF(a=2, b=2, c=48), term="short")
    DU_2 = FuzzySet(function=Triangular_MF(a=2, b=15, c=48), term="normal")
    DU_3 = FuzzySet(function=Triangular_MF(a=2, b=48, c=48), term="long")
    FS.add_linguistic_variable("DU", LinguisticVariable([DU_1, DU_2, DU_3], concept="Duration",
                                                        universe_of_discourse=[2, 48]))

    # detour length (KM)
    DL_1 = FuzzySet(function=Triangular_MF(a=1, b=1, c=4), term="short")
    DL_2 = FuzzySet(function=Triangular_MF(a=1, b=5, c=10), term="medium")
    DL_3 = FuzzySet(function=Triangular_MF(a=5, b=20, c=20), term="long")
    DL_4 = FuzzySet(function=Triangular_MF(a=0, b=0, c=0.1), term="no")
    FS.add_linguistic_variable("DL", LinguisticVariable([DL_1, DL_2, DL_3, DL_4], concept="Detour length",
                                                        universe_of_discourse=[0, 20]))

    # Decrease vehicle speed (%KM)
    VS_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=100), term="small")
    VS_2 = FuzzySet(function=Triangular_MF(a=0, b=90, c=90), term="big")
    FS.add_linguistic_variable("VS", LinguisticVariable([VS_1, VS_2], concept="Vehicle speed",
                                                        universe_of_discourse=[0, 90]))

    # Distance to parc (meters)
    DP_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=1000), term="short")
    DP_2 = FuzzySet(function=Triangular_MF(a=0, b=1000, c=10000), term="medium")
    DP_3 = FuzzySet(function=Triangular_MF(a=1000, b=10000, c=10000), term="long")
    FS.add_linguistic_variable("DP", LinguisticVariable([DP_1, DP_2, DP_3], concept="Distance to parc",
                                                        universe_of_discourse=[0, 10000]))

    # Distance to emergency facility (KM)
    DE_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=5), term="short")
    DE_2 = FuzzySet(function=Triangular_MF(a=3, b=5, c=30), term="medium")
    DE_3 = FuzzySet(function=Triangular_MF(a=5, b=30, c=30), term="long")
    FS.add_linguistic_variable("DE", LinguisticVariable([DE_1, DE_2, DE_3], concept="Distance to emergency facility",
                                                        universe_of_discourse=[0, 30]))

    # Effect on surrounding (score)
    ES_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=30), term="very_low")
    ES_2 = FuzzySet(function=Triangular_MF(a=20, b=30, c=50), term="low")
    ES_3 = FuzzySet(function=Triangular_MF(a=30, b=50, c=70), term="medium")
    ES_4 = FuzzySet(function=Triangular_MF(a=50, b=70, c=80), term="high")
    ES_5 = FuzzySet(function=Triangular_MF(a=70, b=100, c=100), term="very_high")
    FS.add_linguistic_variable("ES", LinguisticVariable([ES_1, ES_2, ES_3, ES_4, ES_5], concept="Effect on surrounding",
                                                        universe_of_discourse=[0, 100]))
    # fuzzy rules
    R1 = "IF (DU IS short) THEN (ES IS very_low)"
    R2 = "IF (DU IS normal) THEN (ES IS high)"
    R3 = "IF (DU IS long) THEN (ES IS very_high)"
    R4 = "IF (DL IS long) AND (DE IS short) THEN (ES IS very_high)"
    R5 = "IF (DL IS long) THEN (ES IS high)"
    R6 = "IF (DL IS medium) THEN (ES IS medium)"
    R7 = "IF (DL IS short) THEN (ES IS very_low)"
    R8 = "IF (VS IS big) AND (DL IS no) THEN (ES IS very_high)"
    R9 = "IF (VS IS big) AND ((DL IS no) AND (DU IS long)) THEN (ES IS very_high)"
    R10 = "IF (VS IS big) AND ((DL IS no) AND (DU IS short)) THEN (ES IS medium)"
    R11 = "IF (VS IS big) AND ((DL IS no) AND ((DP IS long) OR (DE IS long))) THEN (ES IS medium)"
    R12 = "IF (VS IS big) AND ((DL IS no) AND ((DP IS medium) OR (DE IS medium))) THEN (ES IS high)"
    R13 = "IF (DP IS short) THEN (ES IS high)"
    R14 = "IF (DP IS medium) OR (DE IS short) THEN (ES IS medium)"
    R15 = "IF (DP IS long) AND (DE IS medium) THEN (ES IS low)"
    rules = [R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15]
    FS.add_rules(rules)

    # set antecedents
    FS.set_variable("DU", duration)  # [2, 48]
    FS.set_variable("DL", detour_length)  # [0, 20]
    FS.set_variable("VS", vehicle_speed)  # [0, 90]
    FS.set_variable("DP", distance_parc)  # [0, 10000]
    FS.set_variable("DE", distance_emergency)  # [0, 30]

    return round(FS.Mamdani_inference(["ES"], verbose=False)["ES"], 3), FS, rules


def calc_weight_length(a, b):
    dist = np.sqrt((pos[b][0] - pos[a][0])**2+(pos[b][1] - pos[a][1])**2)
    return dist


def shortest_path(paths_options, emergency_option=False):
    short_path = 0
    shortest_path_length = 1000
    all_paths = {}
    bridge_length = 0
    for path1 in paths_options:
        total_dist = 0
        if not emergency_option:
            if len(path1) == 2:
                bridge_length = calc_weight_length(path1[0], path1[1])
                continue
        for j in range(1, len(path1)):
            total_dist += calc_weight_length(path1[j-1], path1[j])
            all_paths[str(path1)] = total_dist
        if total_dist < shortest_path_length:
            short_path = path1
            shortest_path_length = total_dist
    return short_path, shortest_path_length*20, all_paths, bridge_length*20


def dist_emergency(emergency, bridge_loc, x):
    dist = {}
    shortest_dist = []
    for loc in bridge_loc:
        dist[str(loc)] = []
        for fac in emergency:
            routes = []
            for route in nx.algorithms.all_simple_paths(x, source=fac, target=loc, cutoff=10):
                routes.append(route)
            _, distance, _, _ = shortest_path(routes, emergency_option=True)
            dist[str(loc)].append(distance)
        shortest_dist.append(min(dist[str(loc)]))
    return sum(shortest_dist)/2


def calc_weight(a, b, bridge_id, weight, adt_list):
    if weight == "Length of bridge":
        dist = np.sqrt((pos[b][0] - pos[a][0])**2+(pos[b][1] - pos[a][1])**2)
        return dist
    elif weight == "Preferable bridge":
        dist = 0.3
        return dist
    elif weight == "Preferable road":
        dist = 0.7
        return dist
    elif weight == "Average daily traffic":
        adt = (adt_list[bridge_id]-(min(adt_list)-100))/((max(adt_list)+100)-(min(adt_list)-100))
        return adt


def min_max_norm(x, list1):
    return (x-min(list1))/(max(list1)-min(list1))


def calc_network_crit(X, bridge_locs, weight):
    adt_weight = case_file["Average daily traffic"].tolist()
    X.add_edge(1, 4, weight=calc_weight(1, 4, 0, weight, adt_weight),
               weight2=1-calc_weight(1, 4, 0, weight, adt_weight), name="bridge")
    X.add_edge(2, 28, weight=calc_weight(2, 28, 1, weight, adt_weight),
               weight2=1-calc_weight(2, 28, 1, weight, adt_weight), name="bridge")
    X.add_edge(3, 12, weight=calc_weight(3, 12, 2, weight, adt_weight),
               weight2=1-calc_weight(3, 12, 2, weight, adt_weight), name="bridge")
    X.add_edge(6, 11, weight=calc_weight(6, 11, 3, weight, adt_weight),
               weight2=1-calc_weight(6, 11, 3, weight, adt_weight), name="bridge")
    X.add_edge(9, 26, weight=calc_weight(9, 26, 4, weight, adt_weight),
               weight2=1-calc_weight(9, 26, 4, weight, adt_weight), name="bridge")
    X.add_edge(10, 17, weight=calc_weight(10, 17, 5, weight, adt_weight),
               weight2=1-calc_weight(10, 17, 5, weight, adt_weight), name="bridge")
    X.add_edge(11, 16, weight=calc_weight(11, 16, 6, weight, adt_weight),
               weight2=1-calc_weight(11, 16, 6, weight, adt_weight), name="bridge")
    X.add_edge(14, 30, weight=calc_weight(14, 30, 7, weight, adt_weight),
               weight2=1-calc_weight(14, 30, 7, weight, adt_weight), name="bridge")
    X.add_edge(15, 18, weight=calc_weight(15, 18, 8, weight, adt_weight),
               weight2=1-calc_weight(15, 18, 8, weight, adt_weight), name="bridge")
    X.add_edge(19, 30, weight=calc_weight(19, 30, 9, weight, adt_weight),
               weight2=1-calc_weight(19, 30, 9, weight, adt_weight), name="bridge")
    X.add_edge(24, 25, weight=calc_weight(24, 25, 10, weight, adt_weight),
               weight2=1-calc_weight(24, 25, 10, weight, adt_weight), name="bridge")
    X.add_edge(27, 31, weight=calc_weight(27, 31, 11, weight, adt_weight),
               weight2=1-calc_weight(27, 31, 11, weight, adt_weight), name="bridge")
    X.add_edge(29, 32, weight=calc_weight(29, 32, 12, weight, adt_weight),
               weight2=1-calc_weight(29, 32, 12, weight, adt_weight), name="bridge")
    edge_betweenness = nx.algorithms.centrality.edge_betweenness_centrality(X, k=None, normalized=False,
                                                                            weight="weight")
    edge_current_flow = nx.algorithms.centrality.edge_current_flow_betweenness_centrality(X, normalized=False,
                                                                                          weight="weight2")
    edge_load_centrality = nx.algorithms.centrality.edge_load_centrality(X)
    edge_b = []
    edge_c = []
    edge_l = []
    scores = []
    for b in bridge_locs:
        new_loc = (b[1], b[0])
        edge_b.append(edge_betweenness[b])
        edge_l.append(edge_load_centrality[b])
        if b in edge_current_flow:
            edge_c.append(edge_current_flow[b])
        else:
            edge_c.append(edge_current_flow[new_loc])
    for b in range(len(bridge_locs)):
        betweenness = min_max_norm(edge_b[b], edge_b)
        current = min_max_norm(edge_c[b], edge_c)
        load = min_max_norm(edge_l[b], edge_l)
        scores.append(round(betweenness+current+load, 2))
    return scores


def calc_technical_cond(deck, superstr, substr, struc):
    tech_score = []
    if deck >= 5 or superstr >= 5 or substr >= 5:
        tech_score.append(45)
        if deck >= 5 and superstr >= 5 or deck >= 5 and substr >= 5 or substr >= 5 and superstr >= 5:
            tech_score.append(47.5)
            if deck >= 5 and superstr >= 5 and substr >= 5:
                tech_score.append(50)
    elif deck == 4 or superstr == 4 or substr == 4:
        tech_score.append(30)
        if deck == 4 and superstr == 4 or deck == 4 and substr == 4 or substr == 4 and superstr == 4:
            tech_score.append(35)
            if deck == 4 and superstr == 4 and substr == 4:
                tech_score.append(40)
    elif deck == 3 or superstr == 3 or substr == 3:
        tech_score.append(15)
        if deck == 3 and superstr == 3 or deck == 3 and substr == 3 or substr == 3 and superstr == 3:
            tech_score.append(20)
            if deck == 3 and superstr == 3 and substr == 3:
                tech_score.append(25)
    elif deck == 2 or superstr == 2 or substr == 2:
        tech_score.append(5)
        if deck == 2 and superstr == 2 or deck == 2 and substr == 2 or substr == 2 and superstr == 2:
            tech_score.append(10)
            if deck == 2 and superstr == 2 and substr == 2:
                tech_score.append(15)
    else:
        tech_score.append(0)
    if struc <= 2:
        str_score = 50
    elif struc == 3:
        str_score = 40
    elif struc == 4:
        str_score = 30
    elif struc == 5:
        str_score = 20
    elif struc == 6:
        str_score = 15
    elif struc == 7:
        str_score = 10
    elif struc == 8:
        str_score = 5
    else:
        str_score = 0
    return max(tech_score)+str_score


G, pos, options, bridge, road, emergency_cities = make_networkx()
bridges_info = {"Bridge ID": [], "Bridge location": [], "Duration": [], "Shortest path": [], "Length shortest path": [],
                "Decrease vehicle speed": [], "Distance to parc": [], "Distance to emergency": []}
bridges_all_paths = {}

for i in range(len(bridge)):
    paths = []
    for pat in nx.algorithms.all_simple_paths(G, source=bridge[i][0], target=bridge[i][1], cutoff=10):
        paths.append(pat)
    path, path_length, paths_list, length_bridge = shortest_path(paths)
    print(path)
    mean_dist = dist_emergency(emergency_cities, bridge[i], G)
    bridges_all_paths[i] = paths_list
    bridges_info["Bridge ID"].append(i)
    bridges_info["Bridge location"].append(bridge[i])
    sum_cond = sum(case_file.loc[case_file["Bridge ID"] == i][["Deck condition", "Superstructure condition",
                                                               "Substructure condition"]].values.tolist()[0]) * 2.6
    bridges_info["Duration"].append(round(sum_cond, 3))
    if path_length <= 20:
        bridges_info["Shortest path"].append(path)
        bridges_info["Length shortest path"].append(round(path_length, 3))
        bridges_info["Decrease vehicle speed"].append(0)
    else:
        bridges_info["Shortest path"].append([])
        bridges_info["Length shortest path"].append(0)
        vs = max(case_file.loc[case_file["Bridge ID"] == i][["Deck condition", "Superstructure condition",
                                                             "Substructure condition"]].values.tolist()[0]) * 15
        bridges_info["Decrease vehicle speed"].append(vs)
    bridges_info["Distance to parc"].append(round(length_bridge/2, 3))
    bridges_info["Distance to emergency"].append(round(mean_dist, 3))

bridge_info_df = pd.DataFrame(bridges_info)

if side_rad == "Home":
    blank, title_col, blank2 = st.columns([2, 5, 2])
    title_col.title("Bridge prioritization tool")
    col1, col2 = st.columns([4, 4])
    col2.subheader("Introduction")
    col2.write("Welcome to the bridge prioritization tool")
    col2.write("This is an application which prioritizes the maintenance needs of individual bridges. "
               "This tool combines the technical state of the bridge, importance in a network and sustainable "
               "effect on the surrounding to get a total score, which indicates the importance of maintenance "
               "on that specific bridge.")
    col2.write("This project is conducted as a master thesis project of the master: Operations management and logistics"
               " at the Technical university of Eindhoven")
    col1.subheader("Pages")
    col1.write("1. prioritization model: This model will calculate the total 'importance' score.")
    col1.write("2. Info bridges: Here all information about the bridges can be found")
    col1.write("3. Future condition calculator: This is a tool to get future information of the technical state "
               "of the bridges")
    col1.write("4. Fuzzy logic model: This model calculates the (sustainable) effect on the surrounding environment")
    col3, col4 = st.columns([4, 4])
    col3.image("tue1.png")
    col4.image("witteveenbos1.png")

if side_rad == "Prioritization model":
    blank, title_col, blank2 = st.columns([3, 5, 2])
    title_col.title("Ranking model")
    col_df, _ = st.columns([100, 5])
    col1_sl, col2_sl, col3_sl = st.columns([3, 3, 3])
    col1, col2 = st.columns([3, 10])
    dict_total_score = {"Technical score": [], "Effect surrounding": []}
    for i in range(len(bridge)):
        df_bridge_id = bridge_info_df.loc[bridge_info_df["Bridge ID"] == i]
        df_case_id = case_file.loc[case_file["Bridge ID"] == i]
        du = df_bridge_id["Duration"].tolist()[0]
        dl = df_bridge_id["Length shortest path"].tolist()[0]
        vs = df_bridge_id["Decrease vehicle speed"].tolist()[0]
        dp = df_bridge_id["Distance to parc"].tolist()[0]
        de = df_bridge_id["Distance to emergency"].tolist()[0]
        d_c = df_case_id["Deck condition"].tolist()[0]
        sp_c = df_case_id["Superstructure condition"].tolist()[0]
        sb_c = df_case_id["Substructure condition"].tolist()[0]
        str_e = df_case_id["Structural evaluation"].tolist()[0]
        effect, _, _ = fuzzy_logic(du, dl, vs, dp, de)
        tech_cond = calc_technical_cond(d_c, sp_c, sb_c, str_e)
        dict_total_score["Effect surrounding"].append(round(effect, 2))
        dict_total_score["Technical score"].append(round(tech_cond, 2))
    option = col1.radio("Choose a network weight selection option", ["Length of bridge", "Preferable bridge", "Preferable road",
                                                           "Average daily traffic"])
    dict_total_score["Network criticality"] = calc_network_crit(G, bridge, option)
    dict_total_score2 = {"Technical score": [], "Effect surrounding": [], "Network criticality": []}
    for i in range(len(bridge)):
        dict_total_score2["Technical score"].append(round(min_max_norm(dict_total_score["Technical score"][i],
                                                                       dict_total_score["Technical score"]), 2))
        dict_total_score2["Effect surrounding"].append(round(min_max_norm(dict_total_score["Effect surrounding"][i],
                                                                          dict_total_score["Effect surrounding"]), 2))
        dict_total_score2["Network criticality"].append(round(min_max_norm(dict_total_score["Network criticality"][i],
                                                                           dict_total_score["Network criticality"]), 2))
    col_df.subheader("Input variables")
    col_df.dataframe(pd.DataFrame(dict_total_score).T)
    expander = col_df.expander("See normalized scores")
    df_total_score = pd.DataFrame(dict_total_score2)
    expander.dataframe(df_total_score.T)
    col_df.write("")
    col_df.write("")
    col_df.subheader("Weights to calculate total score")
    slider1 = col1_sl.slider("Weight Technical score", value=1, min_value=1, max_value=3)
    slider2 = col2_sl.slider("Weight Effect on surrounding", value=1, min_value=1, max_value=3)
    slider3 = col3_sl.slider("Weight Network criticality", value=1, min_value=1, max_value=3)
    df_total_score["Total score"] = df_total_score["Technical score"]*slider1 + \
                                    df_total_score["Effect surrounding"]*slider2 + \
                                    df_total_score["Network criticality"]*slider3
    df_total_score.sort_values("Total score", inplace=True, ascending=False)
    col2.table(df_total_score["Total score"])

if side_rad == "Info bridges":
    blank, title_col, blank2 = st.columns([2, 3.5, 2])
    col1, col2 = st.columns([5, 5])
    title_col.title("Information bridges")
    option = col1.selectbox(
        "Select bridge to see information:",
        ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'))
    col2.subheader("Bridge characteristics")
    case_file1 = case_file.drop(["Age", "Deck condition", "Superstructure condition",
                                 "Substructure condition", "Age (reconstructed)", "Average daily traffic",
                                 "Structural evaluation"], inplace=False, axis=1)
    make_table(int(option), case_file1, col2, cond=False)
    col2.subheader("Technical state information")
    case_file2 = case_file[["Bridge ID", "Age", "Age (reconstructed)", "Average daily traffic", "Deck condition",
                            "Superstructure condition", "Substructure condition",
                            "Structural evaluation"]]
    option_age, option_deck, option_sup, option_sub, option_struc = make_table(int(option), case_file2, col2)
    condition_values = [option_age, option_deck, option_sup, option_sub, option_struc]
    col1.subheader("Map")
    plot_networkx(int(option), G, bridge, road, options, pos, col1, condition_values, emergency_cities)
    col1.subheader("Location information")
    make_table(int(option), bridge_info_df, col1, cond=False)

if side_rad == "Future condition calculator":
    blank, title_col, blank2 = st.columns([2, 5, 2])
    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
    col_choice, col_table, col_calc = st.columns([2, 3, 5])
    title_col.title("Future condition calculator")
    option1 = col1.selectbox(
        "Select condition term:",
        ('Deck', 'Superstructure', 'Substructure'))
    option2 = col2.selectbox(
        "Select cluster:",
        ('1', '2', '3', '4'))

    if option1 == "Deck":
        cond_term = "deck"
        cond_term2 = "Deck condition"
        radio_options = case_file.loc[case_file["Cluster (deck)"] == int(option2)]["Bridge ID"].tolist()
        # col_array.text(cluster_files[int(option2)-1]["deck"][age_range])
    elif option1 == "Superstructure":
        cond_term = "super"
        cond_term2 = "Superstructure condition"
        radio_options = case_file.loc[case_file["Cluster (superstructure)"] == int(option2)]["Bridge ID"].tolist()
        # col_array.text(cluster_files[int(option2) - 1]["super"][age_range])
    else:
        cond_term = "sub"
        cond_term2 = "Substructure condition"
        radio_options = case_file.loc[case_file["Cluster (substructure)"] == int(option2)]["Bridge ID"].tolist()
        # col_array.text(cluster_files[int(option2) - 1]["sub"][age_range])

    choice_radio = col_choice.selectbox("Select bridge in this cluster:", radio_options)
    choice_age = case_file.loc[case_file["Bridge ID"] == choice_radio]["Age"].tolist()[0]
    choice_curr_cond = case_file.loc[case_file["Bridge ID"] == choice_radio][cond_term2].tolist()[0]
    number1 = col3.number_input('Give a age of the bridge:', value=choice_age, min_value=0, max_value=180)
    number2 = col4.number_input('Give current condition value:', value=choice_curr_cond, min_value=1, max_value=6)
    number3 = col5.number_input('Insert prediction interval (# years into future):', min_value=0, max_value=30)
    ages = ["30", "60", "90", "120", "150", "180"]
    age_range = ages[math.ceil(number1/30)-1]
    if number1 == 0:
        age_range = "30"

    expected_cond, data_list = calc_expected_condition(cluster_files[int(option2)-1], cond_term, number1, age_range, number2, col_calc, number3)
    future_data_dict = {"Expectedfuture condition": expected_cond}
    # col_table.text([data_list,choice_curr_cond+1, round(data_list[-1])])
    data_list = [math.floor(elem) for elem in data_list]
    if number3 in data_list:
        for i in range(choice_curr_cond+1, math.ceil(data_list[-1])+1):
            index = data_list.index(i)
            col_table.text(f"Number of years till condition {i} is reached: " + str(index))

if side_rad == "Fuzzy logic model":
    blank, title_col, blank2 = st.columns([2, 5, 2])
    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
    col_wide, col_choice = st.columns([8, 2])
    title_col.title("Fuzzy logic model simulator")
    choice1 = col_choice.selectbox("Select a bridge ID:",
                                   ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'))
    c1 = bridge_info_df.loc[bridge_info_df["Bridge ID"] == int(choice1)]["Duration"].tolist()[0]
    c2 = bridge_info_df.loc[bridge_info_df["Bridge ID"] == int(choice1)]["Length shortest path"].tolist()[0]
    c3 = bridge_info_df.loc[bridge_info_df["Bridge ID"] == int(choice1)]["Decrease vehicle speed"].tolist()[0]
    c4 = bridge_info_df.loc[bridge_info_df["Bridge ID"] == int(choice1)]["Distance to parc"].tolist()[0]
    c5 = bridge_info_df.loc[bridge_info_df["Bridge ID"] == int(choice1)]["Distance to emergency"].tolist()[0]
    number1 = col1.number_input("Give duration of activity:", value=c1, min_value=2.0, max_value=48.0)
    number2 = col2.number_input("Give detour length:", value=c2, min_value=0.0, max_value=20.0)
    number3 = col3.number_input("Give decrease vehicle speed/volume:", value=c3, min_value=0, max_value=90)
    number4 = col4.number_input("Give distance to parc:", value=c4, min_value=0.0, max_value=10000.0)
    number5 = col5.number_input("Give distance to emergency:", value=c5, min_value=0.0, max_value=30.0)
    effect, FS, rules = fuzzy_logic(number1, number2, number3, number4, number5)
    col1.pyplot(FS.plot_variable("DU"))
    col2.pyplot(FS.plot_variable("DL"))
    col3.pyplot(FS.plot_variable("VS"))
    col4.pyplot(FS.plot_variable("DP"))
    col5.pyplot(FS.plot_variable("DE"))
    firing_strength = FS.get_firing_strengths()
    # col1.text(FS.get_fuzzy_sets("DU"))
    col_wide.header(("Total effect on the surrounding is: " + str(effect)))
    expander = col_wide.expander("See rules with firing strength")
    for i in range(len(rules)):
        expander.write("R" + str(i + 1) + ": " + rules[i])
        expander.write("Firing strength is: " + str(firing_strength[i]))
        expander.write("---------------------------------------------------------")

