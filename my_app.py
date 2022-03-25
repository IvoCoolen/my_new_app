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
side_rad = st.sidebar.radio("", ["Home", "Scores", "Info bridges", "Future condition calculator", "Fuzzy logic model"])


def make_networkx():
    G = nx.Graph()
    G.add_edge(1, 13, weight=1, name="road")
    G.add_edge(1, 19, weight=1, name="road")
    G.add_edge(1, 4, weight=0, name="bridge")
    G.add_edge(1, 21, weight=1, name="road")
    G.add_edge(1, 7, weight=1, name="road")
    G.add_edge(2, 13, weight=1, name="road")
    G.add_edge(2, 31, weight=1, name="road")
    G.add_edge(2, 28, weight=0, name="bridge")
    G.add_edge(3, 12, weight=0, name="bridge")
    G.add_edge(3, 7, weight=1, name="road")
    G.add_edge(3, 21, weight=1, name="road")
    G.add_edge(4, 5, weight=1, name="road")
    G.add_edge(4, 6, weight=1, name="road")
    G.add_edge(4, 17, weight=1, name="road")
    G.add_edge(5, 16, weight=1, name="road")
    G.add_edge(6, 11, weight=0, name="bridge")
    G.add_edge(6, 16, weight=1, name="road")
    G.add_edge(8, 13, weight=1, name="road")
    G.add_edge(8, 30, weight=1, name="road")
    G.add_edge(8, 31, weight=1, name="road")
    G.add_edge(9, 26, weight=0, name="bridge")
    G.add_edge(9, 25, weight=1, name="road")
    G.add_edge(9, 29, weight=1, name="road")
    G.add_edge(9, 27, weight=1, name="road")
    G.add_edge(10, 17, weight=0, name="bridge")
    G.add_edge(10, 16, weight=1, name="road")
    G.add_edge(11, 21, weight=1, name="road")
    G.add_edge(11, 12, weight=1, name="road")
    G.add_edge(11, 16, weight=0, name="bridge")
    G.add_edge(12, 24, weight=1, name="road")
    G.add_edge(14, 30, weight=0, name="bridge")
    G.add_edge(14, 20, weight=1, name="road")
    G.add_edge(15, 18, weight=0, name="bridge")
    G.add_edge(15, 32, weight=1, name="road")
    G.add_edge(15, 22, weight=1, name="road")
    G.add_edge(17, 19, weight=1, name="road")
    G.add_edge(19, 30, weight=0, name="bridge")
    G.add_edge(20, 22, weight=1, name="road")
    G.add_edge(23, 29, weight=1, name="road")
    G.add_edge(24, 25, weight=0, name="bridge")
    G.add_edge(25, 28, weight=1, name="road")
    G.add_edge(25, 27, weight=1, name="road")
    G.add_edge(26, 29, weight=1, name="road")
    G.add_edge(27, 31, weight=0, name="bridge")
    G.add_edge(28, 31, weight=1, name="road")
    G.add_edge(29, 32, weight=0, name="bridge")
    G.add_edge(30, 32, weight=1, name="road")

    # explicitly set positions
    pos = {0: ([0.35315709, 0.07477746]),
           1: ([0.20029783, 0.06232065]),
           2: ([ 0.04269488, -0.06656593]),
           3: ([ 0.31353677, -0.18207499]),
           4: ([0.60853845, 0.25382299]),
           5: ([0.75031606, 0.24013921]),
           6: ([0.76197827, 0.14639941]),
           7: ([ 0.23735862, -0.09027926]),
           8: ([-0.09847993,  0.05632589]),
           9: ([-0.45262878, -0.2806743 ]),
           10: ([0.66330427, 0.37380372]),
           11: ([ 0.68281404, -0.08764636]),
           12: ([ 0.52825229, -0.30698211]),
           13: ([0.08304266, 0.01548455]),
           14: ([-0.45853563,  0.28689702]),
           15: ([-0.58139359,  0.04143607]),
           16: ([1.        , 0.24879184]),
           17: ([0.39197481, 0.40251167]),
           18: ([-0.76844051,  0.03021643]),
           19: ([0.10763779, 0.26509003]),
           20: ([-0.61951835,  0.29871758]),
           21: ([0.5240987 , 0.01526191]),
           22: ([-0.60106666,  0.16596107]),
           23: ([-0.4257711 , -0.16187317]),
           24: ([-0.28127787, -0.50665595]),
           25: ([-0.31411275, -0.34304471]),
           26: ([-0.59480741, -0.34183487]),
           27: ([-0.27137732, -0.24990636]),
           28: ([-0.09172537, -0.29080525]),
           29: ([-0.58920114, -0.14373311]),
           30: ([-0.1846268 ,  0.16329371]),
           31: ([-0.17766676, -0.16416388]),
           32: ([-0.39204571,  0.04649739]),
           33: ([-0.34632686, -0.02150833])}

    options = {
        "font_size": 10,
        "node_size": 150,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 1,
        "width": 1,
    }

    bridge = [(u, v) for (u, v, d) in G.edges(data=True) if d["name"] == "bridge"]
    road = [(u, v) for (u, v, d) in G.edges(data=True) if d["name"] == "road"]
    return G, pos, options, bridge, road


def plot_networkx(bridge_id, x, bridge_list, road_list, options_, pos_, col, values):
    X = x
    X.add_edge(bridge_list[bridge_id][0], bridge_list[bridge_id][1], weight=1, name="selected")
    selected = [(u, v) for (u, v, d) in X.edges(data=True) if d["name"] == "selected"]
    nx.draw_networkx(X, pos_, **options_)
    nx.draw_networkx_edges(
        X, pos_, edgelist=bridge_list, width=3, alpha=0.6, edge_color="r")
    nx.draw_networkx_edges(
        X, pos_, edgelist=road_list, width=2, alpha=1, edge_color="black")
    nx.draw_networkx_edges(
        X, pos_, edgelist=selected, width=4, alpha=0.7, edge_color="blue")
    # Set margins for the axes so that nodes aren't clipped
    # fig, ax = plt.subplots()
    ax = plt.gca()
    ax.margins(0.20)
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
    col.pyplot(plt)


def make_table(bridge_id, file, col):
    row = file.loc[file["Bridge ID"] == bridge_id]
    # row = row.astype(int)
    row = row.astype(str)
    row_transpose = row.transpose()
    row_transpose.set_axis(["Values"], axis="columns", inplace=True)
    col.table(row_transpose)
    age = int(row["Age"].tolist()[0])
    deck = int(row["Deck condition"].tolist()[0])
    sup = int(row["Superstructure condition"].tolist()[0])
    sub = int(row["Substructure condition"].tolist()[0])
    struc_eval = int(row["Structural evaluation"].tolist()[0])
    return age, deck, sup, sub, struc_eval


def calc_expected_condition(cluster, condition_term, begin_age, age_range, curr_cond, col, num_year_fut):
    save_data = []
    save_data.append(curr_cond)
    age = begin_age+2
    array_2 = cluster[condition_term][age_range]
    save_data.append(array_2[curr_cond - 1, 0] * 1 +
                     array_2[curr_cond - 1, 1] * 2 +
                     array_2[curr_cond - 1, 2] * 3 +
                     array_2[curr_cond - 1, 3] * 4 +
                     array_2[curr_cond - 1, 4] * 5 +
                     array_2[curr_cond - 1, 5] * 6)
    while age <= 180:
        ages = ["30", "60", "90", "120", "150", "180"]
        age_range = ages[math.ceil(age / 30) - 1]
        if age == 0:
            age_range = "30"
        array1 = cluster[condition_term][age_range]
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


def shortest_path(paths_options):
    short_path = []
    shortest_path_length = 1000
    all_paths = {}
    for path1 in paths_options:
        total_dist = 0
        if len(path1) == 2:
            continue
        for j in range(1, len(path1)):
            total_dist += calc_weight_length(path1[j-1], path1[j])
            all_paths[path1] = total_dist
        if total_dist == shortest_path_length:
            short_path.append(path1)
        elif total_dist < shortest_path_length:
            short_path = [path1]
            shortest_path_length = total_dist
    return short_path, shortest_path_length, all_paths


G, pos, options, bridge, road = make_networkx()
if side_rad == "Home":
    blank, title_col, blank2 = st.columns([2, 3.5, 2])
    title_col.title("Bridge prioritization tool")

if side_rad == "Scores":
    blank, title_col, blank2 = st.columns([2, 3.5, 2])
    title_col.title("Scores")

    bridges_shortest_paths = {}
    for i in bridge:
        bridges_shortest_paths[i] = {}
        paths = []
        for pat in nx.algorithms.all_simple_paths(G, source=i[0], target=i[1], cutoff=10):
            paths.append(pat)
        if len(paths) > 1:
            path, path_length = shortest_path(paths)
            bridges_shortest_paths[i]["path"] = path
            bridges_shortest_paths[i]["length"] = path_length

if side_rad == "Info bridges":
    blank, title_col, blank2 = st.columns([2, 3.5, 2])
    col1, col2 = st.columns([5, 5])
    title_col.title("Information bridges")
    option = col1.selectbox(
        "Select bridge to see information:",
        ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'))
    option_age, option_deck, option_sup, option_sub, option_struc = make_table(int(option), case_file, col2)
    condition_values = [option_age, option_deck, option_sup, option_sub, option_struc]
    plot_networkx(int(option), G, bridge, road, options, pos, col1, condition_values)

if side_rad == "Future condition calculator":
    blank, title_col, blank2 = st.columns([2, 3.5, 2])
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
    if choice_curr_cond != 6:
        for i in range(choice_curr_cond+1, math.ceil(data_list[-1])+1):
            index = data_list.index(i)
            col_table.text(f"Number of years till condition {i} is reached: " + str(index))

if side_rad == "Fuzzy logic model":
    blank, title_col, blank2 = st.columns([2, 3.5, 2])
    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
    col_wide, col_choice = st.columns([8, 2])
    title_col.title("Fuzzy logic model simulator")
    number1 = col1.number_input("Give duration of activity:", min_value=2, max_value=48)
    number2 = col2.number_input("Give detour length:", min_value=0, max_value=20)
    number3 = col3.number_input("Give decrease vehicle speed/volume:", min_value=0, max_value=90)
    number4 = col4.number_input("Give distance to parc:", min_value=0, max_value=10000)
    number5 = col5.number_input("Give distance to emergency:", min_value=0, max_value=30)
    effect, FS, rules = fuzzy_logic(number1, number2, number3, number4, number5)
    col1.pyplot(FS.plot_variable("DU"))
    col2.pyplot(FS.plot_variable("DL"))
    col3.pyplot(FS.plot_variable("VS"))
    col4.pyplot(FS.plot_variable("DP"))
    col5.pyplot(FS.plot_variable("DE"))
    firing_strength = FS.get_firing_strengths()
    # col1.text(FS.get_fuzzy_sets("DU"))
    col_wide.header(("Total effect on the surrounding is: " + str(effect)))
    choice1 = col_choice.selectbox("Select a bridge ID:",
                             ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'))

    with st.expander("See explanation"):
        for i in range(len(rules)):
            text = "R" + str(1 + 1) + ": " + rules[1] + " --- Firing strength is: " + str(firing_strength[1])
            col_wide.text(text)
