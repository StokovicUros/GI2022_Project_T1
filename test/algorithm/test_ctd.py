"""

    Used for testing of CTD (Connect the dots) algorithm

"""
import random
import timeit
import os.path

import pandas as pd
import copy
import scipy
import unittest

from memory_profiler import memory_usage
from src.util.draw import *
from src.algorithm.ctd import *
from src.util.console import *
from src.util.path import *

# Packages for R code
import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro

# Connect to R function
R_files_location = get_project_root() + '/src/R'
r = robjects.r
r['source'](f'{R_files_location}/graph.diffuseP1.r')
r['source'](f'{R_files_location}/graph.connectToExt.r')
R_function = robjects.globalenv['graph.diffuseP1']

# Define constants
ALLOWED_DIFFERENCE = 1e-15
STARTING_PROBABILITY = 0.5

NUMBER_OF_NODES = [10, 50, 500, 2000]
PROBABILITY = [0.15, 0.25, 0.5, 0.75, 0.9]

NUMBER_OF_EXECUTIONS = 100

data_folder = get_project_root() + "/test/data/graph"


def check_equal(self, a, b):
    self.assertEqual(len(a), len(b), "Results do not have same number of nodes")
    for node in a:
        self.assertLess(abs(a[node] - b[node][0]), ALLOWED_DIFFERENCE,
                        f"Difference between results for node {node} is greater than {ALLOWED_DIFFERENCE}")


def generate_test_graphs():
    write_header_message("Generating test data:")
    destination_folder = get_project_root() + "/test/data/graph"
    if not os.path.isdir(destination_folder):
        os.mkdir(destination_folder)

    for node_count in NUMBER_OF_NODES:
        for edge_probability in PROBABILITY:
            if os.path.isfile(f"{destination_folder}/graph_{node_count}_nodes_{edge_probability}_probability.csv"):
                write_warning_message(
                    f"Test graph \'{relative_path(destination_folder)}/graph_{node_count}_nodes_{edge_probability}_probability.csv\' already exists")
                continue

            graph = nx.random_geometric_graph(node_count, edge_probability, seed=0)
            for (_, _, w) in graph.edges(data=True):
                w['weight'] = random.randint(1, 20)

            df = nx.to_pandas_adjacency(graph, dtype=int)
            df.to_csv(f"{destination_folder}/graph_{node_count}_nodes_{edge_probability}_probability.csv", index=False)
    write_success_message("SUCCESS - Test data generated")


class TestCTD(unittest.TestCase):

    def test_recursive_with_5_node_graph(self):
        try:
            df = pd.read_csv(f"{data_folder}/graph_5_nodes.csv",
                             dtype=int)
        except FileNotFoundError:
            print(
                f"Test not found: \'{relative_path(data_folder)}/graph_5_nodes.csv\'")
            print("--------------------------------------------")
            return

        df.columns = df.columns.astype(int)

        sn_init = 0
        probabilities = {}
        for i in range(5):
            probabilities[i] = 0

        probabilities_test = copy.deepcopy(probabilities)

        print(f"Running test for graph \'{relative_path(data_folder)}/graph_5_nodes.csv\'")

        # Call CTD function
        DIFFUSE_PROB_RECURSIVE(STARTING_PROBABILITY, sn_init, probabilities, set(), df)

        # Calculate execution time
        execution_time = timeit.timeit(
            lambda: DIFFUSE_PROB_RECURSIVE(STARTING_PROBABILITY, sn_init, probabilities_test, set(), df),
            number=NUMBER_OF_EXECUTIONS)
        print(f'Average execution time: {execution_time / NUMBER_OF_EXECUTIONS}s')

        # Calculate memory usage
        print(
            f"Memory usage: "
            f"{memory_usage((DIFFUSE_PROB_RECURSIVE, (STARTING_PROBABILITY, sn_init, probabilities_test, set(), df)), max_usage=True, interval=.0001)}MB")

        # Call R function
        with localconverter(ro.default_converter + pandas2ri.converter):  # Convert data
            df_r = ro.conversion.py2rpy(df)
        # No need to convert response from R list to Python list
        response = R_function(STARTING_PROBABILITY, sn_init + 1, THRESHOLD_DIFF, df_r)

        # Check if we got correct result
        check_equal(self, probabilities, response)
        write_success_message("\nPASS!")

    def test_iterative_with_5_node_graph(self):
        try:
            df = pd.read_csv(f"{data_folder}/graph_5_nodes.csv",
                             dtype=int)
        except FileNotFoundError:
            print(
                f"Test not found: \'{relative_path(data_folder)}/graph_5_nodes.csv\'")
            print("--------------------------------------------")
            return

        df.columns = df.columns.astype(int)

        sn_init = 0
        probabilities = {}
        for i in range(5):
            probabilities[i] = 0

        probabilities_test = copy.deepcopy(probabilities)

        print(f"Running test for graph \'{relative_path(data_folder)}/graph_5_nodes.csv\'")

        # Call CTD function
        DIFFUSE_PROB_ITERATIVE(STARTING_PROBABILITY, sn_init, probabilities, df)

        # Calculate execution time
        execution_time = timeit.timeit(
            lambda: DIFFUSE_PROB_ITERATIVE(STARTING_PROBABILITY, sn_init, probabilities_test, df),
            number=NUMBER_OF_EXECUTIONS)
        print(f'Average execution time: {execution_time / NUMBER_OF_EXECUTIONS}s')

        # Calculate memory usage
        print(
            f"Memory usage: "
            f"{memory_usage((DIFFUSE_PROB_ITERATIVE, (STARTING_PROBABILITY, sn_init, probabilities_test, df)), max_usage=True, interval=.0001)}MB")

        # Call R function
        with localconverter(ro.default_converter + pandas2ri.converter):  # Convert data
            df_r = ro.conversion.py2rpy(df)
        # No need to convert response from R list to Python list
        response = R_function(STARTING_PROBABILITY, sn_init + 1, THRESHOLD_DIFF, df_r)

        # Check if we got correct result
        check_equal(self, probabilities, response)
        write_success_message("\nPASS!")

    def test_recursive_with_10_node_graph(self):
        try:
            df = pd.read_csv(f"{data_folder}/graph_10_nodes.csv",
                             dtype=int)
        except FileNotFoundError:
            print(
                f"Test not found: \'{relative_path(data_folder)}/graph_10_nodes.csv\'")
            print("--------------------------------------------")
            return

        df.columns = df.columns.astype(int)

        sn_init = 0
        probabilities = {}
        for i in range(10):
            probabilities[i] = 0

        probabilities_test = copy.deepcopy(probabilities)

        print(f"Running test for graph \'{relative_path(data_folder)}/graph_10_nodes.csv\'")

        # Call CTD function
        DIFFUSE_PROB_RECURSIVE(STARTING_PROBABILITY, sn_init, probabilities, set(), df)

        # Calculate execution time
        execution_time = timeit.timeit(
            lambda: DIFFUSE_PROB_RECURSIVE(STARTING_PROBABILITY, sn_init, probabilities_test, set(), df),
            number=NUMBER_OF_EXECUTIONS)
        print(f'Average execution time: {execution_time / NUMBER_OF_EXECUTIONS}s')

        # Calculate memory usage
        print(
            f"Memory usage: "
            f"{memory_usage((DIFFUSE_PROB_RECURSIVE, (STARTING_PROBABILITY, sn_init, probabilities_test, set(), df)), max_usage=True, interval=.0001)}MB")

        # Call R function
        with localconverter(ro.default_converter + pandas2ri.converter):  # Convert data
            df_r = ro.conversion.py2rpy(df)
        # No need to convert response from R list to Python list
        response = R_function(STARTING_PROBABILITY, sn_init + 1, THRESHOLD_DIFF, df_r)

        # Check if we got correct result
        check_equal(self, probabilities, response)
        write_success_message("\nPASS!")

    def test_iterative_with_10_node_graph(self):
        try:
            df = pd.read_csv(f"{data_folder}/graph_10_nodes.csv",
                             dtype=int)
        except FileNotFoundError:
            print(
                f"Test not found: \'{relative_path(data_folder)}/graph_10_nodes.csv\'")
            print("--------------------------------------------")
            return

        df.columns = df.columns.astype(int)

        sn_init = 0
        probabilities = {}
        for i in range(10):
            probabilities[i] = 0

        probabilities_test = copy.deepcopy(probabilities)

        print(f"Running test for graph \'{relative_path(data_folder)}/graph_10_nodes.csv\'")

        # Call CTD function
        DIFFUSE_PROB_ITERATIVE(STARTING_PROBABILITY, sn_init, probabilities, df)

        # Calculate execution time
        execution_time = timeit.timeit(
            lambda: DIFFUSE_PROB_ITERATIVE(STARTING_PROBABILITY, sn_init, probabilities_test, df),
            number=NUMBER_OF_EXECUTIONS)
        print(f'Average execution time: {execution_time / NUMBER_OF_EXECUTIONS}s')

        # Calculate memory usage
        print(
            f"Memory usage: "
            f"{memory_usage((DIFFUSE_PROB_ITERATIVE, (STARTING_PROBABILITY, sn_init, probabilities_test, df)), max_usage=True, interval=.0001)}MB")

        # Call R function
        with localconverter(ro.default_converter + pandas2ri.converter):  # Convert data
            df_r = ro.conversion.py2rpy(df)
        # No need to convert response from R list to Python list
        response = R_function(STARTING_PROBABILITY, sn_init + 1, THRESHOLD_DIFF, df_r)

        # Check if we got correct result
        check_equal(self, probabilities, response)
        write_success_message("\nPASS!")

    def test_recursive_and_iterative_on_same_graphs(self, generate_test_data=True):
        if generate_test_data:
            generate_test_graphs()

        for node_count in NUMBER_OF_NODES:
            for edge_probability in PROBABILITY:
                with self.subTest(f"{node_count}_nodes_{edge_probability}_prob"):
                    # Read graph
                    try:
                        df = pd.read_csv(f"{data_folder}/graph_{node_count}_nodes_{edge_probability}_probability.csv",
                                         dtype=int)
                    except FileNotFoundError:
                        write_warning_message(
                            f"Test not found: \'{relative_path(data_folder)}/graph_{node_count}_nodes_{edge_probability}_probability.csv\'")
                        write_normal_message("--------------------------------------------")
                        continue

                    df.columns = df.columns.astype(int)

                    # Determine starting node
                    sn_init = random.randrange(0, node_count)
                    probabilities = {}
                    for i in range(node_count):
                        probabilities[i] = 0

                    probabilities_test = copy.deepcopy(probabilities)

                    write_normal_message(
                        f"Running test for graph \'{relative_path(data_folder)}/graph_{node_count}_nodes_{edge_probability}_probability.csv\'")

                    # Recursive
                    write_normal_message("Recursive algorithm ==============================")

                    # Call CTD function
                    DIFFUSE_PROB_RECURSIVE(STARTING_PROBABILITY, sn_init, probabilities, set(), df)
                    recursive_response = probabilities

                    # Calculate execution time
                    execution_time = timeit.timeit(
                        lambda: DIFFUSE_PROB_RECURSIVE(STARTING_PROBABILITY, sn_init, probabilities_test, set(), df),
                        number=NUMBER_OF_EXECUTIONS)
                    print(f'Average execution time: {execution_time / NUMBER_OF_EXECUTIONS}s')

                    # Draw
                    if node_count == NUMBER_OF_NODES[0]:
                        graph = nx.from_pandas_adjacency(df)
                        draw_graph(graph, probabilities, f"recursive_{node_count}_nodes_{edge_probability}_probability")

                    # Calculate memory usage
                    write_normal_message(
                        f"Memory usage: "
                        f"{memory_usage((DIFFUSE_PROB_RECURSIVE, (STARTING_PROBABILITY, sn_init, probabilities_test, set(), df)), max_usage=True, interval=.0001)}MB")

                    for i in range(node_count):
                        probabilities[i] = 0

                    # Iterative
                    write_normal_message("Iterative algorithm ==============================")

                    # Call CTD function
                    DIFFUSE_PROB_ITERATIVE(STARTING_PROBABILITY, sn_init, probabilities, df)
                    iterative_response = probabilities

                    # Calculate execution time
                    execution_time = timeit.timeit(
                        lambda: DIFFUSE_PROB_ITERATIVE(STARTING_PROBABILITY, sn_init, probabilities_test, df),
                        number=NUMBER_OF_EXECUTIONS)
                    print(f'Average execution time: {execution_time / NUMBER_OF_EXECUTIONS}s')

                    # Draw
                    if node_count == NUMBER_OF_NODES[0]:
                        draw_graph(graph, probabilities, f"iterative_{node_count}_nodes_{edge_probability}_probability")

                    # Calculate memory usage
                    write_normal_message(
                        f"Memory usage: "
                        f"{memory_usage((DIFFUSE_PROB_ITERATIVE, (STARTING_PROBABILITY, sn_init, probabilities_test, df)), max_usage=True, interval=.0001)}MB")

                    # Call R function
                    with localconverter(ro.default_converter + pandas2ri.converter):  # Convert data
                        df_r = ro.conversion.py2rpy(df)
                    # No need to convert response from R list to Python list
                    response = R_function(STARTING_PROBABILITY, sn_init + 1, THRESHOLD_DIFF, df_r)

                    check_equal(self, recursive_response, response)
                    check_equal(self, iterative_response, response)
                    write_success_message("\nPASS!")
                    write_normal_message("--------------------------------------------")


if __name__ == "__main__":
    unittest.main()
