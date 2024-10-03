import itertools
import numpy as np
import pandas as pd
from get_mistral_embedding import load_data
from generate_scenario import save_data

def euclidean_distance(point1, point2):
    """Compute the Euclidean distance between two points."""
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def similarity(similarity_metrics, point1, point2):
    return 1/2*(similarity_metrics[point1][point2] + similarity_metrics[point2][point1])

def find_initial_point(similarity_metrics):
    diag = np.diag(np.diag(similarity_metrics))
    similarity_metrics = similarity_metrics - diag + np.eye(len(similarity_metrics))
    sum_0, sum_1 = np.sum(similarity_metrics, axis=0), np.sum(similarity_metrics, axis=1)
    point_index = np.argmin(sum_0 + sum_1)

    return point_index

def max_min_diversity(metrics, k):
    """Greedy approximation algorithm for Maximum Minimum Diversity Problem."""
    selected_points = [find_initial_point(similarity_metrics=metrics)]
    points = [i for i in range(len(metrics))]
    remaining_points = set(points)
    remaining_points -= set(selected_points)

    while len(selected_points) < k:
        if not remaining_points:
            break

        # Find the point that maximizes the minimum distance to the selected points
        # next_point = max(remaining_points, key=lambda p: min(euclidean_distance(p, selected) for selected in selected_points))

        # Find the point that minimizes the maximum similarity to the selected points
        next_point = min(remaining_points,
                         key=lambda p: max(similarity(metrics, p, selected) for selected in selected_points))

        # Add the selected point to the subset
        selected_points.append(next_point)

        # Remove the selected point # and its symmetric counterparts
        remaining_points -= {next_point} #| {(next_point[1], next_point[0])}

    return selected_points


def compute_metric():
    return


def load_csv(filename):
    data = pd.read_csv(filename)
    embeddings = data['ada_embedding']
    all_embeddings = []
    for embedding in embeddings:
        all_embeddings.append(np.array(eval(embedding)))
    return all_embeddings

if __name__ == "__main__":

    # gpt version
    metrics = np.load('./data/gpt_similarity.npy')
    selected_points = max_min_diversity(metrics, k=2000)

    scenarios = load_data('gpt_generated_scenario_post.txt')
    selected_scenarios = [scenarios[selected] for selected in selected_points]
    save_data(selected_scenarios, './gpt_final.txt')
    selected_points_ = np.array(selected_points)
    np.save('./data/gpt_final_index.npy', selected_points_)
    print('done!')

