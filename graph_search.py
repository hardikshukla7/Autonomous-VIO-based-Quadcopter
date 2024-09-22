import math
import numpy as np
from heapq import heappush, heappop
from .occupancy_map import OccupancyMap
import cProfile

def graph_search(world, resolution, margin, start, goal, astar):
    occ_map = OccupancyMap(world, resolution, margin)
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))
    cost = 0.5
    heuristic_cost = 1.5  # Adjusted heuristic cost for A*
    if astar == "True":
        queue = []
        heappush(queue, (0, start_index))
        visiting_cost = {start_index: 0}
        visited = set()
        # All neighbor offsets for A*
        neighbor_offsets = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
            (-1, -1, 0), (-1, 1, 0),
            (1, -1, 0), (1, 1, 0),
            (-1, 0, -1), (-1, 0, 1),
            (1, 0, -1), (1, 0, 1),
            (0, -1, -1), (0, -1, 1),
            (0, 1, -1), (0, 1, 1),
            (-1, -1, -1), (-1, -1, 1),
            (-1, 1, -1), (-1, 1, 1),
            (1, -1, -1), (1, -1, 1)
        ]

        occupied_indices = set(occ_map.metric_to_index(occ_map.index_to_metric_center(index))
                               for index in np.argwhere(occ_map.map))
        while queue:
            current_cost, current_node = heappop(queue)
            if current_node == goal_index:
                break
            for offset in neighbor_offsets:
                neighbor = (current_node[0] + offset[0], current_node[1] + offset[1], current_node[2] + offset[2])
                # Check if the neighbor is within the map bounds
                if not (0 <= neighbor[0] < occ_map.map.shape[0] and
                        0 <= neighbor[1] < occ_map.map.shape[1] and
                        0 <= neighbor[2] < occ_map.map.shape[2]) or neighbor in visited:
                    continue
                updated_cost = visiting_cost[current_node] + (10000 if neighbor in occupied_indices else cost)
                if neighbor not in visiting_cost or updated_cost < visiting_cost[neighbor]:
                    priority = updated_cost + heuristic_cost * math.sqrt(sum((neighbor[i] - goal_index[i]) ** 2 for i in range(3)))
                    heappush(queue, (priority, neighbor))
                    visiting_cost[neighbor] = updated_cost
                    visited.add(neighbor)
        nodes_expanded = len(visited)
        path = goal
        current = goal_index
        if current not in visited:
            path = None
        while current != start_index:
            path = np.vstack([occ_map.index_to_metric_center(current), path])
            current = visited[current]
        path = np.vstack([start, path])
        return path, nodes_expanded
    else:
        queue = []
        heappush(queue, (0, start_index))
        visited = {}
        # Manually defined neighbor offsets for Dijkstra's
        movements = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
            (-1, -1, 0), (-1, 1, 0),
            (1, -1, 0), (1, 1, 0),
            (-1, 0, -1), (-1, 0, 1),
            (1, 0, -1), (1, 0, 1),
            (0, -1, -1), (0, -1, 1),
            (0, 1, -1), (0, 1, 1)
        ]
        movements = movements[:18]
        movement_costs = {}
        for move in movements:
            movement_costs[move] = cost * math.sqrt(sum((move[i]) ** 2 for i in range(3)))
        while queue:
            current_cost, current_node = heappop(queue)
            if current_node == goal_index:
                break
            for move in movements:
                neighbor = (current_node[0] + move[0], current_node[1] + move[1], current_node[2] + move[2])
                # Check if the neighbor is within the map bounds
                if not (0 <= neighbor[0] < occ_map.map.shape[0] and
                        0 <= neighbor[1] < occ_map.map.shape[1] and
                        0 <= neighbor[2] < occ_map.map.shape[2]) or neighbor in visited:
                    continue
                updated_cost = current_cost + (math.inf if occ_map.is_occupied_index(neighbor) else movement_costs[move])
                if neighbor not in visited or updated_cost < visited[neighbor][0]:
                    heappush(queue, (updated_cost, neighbor))
                    visited[neighbor] = (updated_cost, current_node)
        nodes_expanded = len(visited)
        path = goal
        goal_path = goal_index
        if goal_index not in visited:
            path = None
        while goal_path != start_index:
            if goal_path not in visited:
                path = None
                break
            goal_path = visited[goal_path][1]
            path = np.vstack([occ_map.index_to_metric_center(goal_path), path])
        path = np.vstack([start, path])
        return np.array(path), nodes_expanded