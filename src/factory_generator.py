import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import json
import uuid
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt

import factorio_blueprints
from factorio_blueprints import EncodedBlob
from copy import deepcopy

import sys


ITEM_SIZES = {
    'logistic-chest-requester': 1,
    'logistic-chest-passive-provider': 1,
    'assembling-machine-1': 3,
    'electric-furnace': 3,
    'substation': 2,
    'roboport': 4,
    'long-handed-inserter': 1,
    'inserter': 1,
}

DATA_FILE = '../data/vanilla-1.0.0.json'

with open(DATA_FILE, 'r') as f:
    data = json.loads(f.read())

def pick_machine(category):
    # TODO: Support other entity types
    # TODO: Take into account the rate at which the item is needed
    
    for entity_category in ['assembling-machine', 'furnace']:
        for machine_name in data[entity_category]:
            if category in data[entity_category][machine_name]['crafting_categories']:
                return machine_name
    return 'logistic-chest-requester'


def pick_recipe(requested_item):
    # TODO: Check all recipes which result in this item
    # TODO: Take into account the rate at which the item is needed
    recipe_name = requested_item
    return recipe_name

def generate_children(required_item, parent_node):
    print(required_item)
    node_id = uuid.uuid4()
    
    recipe_name = pick_recipe(required_item)
    try:
        recipe = data['recipes'][recipe_name]
        machine = pick_machine(recipe['category'])
        node = {
            "entity_number": node_id,
            "name": machine,
            "recipe": recipe_name
        }
        incredients = recipe['ingredients']
    except KeyError:
        node = {
            "entity_number": node_id,
            "name": 'logistic-chest-requester',
            'request_filters': [
                {'index': 1, 'name': required_item, 'count': 100}
            ],
        }
        incredients = []

    nodes = [node]
    edges = [(node_id, parent_node)]

    for ingredient in incredients:
        new_nodes, new_edges = generate_children(ingredient['name'], node_id)
        nodes += new_nodes
        edges += new_edges

    return nodes, edges

def generate_graph(item):
    output_node_id = '0'

    output_node = {
        "entity_number": output_node_id,
        "name": 'logistic-chest-passive-provider',
    }

    nodes, edges = generate_children(item, output_node_id)
    nodes.append(output_node)
    return nodes, edges
    
def get_min_distance(pos):
    min_distance = None
    for a in pos:
        for b in pos:
            if a == b:
                continue
            distance = sum((pos[a]-pos[b])**2)**0.5
            if min_distance is None or distance < min_distance:
                min_distance = distance
    return min_distance


def add_node_locations(nodes, edges):
    G = nx.DiGraph()
    G.add_edges_from(edges)

    labels = {
        node['entity_number']: node['name'] for node in nodes
    }

    required_distance = 4

    for scale in range(1, 1000):
        pos = nx.drawing.layout.spring_layout(G, scale=scale, k=3, iterations=500)

        # nx.draw(G, pos, cmap = plt.get_cmap('jet'))
        # nx.draw_networkx_labels(G, pos, labels)
        # plt.show()
        if get_min_distance(pos) > required_distance:
            break

    # Adjust locations to start from origin

    min_x = min([a[0] for a in pos.values()]) - 5
    min_y = min([a[1] for a in pos.values()]) - 5
    for key in pos.keys():
        pos[key] = pos[key] - (min_x, min_y)
        pos[key] = pos[key].astype(int)
        
    for node in nodes:
        node['position'] = {
            'x': int(pos[node['entity_number']][0]),
            'y': int(pos[node['entity_number']][1])
        }
    return nodes


def get_grid(width, height):
    grid = np.zeros([width, height])
    cell_ids = [[i*height + j for j in range(height)] for i in range(width)]
    cell_ids = np.array(cell_ids) 
    return grid, cell_ids


def get_entity_coordinates(entity):
    x = entity['position']['x']
    y = entity['position']['y']
    size = ITEM_SIZES[entity['name']]
    # adjustment = int((size-1)/2)
    adjustment = 0

    for i in range(size):
        for j in range(size):
            yield (x + i - adjustment, y + j - adjustment)


def add_entity(grid, entity):
    for coordinates in get_entity_coordinates(entity):
        try:
            grid[coordinates] = 1
        except:
            print("Warning: entity at {} has location {} out of bounds".format(entity['position'], coordinates))
    return grid


def generate_possible_locations(target, maximum_range, grid_size):
    ideal_x = target[0] + maximum_range
    ideal_y = target[1] + maximum_range
    for i in range(2*maximum_range):
        for j in range(2*maximum_range):
            if i >= grid_size[0]:
                continue
            if j >= grid_size[1]:
                continue

            yield ideal_x - i, ideal_y - j


def can_entity_be_placed(entity, grid):
    for location in get_entity_coordinates(entity):
        try:
            if grid[location]:
                # print(entity['name'], "can not be placed on location", location)
                return False
        except IndexError:
            # print("Location", location, "out of bounds")
            return False
    return True


def get_covering_entity(entity_name, possible_locations, grid):
    entity = {
        'name': entity_name,
        'position': {}
    }

    for entity_location in possible_locations:
        entity['position']['x'] = int(entity_location[0])
        entity['position']['y'] = int(entity_location[1])

        if can_entity_be_placed(entity, grid):
            grid = add_entity(grid, entity)
            return entity
            
    print("Warning: Not able to add coverage for", entity_name)
    assert False
                    
                    
def cover_grid(grid, entity_name = 'substation', maximum_range = 6):
    coverage = np.zeros(grid.shape)
    area_entities = []
    
    while not coverage.all():
        missing_coverage = np.where(coverage == 0)
        target = (missing_coverage[0][0], missing_coverage[1][0])
        
        possible_locations = generate_possible_locations(target, maximum_range, grid.shape)
        try:
            entity = get_covering_entity(entity_name, possible_locations, grid)
        except AssertionError:
            print("Problem for location", target)
            possible_locations = generate_possible_locations(target, maximum_range, grid.shape)
            print([a for a in possible_locations])
            coverage[target] += 1
            continue

        # Update coverage
        for i in range(entity['position']['x']-maximum_range, entity['position']['x']+maximum_range):
            for j in range(entity['position']['y']-maximum_range, entity['position']['y']+maximum_range):
                if i < 0:
                    continue
                if j < 0:
                    continue
                if i >= coverage.shape[0]:
                    continue
                if j >= coverage.shape[1]:
                    continue

                coverage[i, j] += 1

        # print("Added entity", entity_location)
        # print(sum(sum(coverage)))
        area_entities.append(entity)
        plt.imshow(coverage)
        
    return grid, area_entities


def get_inserter_connection(location, direction, arm_length):
    # Based on reverse engineered blueprint
    # 6: moves items right
    # 8 (or no direction given): moves items down
    # 2: moves items left
    # 4: moves items up
    i, j = location

    if direction == 6:
        source = i-arm_length, j
        destination = i+arm_length, j
    if direction == 2:
        source = i+arm_length, j
        destination = i-arm_length, j
    if direction == 8:
        source = i, j-arm_length
        destination = i, j+arm_length
    if direction == 4:
        source = i, j+arm_length
        destination = i, j-arm_length

    return source, destination


def can_inserter_be_added(blocked_tiles, entities, grid):
    for tile in blocked_tiles:
        if tile in [coord for entity in entities for coord in get_entity_coordinates(entity)]:
            # Origin or destination tile is allowed
            continue

        if grid[tile]:
            return False

    return True


def add_inserters(G, grid, cell_ids, entities, arm_length = 1):
    for i in range(arm_length, grid.shape[0]-arm_length):
        for j in range(arm_length, grid.shape[1]-arm_length):
            if grid[(i, j)]:
                continue

            for direction in [2, 4, 6, 8]:

                source, destination = get_inserter_connection((i, j), direction, arm_length)
                blocked_tiles = [
                    (i, j),
                    source,
                    destination
                ]
                
                if not can_inserter_be_added(blocked_tiles, entities, grid):
                    continue

                edge = (
                    cell_ids[source],
                    cell_ids[destination]
                )

                name = 'inserter'
                if arm_length == 2:
                    name = 'long-handed-inserter'

                entity_info = {
                    'name': name,
                    'position': {
                        'x': i,
                        'y': j,
                    },
                    'direction': direction,
                }

                G.add_edges_from([edge], label=entity_info, blocked_tiles=blocked_tiles)


def get_connection_graph(grid, cell_ids, entities):
    G = nx.DiGraph()

    add_inserters(G, grid, cell_ids, entities, arm_length = 1)
    add_inserters(G, grid, cell_ids, entities, arm_length = 2)

    # Add connections inside of the entities
    for entity in entities:
        
        entity_cell_id = get_entity_grid_id(entity, cell_ids)
        
        for coordinates in get_entity_coordinates(entity):
            edges = [
                (
                    cell_ids[coordinates[0]][coordinates[1]],
                    entity_cell_id,
                ),
                (
                    entity_cell_id,
                    cell_ids[coordinates[0]][coordinates[1]],
                ),
            ]
            G.add_edges_from(edges, label='dummy')
    
    if False:
        edges = []
        for i in range(grid.shape[0]-1):
            for j in range(grid.shape[1]-1):
                edges.append((cell_ids[i][j], cell_ids[i+1][j]))
                edges.append((cell_ids[i][j], cell_ids[i][j+1]))
                G.add_edges_from(edges, label='dummy')
                edges = []

    return G


def get_entity_grid_id(entity, cell_ids):
    x = entity['position']['x']
    y = entity['position']['y']
    return cell_ids[x][y]


def add_connection(grid, cell_ids, origin, destination):        
    G = get_connection_graph(grid, cell_ids, [origin, destination])
    
    origin_id = get_entity_grid_id(origin, cell_ids)
    destination_id = get_entity_grid_id(destination, cell_ids)
        
    sp = nx.shortest_path(G, origin_id, destination_id)
    pathGraph = nx.path_graph(sp)

    new_entities = []
    for ea in pathGraph.edges():
        edge = G.edges[ea[0], ea[1]]
        if edge['label'] == 'dummy':
            # print("Dummy connection")
            continue
        new_entity = G.edges[ea[0], ea[1]]['label']
        # print(new_entity, " connection")
        new_entities.append(new_entity)
        for tile in edge['blocked_tiles']:
            # assert grid[tile] == 0
            grid[tile] += 1
        # grid[new_entity['position']['x'], new_entity['position']['y']] = 1

    return grid, new_entities


def add_connections(grid, cell_ids, nodes, edges):
    node_dict = {
        node['entity_number']: node
        for node in nodes
    }

    connection_entities = []
    for i, edge in enumerate(edges):
        grid, new_entities = add_connection(grid, cell_ids, node_dict[edge[0]], node_dict[edge[1]])
        connection_entities = connection_entities + new_entities
        
    return connection_entities, grid


def get_tiles(grid):
    
    tiles = []
    for i in range(0, grid.shape[0]):
        for j in (0, grid.shape[1]-1):
            entity = {
                'name': 'hazard-concrete-left',
                'position': {
                    'x': i,
                    'y': j
                },
            }
            tiles.append(entity)

    for i in (0, grid.shape[0]-1):
        for j in range(0, grid.shape[1]):
            tile = {
                'name': 'hazard-concrete-left',
                'position': {
                    'x': i,
                    'y': j
                },
            }
            tiles.append(tile)
    return tiles


def import_string(entities, tiles):
    entities = deepcopy(entities)
    for i in range(len(entities)):
        entities[i]['entity_number'] = i
        
        # Adjust position with item size
        item_name = entities[i]['name']
        size = ITEM_SIZES[item_name]
        entities[i]['position']['x'] += size/2
        entities[i]['position']['y'] += size/2

    blueprint = {
        "blueprint": {
            'snap-to-grid': {'x': 1, 'y': 1},
            'absolute-snapping': True,
            "icons": [
                {
                    "signal": {
                        "type": "item",
                        "name": "concrete"
                    },
                    "index": 1
                }
            ],
            'entities': entities,
            'tiles': tiles,
        },
        "version": 281474976710656,
        "version_byte": "0"
    }

    import_string = EncodedBlob.from_json_string(json.dumps(blueprint)).to_exchange_string()
    print(import_string)


def main(item='automation-science-pack'):
    nodes, edges = generate_graph(item)
    nodes = add_node_locations(nodes, edges)
    grid, cell_ids = get_grid(
        max([node['position']['x'] for node in nodes])+5,
        max([node['position']['y'] for node in nodes])+5,
    )
    for node in nodes:
        grid = add_entity(grid, node)
        
    connection_entities, grid = add_connections(grid, cell_ids, nodes, edges)
    grid, substations = cover_grid(grid, entity_name = 'substation', maximum_range = 6)
    grid, roboports = cover_grid(grid, entity_name = 'roboport', maximum_range = 15)
    
    entities = nodes + connection_entities + substations + roboports
    tiles = get_tiles(grid)
    import_string(entities, tiles)
    
if __name__ == '__main__':
    main(sys.argv[1])
