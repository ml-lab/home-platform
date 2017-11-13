import json
import os

from multimodalmaze.core import House
from multimodalmaze.rendering import Panda3dRenderWorld
from multimodalmaze.semantic import SuncgSemanticWorld, MaterialColorTable, MaterialTable

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "tests/data")
TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "tests/data", "suncg")


def compute_all_relationships(house):
    """
    Computes relationships between all pairs of objects in the scene.
    """

    house_objects = [x for x in house.objects]
    room_objects = []

    for room in house.rooms:
        for obj in room.objects:
            room_objects.append(obj)

    directions = ['east', 'west', 'north', 'south']
    all_objects = house_objects + room_objects
    all_relationships = {x: [] for x in directions}

    for i, obj1 in enumerate(all_objects):
        related_objects = {x: [] for x in directions}
        for j, obj2 in enumerate(all_objects):
            if obj1 == obj2:
                continue
            diff = [obj1.location[k] - obj2.location[k] for k in [0, 1]]

            if diff[0] > 0:
                related_objects['east'].append(j)
            else:
                related_objects['west'].append(j)

            if diff[1] > 0:
                related_objects['north'].append(j)
            else:
                related_objects['south'].append(j)

        for d in directions:
            all_relationships[d].append(related_objects[d])

    return all_relationships


def set_object_properties(obj):
    """
    Sets category, color and material properties of an object
    """
    semantic_world = SuncgSemanticWorld(TEST_SUNCG_DATA_DIR)

    obj.shape = semantic_world.categoryMapping.getCoarseGrainedCategoryForModelId(obj.modelId)
    obj.size = "normal"
    obj.color = MaterialColorTable.getBasicColorsFromObject(obj, mode='basic')[0]
    obj.material = ' '.join(MaterialTable.getMaterialNameFromObject(obj))

    return obj


def generate_scene_metadata(house_id):
    """
    Generates a JSON file that contains a list of objects, and directional relationships b/w each pair of objects
    :param house_id:
    :return:
    """
    renderer = Panda3dRenderWorld(shadowing=False, showCeiling=False, mode='onscreen')
    house = House.loadFromJson(
        os.path.join(TEST_SUNCG_DATA_DIR, "house", house_id, "house.json"),
        TEST_SUNCG_DATA_DIR)
    renderer.addHouseToScene(house)

    keys = ['shape', 'size', 'color', 'material']

    metadata = {
        'info': {'split': 'train'},
        'scenes': [
            {
                "split": "train",
                "image_index": 0,
                "image_filename": "SUNCG_env_000000.png",
                "objects": [],
                "realationships": {}
            }
        ]
    }

    for idx, obj in enumerate(house.objects):
        obj = set_object_properties(obj)
        house.objects[idx] = obj
        metadata['scenes'][0]['objects'].append({k: v for k, v in vars(obj).iteritems() if k in keys})

    for room in house.rooms:
        for idx, obj in enumerate(room.objects):
            obj = set_object_properties(obj)
            room.objects[idx] = obj
            metadata['scenes'][0]['objects'].append({k: v for k, v in vars(obj).iteritems() if k in keys})

    metadata['scenes'][0]['relationships'] = compute_all_relationships(house)
    renderer.destroy()

    return metadata


if __name__ == '__main__':
    house_id_list = ['0004d52d1aeeb8ae6de39d6bd993e992']
    for house_id in house_id_list:
        metadata = generate_scene_metadata(house_id)
        with open(house_id + '.metadata.json', 'wb') as fp:
            json.dump(metadata, fp)
