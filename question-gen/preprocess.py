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
    print all_objects
    all_relationships = {x: {} for x in directions}

    for obj1 in all_objects:
        related_objects = {x: [] for x in directions}
        for obj2 in all_objects:
            if obj1 == obj2: continue
            diff = [obj1.location[k] - obj2.location[k] for k in [0, 1]]

            if diff[0] > 0:
                related_objects['east'].append(obj2.instanceId)
            else:
                related_objects['west'].append(obj2.instanceId)

            if diff[1] > 0:
                related_objects['north'].append(obj2.instanceId)
            else:
                related_objects['south'].append(obj2.instanceId)

        for d in directions:
            all_relationships[d][obj1.instanceId] = related_objects[d]

    return all_relationships


def set_object_properties(obj):
    """
    Sets category, color and material properties of an object
    """
    semantic_world = SuncgSemanticWorld(TEST_SUNCG_DATA_DIR)

    obj.category = semantic_world.categoryMapping.getCoarseGrainedCategoryForModelId(obj.modelId)
    obj.color = ' '.join(MaterialColorTable.getBasicColorsFromObject(obj, mode='basic'))
    obj.material = ' '.join(MaterialTable.getMaterialNameFromObject(obj))

    return obj


def generate_scene_metadata(house_id):
    """
    Generates a JSON file that contains
    :param house_id:
    :return:
    """
    renderer = Panda3dRenderWorld(shadowing=False, showCeiling=False, mode='onscreen')
    house = House.loadFromJson(
        os.path.join(TEST_SUNCG_DATA_DIR, "house", house_id, "house.json"),
        TEST_SUNCG_DATA_DIR)
    renderer.addHouseToScene(house)

    keys = ['category', 'color', 'material']
    metadata = {'objects': [], 'relationships': {}}

    for idx, obj in enumerate(house.objects):
        obj = set_object_properties(obj)
        house.objects[idx] = obj
        metadata['objects'].append({k: v for k, v in vars(obj).iteritems() if k in keys})

    for room in house.rooms:
        for idx, obj in enumerate(room.objects):
            obj = set_object_properties(obj)
            room.objects[idx] = obj
            metadata['objects'].append({k: v for k, v in vars(obj).iteritems() if k in keys})

    metadata['relationships'] = compute_all_relationships(house)
    renderer.destroy()

    return metadata


if __name__ == '__main__':
    house_id_list = ['0004d52d1aeeb8ae6de39d6bd993e992']
    for house_id in house_id_list:
        metadata = generate_scene_metadata(house_id)
        with open(house_id + '.metadata.json', 'wb') as fp:
            json.dump(metadata, fp)
