# Copyright (c) 2017, IGLU consortium
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
#  - Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation 
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors 
#    may be used to endorse or promote products derived from this software 
#    without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
# POSSIBILITY OF SUCH DAMAGE.

import csv
import logging
import numpy as np

logger = logging.getLogger(__name__)
      
class ModelInformation(object):

    header = 'id,front,nmaterials,minPoint,maxPoint,aligned.dims,index,variantIds'

    def __init__(self, filename):
        self.model_info = {}
        
        self._parseFromCSV(filename)
        
    def _parseFromCSV(self, filename):
        with open(filename, 'rb') as f:
            reader = csv.reader(f, delimiter=',')
            for i,row in enumerate(reader):
                if i == 0:
                    rowStr = ','.join(row)
                    assert rowStr == ModelInformation.header
                else:
                    model_id, front, nmaterials, minPoint, maxPoint, aligned_dims, _, variantIds = row
                    if model_id in self.model_info:
                        raise Exception('Model %s already exists!' % (model_id))
                    
                    front = np.fromstring(front, dtype=np.float64, sep=',')
                    nmaterials = int(nmaterials)
                    minPoint = np.fromstring(minPoint, dtype=np.float64, sep=',')
                    maxPoint = np.fromstring(maxPoint, dtype=np.float64, sep=',')
                    aligned_dims = np.fromstring(aligned_dims, dtype=np.float64, sep=',')
                    variantIds = variantIds.split(',')
                    self.model_info[model_id] = {'front':front,
                                                 'nmaterials':nmaterials,
                                                 'minPoint':minPoint,
                                                 'maxPoint':maxPoint,
                                                 'aligned_dims':aligned_dims,
                                                 'variantIds':variantIds}

    def getModelInfo(self, model_id):
        return self.model_info[model_id]

class ModelCategoryMapping(object):

    header = 'index,model_id,fine_grained_class,coarse_grained_class,empty_struct_obj,nyuv2_40class,wnsynsetid,wnsynsetkey'

    fineMapping = {
                'ATM':'ATM',
                'accordion':'accordion',
                'air_conditioner':'air conditioner',
                'amplifier':'amplifier',
                'arch':'arch',
                'armchair':'armchair',
                'baby_bed':'baby bed',
                'basketball_hoop':'basketball hoop',
                'bathtub':'bathtub',
                'beer':'beer',
                'bench_chair':'bench chair',
                'bicycle':'bicycle',
                'book':'book',
                'books':'books',
                'bookshelf':'bookshelf',
                'bottle':'bottle',
                'bread':'bread',
                'bunker_bed':'bunker bed',
                'cake':'cake',
                'camera':'camera',
                'candle':'candle',
                'car':'car',
                'cart':'cart',
                'ceiling':'ceiling',
                'ceiling_fan':'ceiling fan',
                'cellphone':'cellphone',
                'chair':'chair',
                'chair_set':'chair set',
                'chandelier':'chandelier',
                'chessboard':'chessboard',
                'clock':'clock',
                'cloth':'cloth',
                'coffee_kettle':'coffee kettle',
                'coffee_machine':'coffee machine',
                'coffee_table':'coffee table',
                'coffin':'coffin',
                'column':'column',
                'computer':'computer',
                'container':'container',
                'containers':'containers',
                'cooker':'cooker',
                'cup':'cup',
                'curtain':'curtain',
                'cutting_board':'cutting board',
                'decoration':'decoration',
                'desk':'desk',
                'dining_table':'dining table',
                'dishwasher':'dishwasher',
                'door':'door',
                'double_bed':'double bed',
                'dresser':'dresser',
                'dressing_table':'dressing table',
                'drinkbar':'drinkbar',
                'drumset':'drumset',
                'dryer':'dryer',
                'empty':'empty',
                'fence':'fence',
                'fireplace':'fireplace',
                'fish_tank':'fish tank',
                'fishbowl':'fishbowl',
                'floor':'floor',
                'floor_lamp':'floor lamp',
                'food':'food',
                'food_processor':'food processor',
                'food_tray':'food tray',
                'fork':'fork',
                'fruit_bowl':'fruit bowl',
                'game_table':'game table',
                'garage_door':'garage door',
                'glass':'glass',
                'goal_post':'goal post',
                'gramophone':'gramophone',
                'grill':'grill',
                'guitar':'guitar',
                'gym_equipment':'gym equipment',
                'hair_dryer':'hair dryer',
                'hanger':'hanger',
                'hanging_kitchen_cabinet':'hanging kitchen cabinet',
                'headphones_on_stand':'headphones on stand',
                'headstone':'headstone',
                'heater':'heater',
                'helmet':'helmet',
                'household_appliance':'household appliance',
                'ipad':'ipad',
                'iron':'iron',
                'ironing_board':'ironing board',
                'jug':'jug',
                'kettle':'kettle',
                'keyboard':'keyboard',
                'kitchen_cabinet':'kitchen cabinet',
                'kitchen_set':'kitchen set',
                'knife':'knife',
                'knife_rack':'knife rack',
                'ladder':'ladder',
                'laptop':'laptop',
                'lawn_mower':'lawn mower',
                'loudspeaker':'loudspeaker',
                'magazines':'magazines',
                'mailbox':'mailbox',
                'microphone':'microphone',
                'microwave':'microwave',
                'mirror':'mirror',
                'mortar_and_pestle':'mortar and pestle',
                'motorcycle':'motorcycle',
                'office_chair':'office chair',
                'ottoman':'ottoman',
                'outdoor_lamp':'outdoor lamp',
                'outdoor_seating':'outdoor seating',
                'outdoor_spring':'outdoor spring',
                'pan':'pan',
                'partition':'partition',
                'pedestal_fan':'pedestal fan',
                'person':'person',
                'pet':'pet',
                'piano':'piano',
                'picture_frame':'picture frame',
                'pillow':'pillow',
                'place_setting':'place setting',
                'plant':'plant',
                'plates':'plates',
                'playstation':'playstation',
                'poker_chips':'poker chips',
                'pool':'pool',
                'range_hood':'range hood',
                'range_hood_with_cabinet':'range hood with cabinet',
                'range_oven':'range oven',
                'range_oven_with_hood':'range oven with hood',
                'refrigerator':'refrigerator',
                'rifle_on_wall':'rifle on wall',
                'roof':'roof',
                'rug':'rug',
                'safe':'safe',
                'shelving':'shelving',
                'shoes':'shoes',
                'shoes_cabinet':'shoes cabinet',
                'shower':'shower',
                'single_bed':'single bed',
                'sink':'sink',
                'slot_machine':'slot machine',
                'slot_machine_and_chair':'slot machine and chair',
                'soap_dish':'soap dish',
                'soap_dispenser':'soap dispenser',
                'sofa':'sofa',
                'spoon':'spoon',
                'stairs':'stairs',
                'stand':'stand',
                'stationary_container':'stationary container',
                'stereo_set':'stereo set',
                'storage_bench':'storage bench',
                'surveillance_camera':'surveillance camera',
                'switch':'switch',
                'table':'table',
                'table_and_chair':'table and chair',
                'table_lamp':'table lamp',
                'teapot':'teapot',
                'telephone':'telephone',
                'television':'television',
                'theremin':'theremin',
                'toaster':'toaster',
                'toilet':'toilet',
                'toilet_paper':'toilet paper',
                'toilet_plunger':'toilet plunger',
                'toiletries':'toiletries',
                'towel_hanger':'towel hanger',
                'towel_rack':'towel rack',
                'toy':'toy',
                'trash_can':'trash can',
                'tricycle':'tricycle',
                'trinket':'trinket',
                'tripod':'tripod',
                'tv_stand':'tv stand',
                'umbrella':'umbrella',
                'unknown':'unknown',
                'utensil_holder':'utensil holder',
                'vacuum_cleaner':'vacuum cleaner',
                'vase':'vase',
                'wall':'wall',
                'wall_lamp':'wall lamp',
                'wardrobe_cabinet':'wardrobe cabinet',
                'washer':'washer',
                'water_dispenser':'water dispenser',
                'weight_scale':'weight scale',
                'whiteboard':'whiteboard',
                'window':'window',
                'wood_board':'wood board',
                'workplace':'workplace',
                'xbox':'xbox',
        }
    
    coarseMapping = {
                'ATM':'ATM',
                'air_conditioner':'air conditioner',
                'arch':'arch',
                'bathroom_stuff':'bathroom stuff',
                'bathtub':'bathtub',
                'bed':'bed',
                'bench_chair':'bench chair',
                'books':'books',
                'candle':'candle',
                'cart':'cart',
                'ceiling':'ceiling',
                'chair':'chair',
                'clock':'clock',
                'cloth':'cloth',
                'coffin':'coffin',
                'column':'column',
                'computer':'computer',
                'curtain':'curtain',
                'decoration':'decoration',
                'desk':'desk',
                'door':'door',
                'dresser':'dresser',
                'dressing_table':'dressing table',
                'drinkbar':'drinkbar',
                'empty':'empty',
                'fan':'fan',
                'fence':'fence',
                'fireplace':'fireplace',
                'floor':'floor',
                'garage_door':'garage door',
                'grill':'grill',
                'gym_equipment':'gym equipment',
                'hanger':'hanger',
                'hanging_kitchen_cabinet':'hanging kitchen cabinet',
                'headstone':'headstone',
                'heater':'heater',
                'household_appliance':'household appliance',
                'indoor_lamp':'indoor lamp',
                'kitchen_appliance':'kitchen appliance',
                'kitchen_cabinet':'kitchen cabinet',
                'kitchen_set':'kitchen set',
                'kitchenware':'kitchenware',
                'magazines':'magazines',
                'mailbox':'mailbox',
                'mirror':'mirror',
                'music':'music',
                'ottoman':'ottoman',
                'outdoor_cover':'outdoor cover',
                'outdoor_lamp':'outdoor lamp',
                'outdoor_seating':'outdoor seating',
                'outdoor_spring':'outdoor spring',
                'partition':'partition',
                'person':'person',
                'pet':'pet',
                'picture_frame':'picture frame',
                'pillow':'pillow',
                'plant':'plant',
                'pool':'pool',
                'recreation':'recreation',
                'roof':'roof',
                'rug':'rug',
                'safe':'safe',
                'shelving':'shelving',
                'shoes':'shoes',
                'shoes_cabinet':'shoes cabinet',
                'shower':'shower',
                'sink':'sink',
                'sofa':'sofa',
                'stairs':'stairs',
                'stand':'stand',
                'storage_bench':'storage bench',
                'switch':'switch',
                'table':'table',
                'table_and_chair':'table and chair',
                'television':'television',
                'toilet':'toilet',
                'toy':'toy',
                'trash_can':'trash can',
                'trinket':'trinket',
                'tripod':'tripod',
                'tv_stand':'tv stand',
                'unknown':'unknown',
                'vase':'vase',
                'vehicle':'vehicle',
                'wall':'wall',
                'wardrobe_cabinet':'wardrobe cabinet',
                'whiteboard':'whiteboard',
                'window':'window',
                'wood_board':'wood board',
                'workplace':'workplace',
        }

    def __init__(self, filename):
        self.model_id = []
        self.fine_grained_class = {}
        self.coarse_grained_class = {}
        self.nyuv2_40class = {}
        self.wnsynsetid = {}
        self.wnsynsetkey = {}
        
        self._parseFromCSV(filename)
        
    def _parseFromCSV(self, filename):
        with open(filename, 'rb') as f:
            reader = csv.reader(f, delimiter=',')
            for i,row in enumerate(reader):
                if i == 0:
                    rowStr = ','.join(row)
                    assert rowStr == ModelCategoryMapping.header
                else:
                    _, model_id, fine_grained_class, coarse_grained_class, _, nyuv2_40class, wnsynsetid, wnsynsetkey = row
                    if model_id in self.model_id:
                        raise Exception('Model %s already exists!' % (model_id))
                    self.model_id.append(model_id)
                    
                    self.fine_grained_class[model_id] = fine_grained_class
                    self.coarse_grained_class[model_id] = coarse_grained_class
                    self.nyuv2_40class[model_id] = nyuv2_40class
                    self.wnsynsetid[model_id] = wnsynsetid
                    self.wnsynsetkey[model_id] = wnsynsetkey

    def _printFineGrainedClassListAsDict(self):
        for c in sorted(set(self.fine_grained_class.values())):
            name = c.replace("_", " ")
            print "'%s':'%s'," % (c, name)
    
    def _printCoarseGrainedClassListAsDict(self):
        for c in sorted(set(self.coarse_grained_class.values())):
            name = c.replace("_", " ")
            print "'%s':'%s'," % (c, name)
    
    def getFineGrainedCategoryForModelId(self, modelId):
        return ModelCategoryMapping.fineMapping[str(modelId)]
    
    def getFineGrainedClassList(self):
        return sorted(set(self.fine_grained_class.values()))
    
    def getCoarseGrainedClassList(self):
        return sorted(set(self.coarse_grained_class.values()))
    