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

import os
import scipy.io
import logging
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from string import digits
from evert import Room as EvertRoom
from evert import Source, Listener, Vector3, Matrix3, Polygon, PathSolution, Viewer

from multimodalmaze.rendering import get3DTrianglesFromModel, getColorAttributesFromModel

from panda3d.core import AmbientLight, LVector3f, LVecBase3, VBase4, Mat4, ClockObject, \
                         Material, PointLight, LineStream, SceneGraphAnalyzer, TransformState

from panda3d.core import GraphicsEngine, GraphicsPipeSelection, Loader, LoaderOptions, NodePath, RescaleNormalAttrib, AntialiasAttrib, Filename, \
                         Texture, GraphicsPipe, GraphicsOutput, FrameBufferProperties, WindowProperties, Camera, PerspectiveLens, ModelNode

logger = logging.getLogger(__name__)

MODEL_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "models")

class AcousticObject(object):
   
    def getTransform(self):
        return NotImplementedError()

    def setTransform(self):
        return NotImplementedError()

class AcousticWorld(object):

    def __init__(self):
        pass

    def addStaticSourceToScene(self, source):
        pass

    def addAgentToScene(self, agent):
        pass

    def addObjectToScene(self, obj):
        pass
    
    def addRoomToScene(self, room):
        pass
    
    def addHouseToScene(self, house):
        pass

    def step(self, time):
        pass

def getAcousticPolygonsFromModel(model):
    polygons = []
    triangles = get3DTrianglesFromModel(model)
    for triangle in triangles:
        pts = []
        for pt in triangle:
            #NOTE: EVERT works in milimeter units
            pts.append(Vector3(pt[0]*1000.0,pt[1]*1000.0,pt[2]*1000.0))
        polygons.append(Polygon(pts))
    return polygons, triangles
    
class HRTF(object):
    
    def __init__(self, nbChannels, samplingRate, impulseSamplingRate):
        self.nbChannels = nbChannels
        self.samplingRate = samplingRate
        self.impulseSamplingRate = impulseSamplingRate
        self.elevations = None
        self.azimuts = None
        self.impulses = None
        
        self.timeMargin = 0
        self.impulsesFourier = None
        
    def _calculateImpulsesFourier(self):
        # Caching responses in Fourier domain
        N = self.impulses.shape[-1]
        nbSamples = int(N * self.samplingRate / self.impulseSamplingRate)
        if nbSamples != N:
            try:
                # Use high quality resampling if available
                # https://pypi.python.org/pypi/resampy
                import resampy
                impulses = resampy.resample(self.impulses, self.impulseSamplingRate, self.samplingRate, axis=-1)
            except ImportError:
                logger.warn("Using lower quality resampling routine!")
                impulses = scipy.signal.resample(self.impulses, nbSamples, axis=-1)
        else:
            impulses = self.impulses
        
        self.timeMargin = int(np.floor(N * self.samplingRate/self.impulseSamplingRate))
        self.impulsesFourier = np.fft.fft(impulses, N + self.timeMargin)

    def getImpulseResponse(self, azimut, elevation):
        closestAzimutIdx = np.argmin(np.sqrt((self.azimuts - azimut)**2))
        closestElevationIdx = np.argmin(np.sqrt((self.elevations - elevation)**2))
        return self.impulses[closestAzimutIdx, closestElevationIdx]
        
class CipicHRTF(HRTF):
    
    def __init__(self, filename, samplingRate):
        
        super(CipicHRTF, self).__init__(nbChannels=2,
                                        samplingRate=samplingRate,
                                        impulseSamplingRate=44100.0)
        
        self.filename = filename
        self.elevations = np.linspace(-45, 230.625, num=50) * np.pi/180
        self.azimuts = np.concatenate(([-80, -65, -55], np.linspace(-45, 45, num=19), [55, 65, 80])) * np.pi/180
        self.impulses = self._loadImpulsesFromFile()
        
        self._calculateImpulsesFourier()
        
    def _loadImpulsesFromFile(self):
        
        # Load CIPIC HRTF data
        cipic = scipy.io.loadmat(self.filename)
        hrirLeft = np.transpose(cipic['hrir_l'], [2, 0, 1])
        hrirRight = np.transpose(cipic['hrir_r'], [2, 0, 1])
        
        # Store impulse responses in time domain
        N = len(hrirLeft[:, 0, 0])
        impulses = np.zeros((len(self.azimuts), len(self.elevations), self.nbChannels, N))
        for i in range(len(self.azimuts)):
            for j in range(len(self.elevations)):
                impulses[i, j, 0, :] = hrirLeft[:, i, j]
                impulses[i, j, 1, :] = hrirRight[:, i, j]

        return impulses
    
class TextureAbsorptionTable(object):
    
    textures = {
        "default":['hard surfaces', 'average'],
        "beam":['hard surfaces', 'average'],
        "bricks":['hard surfaces', 'walls rendered brickwork'],
        "carpet":['floor coverings', 'soft carpet'],
        "decorstone":['hard surfaces', 'limestone walls'],
        "facing_stone":['hard surfaces', 'ceramic tiles with a smooth surface'],
        "grass":['floor coverings', 'hairy carpet'],
        "ground":['hard surfaces', 'average'],
        "laminate":['floor coverings', 'linoleum, asphalt, rubber, or cork tile on concrete'],
        "leather":['curtains', 'cotton curtains'],
        "linen":['curtains', 'cotton curtains'],
        "lnm":['floor coverings', 'linoleum, asphalt, rubber, or cork tile on concrete'],
        "paneling":['wood', 'thin plywood panelling'],  
        "path":['hard surfaces', 'concrete'],
        "potang":['linings', 'wooden lining'],
        "prkt":['wood', "stage floor"],
        "solo":['wood', "wood, 1.6 cm thick"],
        "stone":['hard surfaces', 'concrete'],
        "stucco":['linings', 'plasterboard on steel frame'],
        "textile":['curtains', 'cotton curtains'],
        "tile":['hard surfaces', 'ceramic tiles with a smooth surface'],
        "wallp":['linings', 'plasterboard on steel frame'],
        "wood":['wood', "wood, 1.6 cm thick"],
        }
    
    @staticmethod
    def getMeanAbsorptionCoefficientsFromModel(model, units='dB'):
        
        # Get the list of materials
        areas, _, _, textures = getColorAttributesFromModel(model)

        totalCoefficients = np.zeros(len(MaterialAbsorptionTable.frequencies))
        for area, texture in zip(areas, textures):
            
            if texture is None:
                texture = 'default'
            else:
                # Remove any digits
                texture = texture.translate(None, digits)
                
                # Remove trailing underscores
                texture = texture.rstrip("_")
                
                #NOTE: handle many variations of textile and wood in SUNCG texture names
                if "textile" in texture:
                    texture = "textile"
                if "wood" in texture:
                    texture = "wood"
                    
            if not texture in TextureAbsorptionTable.textures:
                logger.warn('Unsupported texture basename for material acoustics: %s' % (texture))
                texture = 'default'
                
            category, material = TextureAbsorptionTable.textures[texture]
            coefficients, _ = MaterialAbsorptionTable.getAbsorptionCoefficients(category, material, units='normalized')
            totalCoefficients += area * coefficients
        
        if units == 'dB':
            eps = np.finfo(np.float).eps
            totalCoefficients = 20.0 * np.log10(1.0 - coefficients + eps)
        elif units == 'normalized':
            # Nothing to do
            pass
        else:
            raise Exception('Unsupported units: %s' % (units))
            
        return totalCoefficients
    
class MaterialAbsorptionTable(object):
    
    # From: Auralization : fundamentals of acoustics, modelling, simulation, algorithms and acoustic virtual reality
    # https://cds.cern.ch/record/1251519/files/978-3-540-48830-9_BookBackMatter.pdf
    
    categories = ['hard surfaces', 'linings', 'glazing', 'wood', 
                  'floor coverings', 'curtains', 'wall absorbers',
                  'ceiling absorbers', 'special absorbers']
    frequencies = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
    
    materials = [[  # Massive constructions and hard surfaces 
                    "average",                                  # Walls, hard surfaces average (brick walls, plaster, hard floors, etc.) 
                    "walls rendered brickwork",                 # Walls, rendered brickwork
                    "rough concrete",                           # Rough concrete
                    "smooth unpainted concrete",                # Smooth unpainted concrete 
                    "rough lime wash",                          # Rough lime wash 
                    "smooth brickwork with flush pointing, painted", # Smooth brickwork with flush pointing, painted 
                    "smooth brickwork, 10 mm deep pointing, pit sand mortar", # Smooth brickwork, 10 mm deep pointing, pit sand mortar 
                    "brick wall, stuccoed with a rough finish", # Brick wall, stuccoed with a rough finish 
                    "ceramic tiles with a smooth surface",      # Ceramic tiles with a smooth surface 
                    "limestone walls",                          # Limestone walls 
                    "reverberation chamber walls",              # Reverberation chamber walls 
                    "concrete",                                 # Concrete floor 
                    "marble floor",                             # Marble floor 
                ],
                [   # Lightweight constructions and linings
                    "plasterboard on steel frame",              # 2 * 13 mm plasterboard on steel frame, 50 mm mineral wool in cavity, surface painted 
                    "wooden lining",                            # Wooden lining, 12 mm fixed on frame 
                ],
                [   # Glazing
                    "single pane of glass",                     # Single pane of glass, 3 mm                                                  
                    "glass window",                             # Glass window, 0.68 kg/m^2
                    "lead glazing",                             # Lead glazing
                    "double glazing, 30 mm gap",                # Double glazing, 2-3 mm glass,  > 30 mm gap 
                    "double glazing, 10 mm gap ",               # Double glazing, 2-3 mm glass,  10 mm gap 
                    "double glazing, lead on the inside",       # Double glazing, lead on the inside
                ],
                [   # Wood
                    "wood, 1.6 cm thick",                       # Wood, 1.6 cm thick,  on 4 cm wooden planks 
                    "thin plywood panelling",                   # Thin plywood panelling
                    "16 mm wood on 40 mm studs",                # 16 mm wood on 40 mm studs 
                    "audience floor",                           # Audience floor, 2 layers,  33 mm on sleepers over concrete 
                    "stage floor",                              # Wood, stage floor, 2 layers, 27 mm over airspace 
                    "solid wooden door",                        # Solid wooden door 
                ],
                [   # Floor coverings
                    "linoleum, asphalt, rubber, or cork tile on concrete", # Linoleum, asphalt, rubber, or cork tile on concrete 
                    "cotton carpet",                            # Cotton carpet 
                    "loop pile tufted carpet",                  # Loop pile tufted carpet, 1.4 kg/m^2, 9.5 mm pile height: On hair pad, 3.0kg/m^2
                    "thin carpet",                              # Thin carpet, cemented to concrete
                    "pile carpet bonded to closed-cell foam underlay", # 6 mm pile carpet bonded to closed-cell foam underlay 
                    "pile carpet bonded to open-cell foam underlay", # 6 mm pile carpet bonded to open-cell foam underlay 
                    "tufted pile carpet",                       # 9 mm tufted pile carpet on felt underlay
                    "needle felt",                              # Needle felt 5 mm stuck to concrete 
                    "soft carpet",                              # 10 mm soft carpet on concrete
                    "hairy carpet",                             # Hairy carpet on 3 mm felt 
                    "rubber carpet",                            # 5 mm rubber carpet on concrete 
                    "carpet on hair felt or foam rubber",       # Carpet 1.35 kg/m^2, on hair felt or foam rubber 
                    "cocos fibre roll felt",                    # Cocos fibre roll felt, 29 mm thick (unstressed), reverse side clad  with paper, 2.2kg/m^2, 2 Rayl 
                ],
                [   # Curtains
                    "cotton curtains",                          # Cotton curtains (0.5 kg/m^2) draped to 3/4 area approx. 130 mm from wall
                    "curtains",                                 # Curtains (0.2 kg/m^2) hung 90 mm from wall 
                    "cotton cloth",                             # Cotton cloth (0.33 kg/m^2) folded to 7/8 area 
                    "densely woven window curtains",            # Densely woven window curtains 90 mm from wall 
                    "vertical blinds, half opened",             # Vertical blinds, 15 cm from wall,   half opened (45 deg) 
                    "vertical blinds, open",                    # Vertical blinds, 15 cm from wall,   open (90 deg) 
                    "tight velvet curtains",                    # Tight velvet curtains 
                    "curtain fabric",                           # Curtain fabric, 15 cm from wall 
                    "curtain fabric, folded",                   # Curtain fabric, folded, 15 cm from wall
                    "curtain of close-woven glass mat",         # Curtains of close-woven glass mat   hung 50 mm from wall 
                    "studio curtain",                           # Studio curtains, 22 cm from wall 
                ],
                [   # Wall absorbers
                    #TODO: fill table from paper
                ],
                [   # Ceiling absorbers
                    #TODO: fill table from paper
                ],
                [   # Special absorbers
                    #TODO: fill table from paper
                ],
            ]
    
    # Tables of random-incidence absorption coefficients
    table = [   [   # Massive constructions and hard surfaces 
                    [0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.05], # Walls, hard surfaces average (brick walls, plaster, hard floors, etc.) 
                    [0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04], # Walls, rendered brickwork
                    [0.02, 0.03, 0.03, 0.03, 0.04, 0.07, 0.07], # Rough concrete
                    [0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05], # Smooth unpainted concrete 
                    [0.02, 0.03, 0.04, 0.05, 0.04, 0.03, 0.02], # Rough lime wash 
                    [0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02], # Smooth brickwork with flush pointing, painted 
                    [0.08, 0.09, 0.12, 0.16, 0.22, 0.24, 0.24], # Smooth brickwork, 10 mm deep pointing, pit sand mortar 
                    [0.03, 0.03, 0.03, 0.04, 0.05, 0.07, 0.07], # Brick wall, stuccoed with a rough finish 
                    [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02], # Ceramic tiles with a smooth surface 
                    [0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.05], # Limestone walls 
                    [0.01, 0.01, 0.01, 0.02, 0.02, 0.04, 0.04], # Reverberation chamber walls 
                    [0.01, 0.03, 0.05, 0.02, 0.02, 0.02, 0.02], # Concrete floor 
                    [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02], # Marble floor 
                ],
                [   # Lightweight constructions and linings
                    [0.15, 0.10, 0.06, 0.04, 0.04, 0.05, 0.05], # 2 * 13 mm plasterboard on steel frame, 50 mm mineral wool in cavity, surface painted 
                    [0.27, 0.23, 0.22, 0.15, 0.10, 0.07, 0.06], # Wooden lining, 12 mm fixed on frame 
                ],
                [   # Glazing
                    [0.08, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02], # Single pane of glass, 3 mm                                                  
                    [0.10, 0.05, 0.04, 0.03, 0.03, 0.03, 0.03], # Glass window,, 0.68 kg/m^2
                    [0.30, 0.20, 0.14, 0.10, 0.05, 0.05, 0.05], # Lead glazing
                    [0.15, 0.05, 0.03, 0.03, 0.02, 0.02, 0.02], # Double glazing, 2-3 mm glass,  > 30 mm gap 
                    [0.10, 0.07, 0.05, 0.03, 0.02, 0.02, 0.02], # Double glazing, 2-3 mm glass,  10 mm gap 
                    [0.15, 0.30, 0.18, 0.10, 0.05, 0.05, 0.05], # Double glazing, lead on the inside
                ],
                [   # Wood
                    [0.18, 0.12, 0.10, 0.09, 0.08, 0.07, 0.07], # Wood, 1.6 cm thick,  on 4 cm wooden planks 
                    [0.42, 0.21, 0.10, 0.08, 0.06, 0.06, 0.06], # Thin plywood panelling
                    [0.18, 0.12, 0.10, 0.09, 0.08, 0.07, 0.07], # 16 mm wood on 40 mm studs 
                    [0.09, 0.06, 0.05, 0.05, 0.05, 0.04, 0.04], # Audience floor, 2 layers,  33 mm on sleepers over concrete 
                    [0.10, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06], # Wood, stage floor, 2 layers, 27 mm over airspace 
                    [0.14, 0.10, 0.06, 0.08, 0.10, 0.10, 0.10], # Solid wooden door 
                ],
                [   # Floor coverings
                    [0.02, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02], # Linoleum, asphalt, rubber, or cork tile on concrete 
                    [0.07, 0.31, 0.49, 0.81, 0.66, 0.54, 0.48], # Cotton carpet 
                    [0.10, 0.40, 0.62, 0.70, 0.63, 0.88, 0.88], # Loop pile tufted carpet, 1.4 kg/m^2, 9.5 mm pile height: On hair pad, 3.0kg/m^2
                    [0.02, 0.04, 0.08, 0.20, 0.35, 0.40, 0.40], # Thin carpet, cemented to concrete
                    [0.03, 0.09, 0.25, 0.31, 0.33, 0.44, 0.44], # 6 mm pile carpet bonded to closed-cell foam underlay 
                    [0.03, 0.09, 0.20, 0.54, 0.70, 0.72, 0.72], # 6 mm pile carpet bonded to open-cell foam underlay 
                    [0.08, 0.08, 0.30, 0.60, 0.75, 0.80, 0.80], # 9 mm tufted pile carpet on felt underlay
                    [0.02, 0.02, 0.05, 0.15, 0.30, 0.40, 0.40], # Needle felt 5 mm stuck to concrete 
                    [0.09, 0.08, 0.21, 0.26, 0.27, 0.37, 0.37], # 10 mm soft carpet on concrete
                    [0.11, 0.14, 0.37, 0.43, 0.27, 0.25, 0.25], # Hairy carpet on 3 mm felt 
                    [0.04, 0.04, 0.08, 0.12, 0.10, 0.10, 0.10], # 5 mm rubber carpet on concrete 
                    [0.08, 0.24, 0.57, 0.69, 0.71, 0.73, 0.73], # Carpet 1.35 kg/m^2, on hair felt or foam rubber 
                    [0.10, 0.13, 0.22, 0.35, 0.47, 0.57, 0.57], # Cocos fibre roll felt, 29 mm thick (unstressed), reverse side clad  with paper, 2.2kg/m^2, 2 Rayl 
                ],
                [   # Curtains
                    [0.30, 0.45, 0.65, 0.56, 0.59, 0.71, 0.71], # Cotton curtains (0.5 kg/m^2) draped to 3/4 area approx. 130 mm from wall
                    [0.05, 0.06, 0.39, 0.63, 0.70, 0.73, 0.73], # Curtains (0.2 kg/m^2) hung 90 mm from wall 
                    [0.03, 0.12, 0.15, 0.27, 0.37, 0.42, 0.42], # Cotton cloth (0.33 kg/m^2) folded to 7/8 area 
                    [0.06, 0.10, 0.38, 0.63, 0.70, 0.73, 0.73], # Densely woven window curtains 90 mm from wall 
                    [0.03, 0.09, 0.24, 0.46, 0.79, 0.76, 0.76], # Vertical blinds, 15 cm from wall,   half opened (45 deg) 
                    [0.03, 0.06, 0.13, 0.28, 0.49, 0.56, 0.56], # Vertical blinds, 15 cm from wall,   open (90 deg) 
                    [0.05, 0.12, 0.35, 0.45, 0.38, 0.36, 0.36], # Tight velvet curtains 
                    [0.10, 0.38, 0.63, 0.52, 0.55, 0.65, 0.65], # Curtain fabric, 15 cm from wall 
                    [0.12, 0.60, 0.98, 1.00, 1.00, 1.00, 1.00], # Curtain fabric, folded, 15 cm from wall
                    [0.03, 0.03, 0.15, 0.40, 0.50, 0.50, 0.50], # Curtains of close-woven glass mat   hung 50 mm from wall 
                    [0.36, 0.26, 0.51, 0.45, 0.62, 0.76, 0.76], # Studio curtains, 22 cm from wall 
                ],
                [   # Wall absorbers
                    #TODO: fill table from paper
                ],
                [   # Ceiling absorbers
                    #TODO: fill table from paper
                ],
                [   # Special absorbers
                    #TODO: fill table from paper
                ],
            ]
    
    @staticmethod
    def getAbsorptionCoefficients(category, material, units='dB'):
        
        category = category.lower().strip()
        if category not in MaterialAbsorptionTable.categories:
            raise Exception('Unknown category for material absorption table: %s' % (category))
        categoryIdx = MaterialAbsorptionTable.categories.index(category)
        
        material = material.lower().strip()
        if material not in MaterialAbsorptionTable.materials[categoryIdx]:
            raise Exception('Unknown material for category %s in material absorption table: %s' % (category, material))
        materialIdx = MaterialAbsorptionTable.materials[categoryIdx].index(material)
        
        coefficients = np.array(MaterialAbsorptionTable.table[categoryIdx][materialIdx])
        frequencies = np.array(AirAttenuationTable.frequencies)
        
        if units == 'dB':
            eps = np.finfo(np.float).eps
            coefficients = 20.0 * np.log10(1.0 - coefficients + eps)
        elif units == 'normalized':
            # Nothing to do
            pass
        else:
            raise Exception('Unsupported units: %s' % (units))
            
        return coefficients, frequencies
    
class AirAttenuationTable(object):
    
    # From: Auralization : fundamentals of acoustics, modelling, simulation, algorithms and acoustic virtual reality
    
    temperatures = [10.0, 20.0]
    relativeHumidities = [40.0, 60.0, 80.0]
    frequencies = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
    
    # Air attenuation coefficient, in 10^-3 / m 
    table = [   [ # 10 deg C
                    [0.1, 0.2, 0.5, 1.1, 2.7, 9.4, 29.0],  # 30-50% hum
                    [0.1, 0.2, 0.5, 0.8, 1.8, 5.9, 21.1],  # 50-70% hum
                    [0.1, 0.2, 0.5, 0.7, 1.4, 4.4, 15.8],  # 70-90% hum
                ],
                [ # 20 deg C
                    [0.1, 0.3, 0.6, 1.0, 1.9, 5.8, 20.3],  # 30-50% hum
                    [0.1, 0.3, 0.6, 1.0, 1.7, 4.1, 13.5],  # 50-70% hum
                    [0.1, 0.3, 0.6, 1.1, 1.7, 3.5, 10.6],  # 70-90% hum
                ]
            ]
    
    @staticmethod
    def getAttenuations(distance, temperature, relativeHumidity, units='dB'):
        closestTemperatureIdx = np.argmin(np.sqrt((np.array(AirAttenuationTable.temperatures) - temperature)**2))
        closestHumidityIdx = np.argmin(np.sqrt((np.array(AirAttenuationTable.relativeHumidities) - relativeHumidity)**2))
        
        attenuations = np.array(AirAttenuationTable.table[closestTemperatureIdx][closestHumidityIdx])
        frequencies = np.array(AirAttenuationTable.frequencies)
        
        eps = np.finfo(np.float).eps
        attenuations = np.clip(distance * 1e-3 * attenuations, 0.0, 1.0 - eps)
        
        if units == 'dB':
            eps = np.finfo(np.float).eps
            attenuations = 20.0 * np.log10(1.0 - attenuations + eps)
        elif units == 'normalized':
            # Nothing to do
            pass
        else:
            raise Exception('Unsupported units: %s' % (units))
        
        return attenuations, frequencies
    
class FilterBank(object):
    
    def __init__(self, n, centerFrequencies, samplingRate):
        self.n = n
        
        if n % 2 == 0:
            self.n = n + 1
            logger.warn('Length of the FIR filter adjusted to the next odd number to ensure symmetry: %d' % (self.n))
        else:
            self.n = n
            
        self.centerFrequencies = centerFrequencies
        self.samplingRate = samplingRate
    
        centerFrequencies = np.array(centerFrequencies, dtype=np.float)
        centerNormFreqs = centerFrequencies/(self.samplingRate/2.0)
        cutoffs = centerNormFreqs[:-1] + np.diff(centerNormFreqs)/2
        
        filters = []
        for i in range(len(centerFrequencies)):
            if i == 0:
                # Low-pass filter
                b = signal.firwin(self.n, cutoff=cutoffs[0], window='hamming')
            elif i == len(centerFrequencies) - 1:
                # High-pass filter
                b = signal.firwin(self.n, cutoff=cutoffs[-1], window = 'hamming', pass_zero=False)
            else:
                # Band-pass filter
                b = signal.firwin(self.n, [cutoffs[i-1], cutoffs[i]], pass_zero=False)
                
            filters.append(b)
        self.filters = np.array(filters)
    
    def getScaledImpulseResponse(self, scales=1):
        if not isinstance(scales, (list, tuple)):
            scales = scales * np.ones(len(self.filters))
        return np.sum(self.filters * scales[:, np.newaxis], axis=0)
        
    def display(self, scales=1, merged=False):
        # Adapted from: http://mpastell.com/2010/01/18/fir-with-scipy/
        
        if merged:
            b = self.getScaledImpulseResponse(scales)
            filters = [b]
        else:
            filters = np.copy(self.filters)
            if not isinstance(scales, (list, tuple)):
                scales = scales * np.ones(len(filters))
            filters *= scales[:,np.newaxis]
        
        fig = plt.figure(figsize=(8,6), facecolor='white', frameon=True)
        for b in filters:
            w,h = signal.freqz(b,1)
            h_dB = 20 * np.log10(abs(h))
            plt.subplot(211)
            plt.plot(w/max(w),h_dB)
            plt.ylim(-150, 5)
            plt.ylabel('Magnitude (db)')
            plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
            plt.title(r'Frequency response')
            plt.subplot(212)
            h_Phase = np.unwrap(np.arctan2(np.imag(h),np.real(h)))
            plt.plot(w/max(w),h_Phase)
            plt.ylabel('Phase (radians)')
            plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
            plt.title(r'Phase response')
            plt.subplots_adjust(hspace=0.5)
        return fig
    
def getPathLength(path):
    
    # Calculate path length and corresponding delay
    pathLength = 0.0
    lastPt = path.m_points[0]
    for pt in path.m_points[1:]:
        pathLength += np.sqrt((lastPt.x - pt.x)**2 + 
                              (lastPt.y - pt.y)**2 + 
                              (lastPt.z - pt.z)**2)
        lastPt = pt
        
    #NOTE: EVERT works in milimeter units
    pathLength = pathLength / 1000.0 # mm to m
    return pathLength
    
def getIntersectionPointsFromPath(path):
    pts = []
    epts = path.m_points
    for i in range(len(epts)):
        if i > 0:
            segLength = np.sqrt((epts[i-1].x - epts[i].x)**2 + 
                                (epts[i-1].y - epts[i].y)**2 + 
                                (epts[i-1].z - epts[i].z)**2)
        
            # Skip duplicated points
            #TODO: we may not need to check for duplicates in geometry, as this was due to another bug
            if segLength == 0.0: 
                continue
            
        #NOTE: EVERT works in milimeter units        
        pts.append(LVector3f(epts[i].x / 1000.0, epts[i].y / 1000.0, epts[i].z / 1000.0))
    return pts
    
def getIntersectedMaterialIdsFromPath(path):
    
    polygons = []
    lastPt = path.m_points[0]
    for i, pt in enumerate(path.m_points[1:]):
        segLength = np.sqrt((lastPt.x - pt.x)**2 + 
                            (lastPt.y - pt.y)**2 + 
                            (lastPt.z - pt.z)**2)
        
        # Skip duplicated points
        #TODO: we may not need to check for duplicates in geometry, as this was due to another bug
        if segLength == 0.0: 
            continue
        
        if i >= 2:
            polygons.append(path.m_polygons[i-2])
        lastPt = pt
    
    materialIds = []
    for polygon in polygons:
        materialIds.append(polygon.getMaterialId())
    
    return materialIds


# Moller-Trumbore ray-triangle intersection algorithm vectorized with Numpy
# Adapted from: https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
# See also: https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
def rayIntersectsTriangles(startPt, endPt, triangles, eps=1e-3):
    
    mask = np.ones((triangles.shape[0],), dtype=np.bool)
    
    d = np.linalg.norm((endPt - startPt), 2)
    vdir = (endPt - startPt) / d
    
    vertex0, vertex1, vertex2  = triangles[:,0,:], triangles[:,1,:], triangles[:,2,:]
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    
    h = np.cross(vdir, edge2)
    a = np.einsum('ij,ij->i', edge1, h)
    mask &= np.logical_not((a > -eps) & (a < eps))
    
    f = 1/a
    s = startPt - vertex0
    u = f * np.einsum('ij,ij->i', s, h)
    mask &= np.logical_not((u < 0.0) | (u > 1.0))
    
    q = np.cross(s, edge1)
    v = f * np.einsum('ij,ij->i', vdir[np.newaxis,:], q)
    mask &= np.logical_not((v < 0.0) | (u + v > 1.0))
    
    # NOTE: t is the distance from the start point to the intersection on the plane defined by the triangle
    t = f * np.einsum('ij,ij->i', edge2, q)
    mask &= (t > eps)
    mask &= np.logical_not(np.isclose(t, d[np.newaxis], atol=eps))
    mask &= (t < d[np.newaxis])
        
    return np.any(mask)

def validatePath(path, triangles, eps):
    
    isValid = True
    pts = getIntersectionPointsFromPath(path)
    lastPt = pts[0]
    for pt in pts[1:]:
        
        if rayIntersectsTriangles(np.array([lastPt.x, lastPt.y, lastPt.z]),
                                  np.array([pt.x, pt.y, pt.z]),
                                  triangles,
                                  eps):
            isValid = False
            break
         
        lastPt = pt
    
    return isValid

class EvertAcousticObject(AcousticObject):
    
    def __init__(self, world, nodePath, recenterTransform=None):
        self.world = world
        self.nodePath = nodePath
        
        if recenterTransform is None:
            recenterTransform = TransformState.makeIdentity()
        self.recenterTransform = recenterTransform
    
    def getTransform(self):
        transform = self.nodePath.node().getTransform()
        mat = transform.compose(TransformState.makePos(-self.recenterTransform.getPos())).getMat()
        return np.array([[mat[0][0], mat[0][1], mat[0][2], mat[0][3]],
                         [mat[1][0], mat[1][1], mat[1][2], mat[1][3]],
                         [mat[2][0], mat[2][1], mat[2][2], mat[2][3]],
                         [mat[3][0], mat[3][1], mat[3][2], mat[3][3]]])

    def getRecenterPosition(self):
        position = self.recenterTransform.getPos()
        return np.array([position.x, position.y, position.z])

    def setTransform(self, transform):
        mat = Mat4(*transform.ravel())
        self.nodePath.setTransform(TransformState.makeMat(mat).compose(self.recenterTransform))


class EvertAcousticWorld(AcousticWorld):

    # NOTE: the model ids of objects that correspond to opened doors. They will be ignored in the acoustic scene.
    openedDoorModelIds = [
                            # Doors
                            '122', '133', '214', '246', '247', '361', '73','756','757','758','759','760',
                            '761','762','763','764','765', '768','769','770','771','778','779','780',
                            's__1762','s__1763','s__1764','s__1765','s__1766','s__1767','s__1768','s__1769',
                            's__1770','s__1771','s__1772','s__1773',
                            # Curtains
                            '275'
                         ]

    rayColors = [
                    (1.0, 1.0, 0.0, 0.2), # yellow
                    (0.0, 0.0, 1.0, 0.2), # blue
                    (0.0, 0.0, 1.0, 0.2), # green
                    (0.0, 1.0, 1.0, 0.2), # cyan
                    (1.0, 0.0, 1.0, 0.2), # magenta
                    (1.0, 0.0, 0.0, 0.2), # red
                ]

    minRayRadius = 0.01 # m
    maxRayRadius = 0.1 # m
    
    minWidthThresholdPolygons = 0.1 # m

    def __init__(self, samplingRate=16000, maximumOrder=3, 
                 materialAbsorption=True, frequencyDependent=True,
                 size=(512,512), showCeiling=True, debug=False):

        self.debug = debug
        self.samplingRate = samplingRate
        self.maximumOrder = maximumOrder
        self.materialAbsorption = materialAbsorption
        self.frequencyDependent = frequencyDependent
        self.world = EvertRoom()
        self.solutions = []
        self.render = NodePath('acoustic-render')
        self.globalClock = ClockObject.getGlobalClock()
        
        self.filterbank = FilterBank(n=257, 
                                     centerFrequencies=MaterialAbsorptionTable.frequencies,
                                     samplingRate=samplingRate)
        
        #TODO: add infinite ground plane?

        self.setAirConditions()
        self.coefficientsForMaterialId = []
        
        self._initRender(size, showCeiling)
        
        self.agents = []
        self.listenerNodesByInstanceId = dict()
    
        self.sources = []
        self.sourceNodesByInstanceId = dict()
        
        self.acousticObjects = []
        self.nbTotalRejectedAcousticPolygons = 0
        
        self.geometryInitialized = False
    
    def _initRender(self, size, showCeiling):
        # Rendering attributes
        self.size = size
        self.showCeiling = showCeiling
        
        self.loader = Loader.getGlobalPtr()
        
        if self.debug:
            self.graphicsEngine = GraphicsEngine.getGlobalPtr()
            self.graphicsEngine.setDefaultLoader(self.loader)
        
            selection = GraphicsPipeSelection.getGlobalPtr()
            self.pipe = selection.makeDefaultPipe()
            logger.debug('Using %s' % (self.pipe.getInterfaceName()))
        
        self.render = NodePath('render')
        self.render.setAttrib(RescaleNormalAttrib.makeDefault())
        self.render.setTwoSided(1)
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.render.setTextureOff(1)
        
        self.camera = self.render.attachNewNode(ModelNode('camera'))
        self.camera.node().setPreserveTransform(ModelNode.PTLocal)
        
        self.scene = self.render.attachNewNode('scene')
        self.obstacles = self.scene.attachNewNode('obstacles')
        
        if self.debug:
            self._initRgbCapture()
            self._addDefaultLighting()
            self._preloadRayModels()
        
    def _initRgbCapture(self):

        camNode = Camera('RGB camera')
        lens = PerspectiveLens()
        lens.setFov(75.0)
        lens.setAspectRatio(1.0)
        lens.setNear(0.1)
        lens.setFar(1000.0)
        camNode.setLens(lens)
        camNode.setScene(self.render)
        cam = self.camera.attachNewNode(camNode)
        
        winprops = WindowProperties.getDefault()
        winprops = WindowProperties(winprops)
        winprops.setSize(self.size[0], self.size[1])
        fbprops = FrameBufferProperties.getDefault()
        fbprops = FrameBufferProperties(fbprops)
        fbprops.setRgbColor(1)
        fbprops.setColorBits(24)
        fbprops.setAlphaBits(8)
        fbprops.setDepthBits(1) 
        flags = GraphicsPipe.BFFbPropsOptional | GraphicsPipe.BFRefuseWindow
        buf = self.graphicsEngine.makeOutput(self.pipe, 'RGB buffer', 0, fbprops,
                                             winprops, flags)
        
        dr = buf.makeDisplayRegion()
        dr.setSort(0)
        dr.setCamera(cam)
        dr = camNode.getDisplayRegion(0)
        
        tex = Texture()
        tex.setFormat(Texture.FRgb)
        buf.addRenderTexture(tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor)
        
        self.rgbBuffer = buf
        self.rgbTex = tex    
        
    def _addDefaultLighting(self):
        alight = AmbientLight('alight')
        alight.setColor(VBase4(0.2, 0.2, 0.2, 1))
        alnp = self.scene.attachNewNode(alight)
        self.scene.setLight(alnp)
        
        #NOTE: Point light following the camera
        plight = PointLight('plight')
        plight.setColor(VBase4(1.0, 1.0, 1.0, 1))
        plnp = self.camera.attachNewNode(plight)
        self.scene.setLight(plnp)
        
    def destroy(self):
        if self.debug:
            self.graphicsEngine.removeAllWindows()
            del self.pipe

    def setCamera(self, mat):
        mat = Mat4(*mat.ravel())
        self.camera.setMat(mat)

    def getRgbImage(self, channelOrder="RGBA"):
        if self.debug:
            data = self.rgbTex.getRamImageAs(channelOrder)
            image = np.frombuffer(data.get_data(), np.uint8)
            image.shape = (self.rgbTex.getYSize(), self.rgbTex.getXSize(), self.rgbTex.getNumComponents())
            image = np.flipud(image)
        else:
            image = np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)
            
        return image
    
    def visualize(self):
        self._updateSources()
        self._updateListeners()
        viewer = Viewer(self.world, self.maximumOrder)
        viewer.show()
    
    def _renderInfo(self):
        sga = SceneGraphAnalyzer()
        sga.addNode(self.render.node())
        
        ls = LineStream()
        sga.write(ls)
        desc = []
        while ls.isTextAvailable():
            desc.append(ls.getLine())
        desc = '\n'.join(desc)
        return desc
    
    def _loadModel(self, modelPath):
        loader = Loader.getGlobalPtr()
        loaderOptions = LoaderOptions()
        node = loader.loadSync(Filename(modelPath), loaderOptions)
        if node is not None:
            nodePath = NodePath(node)
        else:
            raise IOError('Could not load model file: %s' % (modelPath))
        return nodePath
    
    def _loadSphereModel(self, refCenter, radius, color=(1.0,0.0,0.0,1.0)):
    
        model = self._loadModel(os.path.join(MODEL_DATA_DIR, 'sphere.egg')) 
        
        # Rescale the cube model to match the bounding box of the original model
        minBounds, maxBounds = model.getTightBounds()
        dims = maxBounds - minBounds
        pos = model.getPos()
        center = minBounds + (maxBounds - minBounds) / 2.0
        deltaCenter = center - pos
        
        position = refCenter - deltaCenter
        model.setPos(position)
        
        scale = LVector3f(radius/dims.x, radius/dims.y, radius/dims.z)
        model.setScale(scale)
        
        # Validate approximation
        eps = 1e-4
        minBounds, maxBounds = model.getTightBounds()
        center = minBounds + (maxBounds - minBounds) / 2.0
        dims = maxBounds - minBounds
        assert np.allclose([center.x, center.y, center.z], 
                           [refCenter.x, refCenter.y, refCenter.z], 
                           atol=eps)
        
        material = Material()
        material.setAmbient(color)
        material.setDiffuse(color)
        model.clearMaterial()
        model.setMaterial(material, 1)
        
        return model
    
    def setAirConditions(self, pressureAtm=1.0, temperature=20.0, relativeHumidity=65.0):
        self.pressureAtm = pressureAtm
        self.temperature = temperature
        self.relativeHumidity = relativeHumidity

    def _calculateSoundSpeed(self):
        # Approximate speed of sound in dry (0% humidity) air, in meters per second, at temperatures near 0 deg C
        #TODO: alternative with humidity: http://resource.npl.co.uk/acoustics/techguides/speedair/
        # See paper: http://asa.scitation.org/doi/pdf/10.1121/1.405827
        return 331.3*np.sqrt(1+self.temperature/273.15)
    
    def _calculateDelayAndAttenuation(self, path):
        
        pathLength = getPathLength(path)
        delay = pathLength/self._calculateSoundSpeed()
        
        # Calculate air attenuation coefficient (dB)
        airAttenuations, frequencies = AirAttenuationTable.getAttenuations(pathLength, self.temperature, self.relativeHumidity, units='dB')

        # Calculate spherical geometric spreading attenuation (dB)
        distanceAttenuations = 20.0 * np.log10(1.0/pathLength)
        
        # Calculat material attenuation (dB)
        materialAttenuations = np.zeros((len(MaterialAbsorptionTable.frequencies),))
        if self.materialAbsorption:
            for materialId in getIntersectedMaterialIdsFromPath(path):
                materialAbsorption = self.coefficientsForMaterialId[materialId]
                eps = np.finfo(np.float).eps
                materialAbsorptionDb = 20.0 * np.log10(1.0 - materialAbsorption + eps)
                materialAttenuations += materialAbsorptionDb
        
        # Total attenuation (dB)
        attenuation = airAttenuations + distanceAttenuations + materialAttenuations
        #assert np.all(attenuation < 0.0)
         
        return delay, attenuation, frequencies
    
    def _preloadRayModels(self, initialCacheSize=256):
        
        self.rays = []
        self.nbUsedRays = 0
        
        # Load cylinder model and calculate the scaling factor to make the model unit-norm over the Z axis
        model = self._loadModel(os.path.join(MODEL_DATA_DIR, 'cylinder.egg'))
        minBounds, maxBounds = model.getTightBounds()
        dims = maxBounds - minBounds
        self.rayModelZScaling = 1.0/dims.z
        model.removeNode()
        
        self.rayGroupNode = self.scene.attachNewNode('rays')
        self.rayGroupNode.reparentTo(self.scene)

        self._resizeRayCache(initialCacheSize)
        
    def _resizeRayCache(self, size):
        
        if size < len(self.rays):
            # Remove unused rays from the cache
            for model in self.rays[size:]:
                model.detachNode()
                model.removeNode()
                
            self.rays = self.rays[:size]
        else:
            # Add new rays to the cache
            for _ in range(size - len(self.rays)):
                
                # Load cylinder model
                model = self._loadModel(os.path.join(MODEL_DATA_DIR, 'cylinder.egg'))
                model.clearMaterial()
                model.reparentTo(self.rayGroupNode)
                model.hide()
                
                self.rays.append(model)
    
    def _updateRayModelFromEndpoints(self, model, startPt, endPt, radius, color):

        # Clear previous transform
        model.clearMat()
            
        # Reference length and center position of ray
        refLength = np.sqrt( (endPt.x - startPt.x)**2 + 
                             (endPt.y - startPt.y)**2 + 
                             (endPt.z - startPt.z)**2)
        refCenter = (endPt + startPt)/2.0
            
        # Change orientation by calculating rotation angles from the endpoints
        # NOTE: H angle is how the model rotates around the (0, 0, 1) axis,
        #       P angle how much it rotates around the (1, 0, 0) axis, 
        #       R angle how much it rotates around the (0, 1, 0) axis.
        normVec = (endPt - startPt) / refLength
        
        # Rotation around X in the Y-Z plane
        angle = np.pi/2 + np.arcsin(-normVec.z)
        model.setHpr(0.0, angle*180/np.pi, 0.0)
        transAroundX = model.getMat()
        
        # Rotation around Z in the X-Y plane
        angle = np.pi/2 + np.arctan2(normVec.y, normVec.x)
        model.setHpr(angle*180/np.pi, 0.0, 0.0)
        transAroundZ = model.getMat()
        
        model.setMat(transAroundX * transAroundZ)
        
        # Move ray to reference center position
        minBounds, maxBounds = model.getTightBounds()
        center = minBounds + (maxBounds - minBounds) / 2.0
        deltaCenter = center - model.getPos()
        model.setPos(refCenter - deltaCenter)
        
        # Change radius based on attenuation
        model.setScale((radius, radius, refLength * self.rayModelZScaling))
        
        # Change color
        model.setColor(*color)
            
    def _calculateAttenuationPerSegment(self, path):
        
        pts = getIntersectionPointsFromPath(path)
        
        totalSegAttenuations = np.zeros((len(pts)-1,))
        lastPt = pts[0]
        cumLength = 0.0
        for i, pt in enumerate(pts[1:]):
            segLength = np.sqrt((lastPt.x - pt.x)**2 + 
                                (lastPt.y - pt.y)**2 + 
                                (lastPt.z - pt.z)**2)
            assert segLength > 0.0
            cumLength += segLength
            
            # Calculate air attenuation coefficient (dB)
            airAttenuations, _ = AirAttenuationTable.getAttenuations(cumLength, self.temperature, self.relativeHumidity, units='dB')
    
            # Calculate spherical geometric spreading attenuation (dB)
            distanceAttenuations = 20.0 * np.log10(1.0/cumLength)
            
            totalSegAttenuations[i] = np.mean(airAttenuations) + distanceAttenuations
            
            lastPt = pt
        
        # Calculate material attenuation (dB)
        if self.materialAbsorption:
            for i, materialId in enumerate(getIntersectedMaterialIdsFromPath(path)):
                materialAbsorption = self.coefficientsForMaterialId[materialId]
                eps = np.finfo(np.float).eps
                materialAbsorptionDb = 20.0 * np.log10(1.0 - materialAbsorption + eps)
                totalSegAttenuations[i+1:] += np.mean(materialAbsorptionDb)
        
        #assert np.all(totalSegAttenuations < 0.0)
        
        return totalSegAttenuations
        
    def _renderAcousticPath(self, path, color):
        
        totalSegAttenuationsDb = self._calculateAttenuationPerSegment(path)
        
        pts = getIntersectionPointsFromPath(path)
        startPt = pts[0]
        for endPt, attenuationDb in zip(pts[1:], totalSegAttenuationsDb):
            
            coefficient = 10.0 ** (attenuationDb/20.0)
            #assert coefficient >= 0.0 and coefficient <= 1.0
            radius = np.clip(self.maxRayRadius * coefficient, a_min=self.minRayRadius, a_max=self.maxRayRadius)
            
            if self.nbUsedRays < len(self.rays):
                model = self.rays[self.nbUsedRays]
                self._updateRayModelFromEndpoints(model, startPt, endPt, radius, color)
                model.show()
                self.nbUsedRays += 1
            else:
                nextSize = 2 * len(self.rays)
                logger.debug('Ray cache is full: increasing the size from %d to %d' % (len(self.rays), nextSize))
                self._resizeRayCache(nextSize)
                
            startPt = endPt
                    
    def _renderAcousticSolutions(self):
        
        # Reset the number of used rays
        self.nbUsedRays = 0
        
        # Draw each solution with a different color
        for i, solution in enumerate(self.solutions):
            
            # Rotate amongst colors of the predefined table
            color = self.rayColors[i%len(self.rayColors)]
            
            # Sort by increasing path lengh
            paths = []
            for n in range(solution.numPaths()):
                path = solution.getPath(n)
                pathLength = getPathLength(path)
                paths.append((pathLength, path))
            paths.sort(key=lambda x: x[0])
            paths = [path for _, path in paths]
            logger.debug('Number of paths found for solution %d: %d' % (i, solution.numPaths()))
            
            # Draw each solution path
            for path in paths:
                self._renderAcousticPath(path, color)
            
        # Hide all unused rays in the cache
        for i in range(self.nbUsedRays, len(self.rays)):
            self.rays[i].hide()
            
    def calculateImpulseResponse(self, solution, maxImpulseLength=1.0, threshold=120.0):
    
        impulse = np.zeros((int(maxImpulseLength * self.samplingRate),))
        realImpulseLength = 0.0
        for i in range(solution.numPaths()):
            path = solution.getPath(i)
        
            delay, attenuationsDb, _ = self._calculateDelayAndAttenuation(path)
            
            # Random phase inversion
            phase = 1.0
            if np.random.random() > 0.5:
                phase *= -1
            
            # Add path impulse to global impulse
            delaySamples = int(delay * self.samplingRate)
            
            # Skip paths that are below attenuation threshold (dB)
            if np.any(abs(attenuationsDb) < threshold):
                
                if self.frequencyDependent:
                
                    # Skip paths that would have their impulse responses truncated at the end
                    if delaySamples + self.filterbank.n < len(impulse):
                    
                        linearGains = 10.0 ** (attenuationsDb/20.0)
                        pathImpulse = self.filterbank.getScaledImpulseResponse(linearGains)
                            
                        startIdx = delaySamples - self.filterbank.n/2
                        endIdx = startIdx + self.filterbank.n - 1
                        if startIdx < 0:
                            trimStartIdx = -startIdx
                            startIdx = 0
                        else:
                            trimStartIdx = 0
                        
                        impulse[startIdx:endIdx+1] += phase * pathImpulse[trimStartIdx:]
                    
                        if endIdx+1 > realImpulseLength:
                            realImpulseLength = endIdx+1
                else:
                    # Use attenuation at 1000 Hz
                    linearGain = 10.0 ** (attenuationsDb[3]/20.0)
                    impulse[delaySamples] += phase * linearGain
                    if delaySamples+1 > realImpulseLength:
                        realImpulseLength = delaySamples+1
                
        return impulse[:realImpulseLength]
    
    def addStaticSourceToScene(self, obj, radius=0.25, color=(1.0,0.0,0.0,1.0)):

        # Load model for the sound source
        sourceNode = self.scene.attachNewNode('acoustic-source-' + str(obj.instanceId))
        sourceNode.reparentTo(self.scene)
        model = self._loadSphereModel(LVector3f(0,0,0), radius, color)
        model.reparentTo(sourceNode)
        
        # Create source in EVERT
        src = Source()
        src.setName(str(obj.instanceId) + '-src')
        self.world.addSource(src)
        
        self.sources.append(obj)
        self.sourceNodesByInstanceId[obj.instanceId] = sourceNode
        
        instance = EvertAcousticObject(self.world, sourceNode)
        obj.setAcousticObject(instance)
        self.acousticObjects.append(instance)
        
        return sourceNode
    
    def addAgentToScene(self, agent, interauralDistance=0.25):
        
        #TODO: for binaural, add a plane between the two listeners to mimick head-related occlusion?
        #      The problem is that accounts for a moving polygon, which doesn't fit with the precomputed
        #      beam search tree. But we could still find the solution paths and remove manually those intersecting this polygon! 
        
        # Load model for the agent
        agentNode = self.scene.attachNewNode('agent-' + str(agent.instanceId))
        agentNode.reparentTo(self.scene)
        
        # Load model for the left microphone
        leftMicNode = agentNode.attachNewNode('microphone-left')
        transform = TransformState.makePos(LVector3f(-interauralDistance/2, 0.0, 0.0))
        leftMicNode.setTransform(transform)
        model = self._loadSphereModel(refCenter=LVecBase3(0.0, 0.0, 0.0), radius=0.10, color=(1.0,0.0,0.0,1.0))
        model.reparentTo(leftMicNode)
        
        # Load model for the right microphone
        rightMicNode = agentNode.attachNewNode('microphone-right')
        transform = TransformState.makePos(LVector3f(interauralDistance/2, 0.0, 0.0))
        rightMicNode.setTransform(transform)
        model = self._loadSphereModel(refCenter=LVecBase3(0.0, 0.0, 0.0), radius=0.10, color=(1.0,0.0,0.0,1.0))
        model.reparentTo(rightMicNode)
        
        # Add listeners for EVERT
        lstLeft = Listener()
        lstLeft.setName(str(agent.instanceId) + '-lst-left')
        self.world.addListener(lstLeft)
     
        lstRight = Listener()
        lstRight.setName(str(agent.instanceId) + '-lst-right')
        self.world.addListener(lstRight)
                
        self.agents.append(agent)
        self.listenerNodesByInstanceId[agent.instanceId] = [leftMicNode, rightMicNode]
                
        instance = EvertAcousticObject(self.world, agentNode)
        agent.setAcousticObject(instance)
        self.acousticObjects.append(instance)
                
        return agentNode
                
    def addObjectToScene(self, obj, mode='box'):

        nodePath = self.obstacles.attachNewNode('object-' + str(obj.instanceId))
        if not obj.modelId in self.openedDoorModelIds:

            # Load model from file
            model = self._loadModel(obj.modelFilename)
            
            coefficients = TextureAbsorptionTable.getMeanAbsorptionCoefficientsFromModel(model, units='normalized')
            materialId = len(self.coefficientsForMaterialId)
            self.coefficientsForMaterialId.append(coefficients)
    
            transform = TransformState.makeIdentity()
            if mode == 'mesh':
                # Nothing to do
                pass
            elif mode == 'box':
                # Bounding box approximation
                minRefBounds, maxRefBounds = model.getTightBounds()
                refDims = maxRefBounds - minRefBounds
                refPos = model.getPos()
                refCenter = minRefBounds + (maxRefBounds - minRefBounds) / 2.0
                refDeltaCenter = refCenter - refPos
                
                model.removeNode()
                model = self._loadModel(os.path.join(MODEL_DATA_DIR, 'cube.egg')) 
                
                # Rescale the cube model to match the bounding box of the original model
                minBounds, maxBounds = model.getTightBounds()
                dims = maxBounds - minBounds
                pos = model.getPos()
                center = minBounds + (maxBounds - minBounds) / 2.0
                deltaCenter = center - pos
                
                position = refPos + refDeltaCenter - deltaCenter
                scale = LVector3f(refDims.x/dims.x, refDims.y/dims.y, refDims.z/dims.z)
                transform = TransformState.makePos(position).compose(TransformState.makeScale(scale))
                
            else:
                raise Exception('Unknown mode type for acoustic object shape: %s' % (mode))
        
            model.reparentTo(nodePath)
            
            material = Material()
            intensity = np.mean(1.0 - coefficients)
            material.setAmbient((intensity, intensity, intensity, 1))
            material.setDiffuse((intensity, intensity, intensity, 1))
            model.clearMaterial()
            model.setMaterial(material, 1)
            model.setTag('materialId', str(materialId))
            
            instance = EvertAcousticObject(self.world, nodePath, transform)
            obj.setAcousticObject(instance)
            self.acousticObjects.append(instance)
    
        return nodePath
    
    def addRoomToScene(self, room, ignoreObjects=False):

        nodePath = self.obstacles.attachNewNode('room-' + str(room.instanceId))
        for modelFilename in room.modelFilenames:
            
            partId = os.path.splitext(os.path.basename(modelFilename))[0]
            objNode = nodePath.attachNewNode('room-' + str(room.instanceId) + '-' + partId)
            model = self._loadModel(modelFilename)
            model.reparentTo(objNode)
            
            if not self.showCeiling and 'c' in os.path.basename(modelFilename):
                objNode.hide()
            
            coefficients = TextureAbsorptionTable.getMeanAbsorptionCoefficientsFromModel(model, units='normalized')
            materialId = len(self.coefficientsForMaterialId)
            self.coefficientsForMaterialId.append(coefficients)
            
            material = Material()
            intensity = np.mean(1.0 - coefficients)
            material.setAmbient((intensity, intensity, intensity, 1))
            material.setDiffuse((intensity, intensity, intensity, 1))
            model.clearMaterial()
            model.setMaterial(material, 1)
            model.setTag('materialId', str(materialId))
            
        if not ignoreObjects:
            for obj in room.objects:
                objNode = self.addObjectToScene(obj)
                objNode.reparentTo(nodePath)
        
        return nodePath

    def addHouseToScene(self, house, ignoreObjects=False):

        nodePath = self.obstacles.attachNewNode('house-' + str(house.instanceId))
    
        for room in house.rooms:
            roomNode = self.addRoomToScene(room, ignoreObjects)
            roomNode.reparentTo(nodePath)
        
        for room in house.grounds:
            roomNode = self.addRoomToScene(room, ignoreObjects)
            roomNode.reparentTo(nodePath)
        
        if not ignoreObjects:
            for obj in house.objects:
                objNode = self.addObjectToScene(obj)
                objNode.reparentTo(nodePath)
        
        return nodePath
    
    def _updateSources(self):
        
        # Update positions of the static sources
        for source in self.sources:
            
            src = None
            for i in range(self.world.numSources()):
                wsrc = self.world.getSource(i)
                if wsrc.getName() == str(source.instanceId) + '-src':
                    src = wsrc
            assert src is not None
            
            sourceNode = self.sourceNodesByInstanceId[source.instanceId]
            
            sourceNetTans = sourceNode.getNetTransform().getMat()
            sourceNetPos = sourceNetTans.getRow3(3)
            sourceNetMat = sourceNetTans.getUpper3()
            src.setPosition(Vector3(sourceNetPos.x * 1000.0, sourceNetPos.y * 1000.0, sourceNetPos.z * 1000.0)) # m to mm
            #FIXME: EVERT seems to work in Y-up coordinate system, not Z-up like Panda3d
            #yupTransformMat = Mat4.convertMat(CS_zup_right, CS_yup_right)
            #sourceNetMat = (sourceNetTans * yupTransformMat).getUpper3()
            src.setOrientation(Matrix3(sourceNetMat.getCell(0,0), sourceNetMat.getCell(0,1), sourceNetMat.getCell(0,2),
                                       sourceNetMat.getCell(1,0), sourceNetMat.getCell(1,1), sourceNetMat.getCell(1,2),
                                       sourceNetMat.getCell(2,0), sourceNetMat.getCell(2,1), sourceNetMat.getCell(2,2)))
            logger.debug('Static source %s: at position (x=%f, y=%f, z=%f)' % (source.instanceId, sourceNetPos.x, sourceNetPos.y, sourceNetPos.z))
    
    def updateGeometry(self):
        
        # Find all model nodes in the graph
        modelNodes = self.obstacles.findAllMatches('**/+ModelNode')
        for model in modelNodes:
    
            materialId = int(model.getTag('materialId'))
    
            # Add polygons to EVERT engine
            #TODO: for bounding box approximations, we could reduce the number of triangles by
            #      half if each face of the box was modelled as a single rectangular polygon.
            polygons, triangles = getAcousticPolygonsFromModel(model)
            for polygon, triangle in zip(polygons,triangles):
                
                # Validate triangle
                dims = np.max(triangle, axis=0) - np.min(triangle, axis=0)
                s = np.sum(np.array(dims > self.minWidthThresholdPolygons, dtype=np.int))
                if s < 2:
                    continue
                
                polygon.setMaterialId(materialId)
                self.world.addPolygon(polygon, Vector3(1.0,1.0,1.0))
    
        self._updateSources()
        
        self.world.constructBSP()
        
        del self.solutions[:]
        for s in range(self.world.numSources()):
            for l in range(self.world.numListeners()):
                src = self.world.getSource(s)
                lst = self.world.getListener(l)
                solution = PathSolution(self.world, src, lst, self.maximumOrder)
                self.solutions.append(solution)
    
        self.geometryInitialized = True
    
    def _updateListeners(self):
    
        # Update positions of listeners
        for agent in self.agents:
            
            lstLeft = None
            lstRight = None
            for i in range(self.world.numListeners()):
                lst = self.world.getListener(i)
                if lst.getName() == str(agent.instanceId) + '-lst-left':
                    lstLeft = lst
                elif lst.getName() == str(agent.instanceId) + '-lst-right':
                    lstRight = lst
            assert lstLeft is not None
            assert lstRight is not None
            
            lstLeftNode, lstRightNode = self.listenerNodesByInstanceId[agent.instanceId]
            
            leftNetTans = lstLeftNode.getNetTransform().getMat()
            leftNetPos = leftNetTans.getRow3(3)
            leftNetMat = leftNetTans.getUpper3()
            lstLeft.setPosition(Vector3(leftNetPos.x * 1000.0, leftNetPos.y * 1000.0, leftNetPos.z * 1000.0)) # m to mm
            #FIXME: EVERT seems to work in Y-up coordinate system, not Z-up like Panda3d
            #yupTransformMat = Mat4.convertMat(CS_zup_right, CS_yup_right)
            #leftNetMat = (leftNetTans * yupTransformMat).getUpper3()
            lstLeft.setOrientation(Matrix3(leftNetMat.getCell(0,0), leftNetMat.getCell(0,1), leftNetMat.getCell(0,2),
                                           leftNetMat.getCell(1,0), leftNetMat.getCell(1,1), leftNetMat.getCell(1,2),
                                           leftNetMat.getCell(2,0), leftNetMat.getCell(2,1), leftNetMat.getCell(2,2)))
            
            rightNetTans = lstRightNode.getNetTransform().getMat()
            rightNetPos = rightNetTans.getRow3(3)
            rightNetMat = rightNetTans.getUpper3()
            lstRight.setPosition(Vector3(rightNetPos.x * 1000.0, rightNetPos.y * 1000.0, rightNetPos.z * 1000.0)) # m to mm
            #FIXME: EVERT seems to work in Y-up coordinate system, not Z-up like Panda3d
            #yupTransformMat = Mat4.convertMat(CS_zup_right, CS_yup_right)
            #rightNetMat = (leftNetTans * yupTransformMat).getUpper3()
            lstRight.setOrientation(Matrix3(rightNetMat.getCell(0,0), rightNetMat.getCell(0,1), rightNetMat.getCell(0,2),
                                            rightNetMat.getCell(1,0), rightNetMat.getCell(1,1), rightNetMat.getCell(1,2),
                                            rightNetMat.getCell(2,0), rightNetMat.getCell(2,1), rightNetMat.getCell(2,2)))
        
            logger.debug('Agent %s: left microphone at position (x=%f, y=%f, z=%f)' % (agent.instanceId, leftNetPos.x, leftNetPos.y, leftNetPos.z))
            logger.debug('Agent %s: right microphone at position (x=%f, y=%f, z=%f)' % (agent.instanceId, rightNetPos.x, rightNetPos.y, rightNetPos.z))
    
    def step(self):
        #dt = self.globalClock.getDt()
        
        if not self.geometryInitialized:
            self.updateGeometry()
        
        self._updateListeners()
            
        # Update solutions
        for solution in self.solutions:
            solution.update()
        
        if self.debug:
            #NOTE: we may need to call frame rendering twice because of double-buffering
            self._renderAcousticSolutions()
            self.graphicsEngine.renderFrame()
        