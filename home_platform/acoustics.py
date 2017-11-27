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
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from string import digits
from evert import Room as EvertRoom
from evert import Source, Listener, Vector3, Matrix3, Polygon, PathSolution
from evert import Viewer as EvertViewer

from home_platform.core import World
from home_platform.suncg import loadModel
from home_platform.rendering import get3DTrianglesFromModel, getColorAttributesFromModel
from home_platform.utils import vec3ToNumpyArray

from panda3d.core import NodePath, LVector3f, LVecBase3, Material, TransformState, AudioSound, VBase3, CS_zup_right,\
    ClockObject
from direct.showbase.Audio3DManager import Audio3DManager
from direct.task.TaskManagerGlobal import taskMgr
from direct.task.Task import Task

logger = logging.getLogger(__name__)

MODEL_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "models")

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
    
    def __init__(self, nbChannels, samplingRate, maxLength=None):
        self.nbChannels = nbChannels
        self.samplingRate = samplingRate
        self.maxLength = maxLength
        self.elevations = None
        self.azimuts = None
        self.impulses = None
        self.channels = None
        
        self.timeMargin = 0
        self.impulsesFourier = None
    
    def _precomputeImpulsesFourier(self):
        
        N = self.impulses.shape[-1]
        if self.maxLength is not None:
            N = self.maxLength
            
        self.timeMargin = N
        self.impulsesFourier = np.fft.fft(self.impulses, N + self.timeMargin)

    def resample(self, newSamplingRate, maxLength=None):
        
        if maxLength is not None:
            self.maxLength = maxLength
        
        N = self.impulses.shape[-1]
        nbSamples = int(N * newSamplingRate / self.samplingRate)
        if nbSamples != N:
            #TODO: resampy function doesn't seem to work for 3D and 4D tensors (it returns only zeros)
#             try:
#                 # Use high quality resampling if available
#                 # https://pypi.python.org/pypi/resampy
#                 import resampy
#                 self.impulses = resampy.resample(self.impulses, self.samplingRate, newSamplingRate, axis=-1)
#             except ImportError:
#                 logger.warn("Using lower quality resampling routine!")

            #TODO: for high-quality resampling, we may simply do linear interpolation (but it is slower)
            self.impulses = scipy.signal.resample(self.impulses, nbSamples, axis=-1)
        
        self._precomputeImpulsesFourier()
        
    def getImpulseResponse(self, azimut, elevation):
        closestAzimutIdx = np.argmin(np.sqrt((self.azimuts - azimut)**2))
        closestElevationIdx = np.argmin(np.sqrt((self.elevations - elevation)**2))
        return self.impulses[closestAzimutIdx, closestElevationIdx]
    
    def getFourierImpulseResponse(self, azimut, elevation):
        if self.impulsesFourier is None:
            self._calculateImpulsesFourier()
        
        closestAzimutIdx = np.argmin(np.sqrt((self.azimuts - azimut)**2))
        closestElevationIdx = np.argmin(np.sqrt((self.elevations - elevation)**2))
        return self.impulsesFourier[closestAzimutIdx, closestElevationIdx]
        
def interauralPolarToVerticalPolarCoordinates(elevations, azimuts):
    pass    
    
class CipicHRTF(HRTF):
    
    def __init__(self, filename, samplingRate):
        
        super(CipicHRTF, self).__init__(nbChannels=2,
                                        samplingRate=44100.0)
        
        self.filename = filename
        
        # FIXME: CIPIC defines elevation and azimut angle in a interaural-polar coordinate system.
        #        We actually want to use the vertical-polar coordinate system.
        # see: http://www.ece.ucdavis.edu/cipic/spatial-sound/tutorial/psychoacoustics-of-spatial-hearing/
        self.elevations = np.linspace(-45, 230.625, num=50) * np.pi/180
        self.azimuts = np.concatenate(([-80, -65, -55], np.linspace(-45, 45, num=19), [55, 65, 80])) * np.pi/180
        
        self.impulses = self._loadImpulsesFromFile()
        self.channels = ['left', 'right']

        self.resample(samplingRate)
        self._precomputeImpulsesFourier()
        
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
    
    def __init__(self, n, centerFrequencies, samplingRate, maxLength=None):
        self.n = n
        
        if n % 2 == 0:
            self.n = n + 1
            logger.warn('Length of the FIR filter adjusted to the next odd number to ensure symmetry: %d' % (self.n))
        else:
            self.n = n
            
        self.centerFrequencies = centerFrequencies
        self.samplingRate = samplingRate
        self.maxLength = maxLength
    
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
        
        self._precomputeFiltersFourier()
        
    def _precomputeFiltersFourier(self):
        N = self.filters.shape[-1]
        if self.maxLength is not None:
            N = self.maxLength
            
        self.filtersFourier = np.fft.fft(self.filters, N)
        
    def getScaledImpulseResponse(self, scales=1):
        if not isinstance(scales, (list, tuple)):
            scales = scales * np.ones(len(self.filters))
        return np.sum(self.filters * scales[:, np.newaxis], axis=0)

    def getScaledImpulseResponseFourier(self, scales=1):
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

def getAcousticModelNodeForModel(model, mode='box'):
    
    transform = TransformState.makeIdentity()
    if mode == 'mesh':
        acousticModel = model.copyTo(model.getParent())
        acousticModel.detachNode()
        acousticModel.show()
        
    elif mode == 'box':
        # Bounding box approximation
        minRefBounds, maxRefBounds = model.getTightBounds()
        refDims = maxRefBounds - minRefBounds
        refPos = model.getPos()
        refCenter = minRefBounds + (maxRefBounds - minRefBounds) / 2.0
        refDeltaCenter = refCenter - refPos
        
        acousticModel = loadModel(os.path.join(MODEL_DATA_DIR, 'cube.egg')) 
        
        # Rescale the cube model to match the bounding box of the original model
        minBounds, maxBounds = acousticModel.getTightBounds()
        dims = maxBounds - minBounds
        pos = acousticModel.getPos()
        center = minBounds + (maxBounds - minBounds) / 2.0
        deltaCenter = center - pos
        
        position = refPos + refDeltaCenter - deltaCenter
        scale = LVector3f(refDims.x/dims.x, refDims.y/dims.y, refDims.z/dims.z)
        transform = TransformState.makePos(position).compose(TransformState.makeScale(scale))
        
        # TODO: validate applied transform here
        
    else:
        raise Exception('Unknown mode type for acoustic object shape: %s' % (mode))

    acousticModel.setName(model.getName())
    acousticModel.setTransform(acousticModel.getTransform().compose(transform))

    return acousticModel

class AcousticImpulseResponse(object):
    
    def __init__(self, impulse, samplingRate, source, target):
        self.__dict__.update(impulse=impulse, samplingRate=samplingRate,
                             source=source, target=target)

        # EVERT instance
        self.solution = None

class EvertAcoustics(World):

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
    
    def __init__(self, scene, hrtf=None, samplingRate=16000, maximumOrder=3, materialAbsorption=True, frequencyDependent=True, debug=False,
                 microphoneTransform=None, objectMode='box', minWidthThresholdPolygons=0.0, maxImpulseLength=1.0, threshold=120.0):

        super(EvertAcoustics, self).__init__()

        self.__dict__.update(scene=scene, hrtf=hrtf, samplingRate=samplingRate, maximumOrder=maximumOrder, materialAbsorption=materialAbsorption, 
                             frequencyDependent=frequencyDependent, debug=debug, microphoneTransform=microphoneTransform, objectMode=objectMode,
                             minWidthThresholdPolygons=minWidthThresholdPolygons, maxImpulseLength=maxImpulseLength,
                             threshold=threshold)

        self.debug = debug
        self.samplingRate = samplingRate
        self.maximumOrder = maximumOrder
        self.materialAbsorption = materialAbsorption
        self.frequencyDependent = frequencyDependent
        
        self.world = EvertRoom()
        self.solutions = dict()
        self.render = NodePath('acoustic-render')
        
        self.filterbank = FilterBank(n=257, 
                                     centerFrequencies=MaterialAbsorptionTable.frequencies,
                                     samplingRate=samplingRate)
        
        if self.hrtf is not None:
            self.hrtf.resample(samplingRate)
        
        #TODO: add infinite ground plane?

        self.setAirConditions()
        self.coefficientsForMaterialId = []
        
        self.agents = []
        self.sources = []
        self.acousticImpulseResponses = []
        
        self.geometryInitialized = False
    
        self._initLayoutModels()
        self._initObjects()
        self._initAgents()
        self._initSources()
        
        if self.debug:
            self._preloadRayModels()
        
        self.scene.worlds['acoustics'] = self
        
    def destroy(self):
        # Nothing to do
        pass
        
    def visualizeEVERT(self):
        self._updateSources()
        self._updateListeners()
        viewer = EvertViewer(self.world, self.maximumOrder)
        viewer.show()
    
    def _loadSphereModel(self, refCenter, radius, color=(1.0,0.0,0.0,1.0)):
    
        model = loadModel(os.path.join(MODEL_DATA_DIR, 'sphere.egg')) 
        
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
        model = loadModel(os.path.join(MODEL_DATA_DIR, 'cylinder.egg'))
        minBounds, maxBounds = model.getTightBounds()
        dims = maxBounds - minBounds
        self.rayModelZScaling = 1.0/dims.z
        model.removeNode()
        
        self.rayGroupNode = self.scene.scene.attachNewNode('rays')

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
                model = loadModel(os.path.join(MODEL_DATA_DIR, 'cylinder.egg'))
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
                    
    def _updateRenderedAcousticSolutions(self):
        
        # Reset the number of used rays
        self.nbUsedRays = 0
        
        # Draw each solution with a different color
        for i, (solution, _, _) in enumerate(self.solutions.itervalues()):
            
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
    
    def _calculatePathRelativeToMicrophone(self, path, microphoneNp):
        
        # Get the last segment of the path
        fromPt = LVecBase3(path.m_points[-2].x, path.m_points[-2].y, path.m_points[-2].z) / 1000.0 # mm to m
        toPt = LVecBase3(path.m_points[-1].x, path.m_points[-1].y, path.m_points[-1].z) / 1000.0  # mm to m
        assert np.allclose(vec3ToNumpyArray(toPt), 
                           vec3ToNumpyArray(microphoneNp.getNetTransform().getPos()),
                           atol=1e-6)
        
        srcDirVec = (fromPt - toPt).normalized()
        
        headTransform = microphoneNp.getNetTransform()
        headDirVec = headTransform.getNormQuat().getForward(CS_zup_right)
        
        #XXX: apply a more general calculation of azimut and elevation angles
        headRollAngle = headTransform.getHpr().getZ()
        if not np.allclose(headRollAngle, 0.0, atol=1e-6):
            logger.warn('Microphone has non-zero roll angle, which is not taken into account!')
        
        # Get the azimut in X-Y plane
        srcDirVecXY = LVector3f(srcDirVec.x, srcDirVec.y, 0.0).normalized()
        headDirVecXY = LVector3f(headDirVec.x, headDirVec.y, 0.0).normalized()
        azimut = srcDirVecXY.signedAngleRad(headDirVecXY, headDirVecXY)
        
        # Get the elevation in Y-Z plane
        srcDirVecYZ = LVector3f(0.0, srcDirVec.y, srcDirVec.z).normalized()
        headDirVecYZ = LVector3f(0.0, headDirVec.y, headDirVec.z).normalized()
        elevation = srcDirVecYZ.signedAngleRad(headDirVecYZ, headDirVecYZ)
    
        return azimut, elevation
    
    def _calculateImpulseResponse(self, solution, srcName, lstName):
    
        # Calculate impulse response in time domain, accounting for frequency-dependent absorption,
        # then when it is time to convolve with the HRTF, do it in fourrier domain.
    
        # TODO: Get the source and microphone related to this solution
        agentNp = self.scene.scene.find('**/' + lstName)
        microphoneNp = agentNp.find('**/acoustics/microphone*')
    
        sourceNp = self.scene.scene.find('**/' + srcName)
    
        if self.hrtf is not None:
            nbChannels = len(self.hrtf.channels)
        else:
            nbChannels = 1
        
        impulse = np.zeros((nbChannels, int(self.maxImpulseLength * self.samplingRate)))
        realImpulseLength = 0
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
            if np.any(abs(attenuationsDb) < self.threshold):
                
                if self.hrtf is not None:
                    # Calculate azimut and elevation angles compared to the agent
                    azimut, elevation = self._calculatePathRelativeToMicrophone(path, microphoneNp)
                    hrtfImpulse = self.hrtf.getImpulseResponse(azimut, elevation)
                
                if self.frequencyDependent:
                
                    # Skip paths that would have their impulse responses truncated at the end
                    if delaySamples + self.filterbank.n < impulse.shape[-1]:
                    
                        linearGains = 10.0 ** (attenuationsDb/20.0)
                        pathImpulse = self.filterbank.getScaledImpulseResponse(linearGains)
                        
                        for channel in range(nbChannels):
                            
                            if self.hrtf is not None:
                                #FIXME: should be using 'full' mode for convolution, and flip the hrtf impulse?
                                pathImpulseChan = signal.fftconvolve(pathImpulse, hrtfImpulse[channel] , mode='same')
                            else:
                                pathImpulseChan = pathImpulse
                                
                            startIdx = delaySamples - self.filterbank.n/2
                            endIdx = startIdx + len(pathImpulseChan) - 1
                            if startIdx < 0:
                                trimStartIdx = -startIdx
                                startIdx = 0
                            else:
                                trimStartIdx = 0
                            
                            impulse[channel, startIdx:endIdx+1] += phase * pathImpulseChan[trimStartIdx:]
                    
                        if endIdx+1 > realImpulseLength:
                            realImpulseLength = endIdx+1
                else:
                    # Use attenuation at 1000 Hz
                    linearGain = 10.0 ** (attenuationsDb[3]/20.0)
                    
                    for channel in range(nbChannels):
                        
                        if self.hrtf is not None:
                            pathImpulseChan = linearGain * hrtfImpulse[channel]
    
                            #FIXME: should be checking for truncation at the beginning and end                       
                            startIdx = delaySamples
                            endIdx = startIdx + len(pathImpulseChan) - 1
                            
                            impulse[channel, startIdx:endIdx+1] += phase * pathImpulseChan
                            if endIdx+1 > realImpulseLength:
                                realImpulseLength = endIdx+1
                
                        else:
                            impulse[channel, delaySamples] += phase * linearGain
                            if delaySamples+1 > realImpulseLength:
                                realImpulseLength = delaySamples+1
                
        # Trim impulse to effective length
        impulse = impulse[:,:realImpulseLength]
                
        return AcousticImpulseResponse(impulse, self.samplingRate, sourceNp, microphoneNp)
    

    def _initLayoutModels(self):
        
        # Load layout objects as meshes
        for model in self.scene.scene.findAllMatches('**/layouts/object*/model*'):
            
            model.getParent().setTag('acoustics-mode', 'obstacle')
            
            coefficients = TextureAbsorptionTable.getMeanAbsorptionCoefficientsFromModel(model, units='normalized')
            materialId = len(self.coefficientsForMaterialId)
            self.coefficientsForMaterialId.append(coefficients)
            
            acousticModel = getAcousticModelNodeForModel(model, mode='mesh')
            
            objectNp = model.getParent()
            acousticsNp = objectNp.attachNewNode('acoustics')
            acousticModel.reparentTo(acousticsNp)
            
            material = Material()
            intensity = np.mean(1.0 - coefficients)
            material.setAmbient((intensity, intensity, intensity, 1))
            material.setDiffuse((intensity, intensity, intensity, 1))
            acousticModel.clearMaterial()
            acousticModel.setMaterial(material, 1)
            acousticModel.setTextureOff(1)
            acousticModel.setTag('materialId', str(materialId))
            
            parent = objectNp.find('**/physics*')
            if parent.isEmpty():
                parent = objectNp
            acousticsNp.reparentTo(parent)

    def _initAgents(self):
    
        # Load agents
        for agent in self.scene.scene.findAllMatches('**/agents/agent*'):

            agent.setTag('acoustics-mode', 'listener')
            
            acousticNode = agent.attachNewNode('acoustics')
            
            # Load model for the microphone
            microphoneNp = acousticNode.attachNewNode('microphone')
            if self.microphoneTransform is not None:
                microphoneNp.setTransform(self.microphoneTransform)
            model = self._loadSphereModel(refCenter=LVecBase3(0.0, 0.0, 0.0), radius=0.15, color=(1.0,0.0,0.0,1.0))
            model.reparentTo(microphoneNp)
            
            # Add listeners for EVERT
            lst = Listener()
            lst.setName(agent.getName())
            self.world.addListener(lst)
                    
            self.agents.append(agent)
        
    def _initObjects(self):
    
        # Load objects
        for model in self.scene.scene.findAllMatches('**/objects/object*/model*'):
            modelId = model.getParent().getTag('model-id')
                 
            if modelId in self.openedDoorModelIds:
                continue
                
            # Check if object is static
            isStatic = True
            if model.hasNetTag('physics-mode'):
                # Ignore dynamic models
                if model.getNetTag('physics-mode') == 'dynamic':
                    isStatic = False
            
            isObstacle = True
            if model.hasNetTag('acoustics-mode'):
                if model.getNetTag('acoustics-mode') != 'obstacle':
                    isObstacle = False
                    
            if isObstacle and isStatic:

                model.getParent().setTag('acoustics-mode', 'obstacle')
    
                coefficients = TextureAbsorptionTable.getMeanAbsorptionCoefficientsFromModel(model, units='normalized')
                materialId = len(self.coefficientsForMaterialId)
                self.coefficientsForMaterialId.append(coefficients)
        
                acousticModel = getAcousticModelNodeForModel(model, mode=self.objectMode)
        
                objectNp = model.getParent()
                acousticsNp = objectNp.attachNewNode('acoustics')
                acousticModel.reparentTo(acousticsNp)
        
                material = Material()
                intensity = np.mean(1.0 - coefficients)
                material.setAmbient((intensity, intensity, intensity, 1))
                material.setDiffuse((intensity, intensity, intensity, 1))
                acousticModel.clearMaterial()
                acousticModel.setMaterial(material, 1)
                acousticModel.setTextureOff(1)
                acousticModel.setTag('materialId', str(materialId))
    
                parent = objectNp.find('**/physics*')
                if parent.isEmpty():
                    parent = objectNp
                acousticsNp.reparentTo(parent)
    
    def _initSources(self):
  
        # Load objects
        for obj in self.scene.scene.findAllMatches('**/objects/object*'):
            if obj.hasTag('acoustics-mode') and obj.getTag('acoustics-mode') == 'source':
                     
                if obj.hasTag('physics-mode') and obj.getTag('physics-mode') != 'static':
                    raise Exception('Sources in EVERT must be static!')
                     
                # Create source in EVERT
                src = Source()
                src.setName(obj.getName())
                self.world.addSource(src)
                  
                acousticsNp = obj.attachNewNode('acoustics')
                
                if self.debug:
                    # Load model for the sound source
                    acousticModel = self._loadSphereModel(LVector3f(0,0,0), radius=0.25, color=(1.0,0.0,0.0,1.0))
                    acousticModel.reparentTo(acousticsNp)
                  
                self.sources.append(acousticsNp)
    
    def _updateSources(self):
        
        # Update positions of the static sources
        for source in self.sources:
            
            obj = source.getParent()
            
            src = None
            for i in range(self.world.numSources()):
                wsrc = self.world.getSource(i)
                if wsrc.getName() == obj.getName():
                    src = wsrc
            assert src is not None
            
            sourceNetTrans = source.getNetTransform().getMat()
            sourceNetPos = sourceNetTrans.getRow3(3)
            sourceNetMat = sourceNetTrans.getUpper3()
            src.setPosition(Vector3(sourceNetPos.x * 1000.0, sourceNetPos.y * 1000.0, sourceNetPos.z * 1000.0)) # m to mm
            #FIXME: EVERT seems to work in Y-up coordinate system, not Z-up like Panda3d
            #yupTransformMat = Mat4.convertMat(CS_zup_right, CS_yup_right)
            #sourceNetMat = (sourceNetTans * yupTransformMat).getUpper3()
            src.setOrientation(Matrix3(sourceNetMat.getCell(0,0), sourceNetMat.getCell(0,1), sourceNetMat.getCell(0,2),
                                       sourceNetMat.getCell(1,0), sourceNetMat.getCell(1,1), sourceNetMat.getCell(1,2),
                                       sourceNetMat.getCell(2,0), sourceNetMat.getCell(2,1), sourceNetMat.getCell(2,2)))
            logger.debug('Static source %s: at position (x=%f, y=%f, z=%f)' % (obj.getName(), sourceNetPos.x, sourceNetPos.y, sourceNetPos.z))
    
    def updateGeometry(self):
        
        # Find all model nodes in the graph
        modelNodes = []
        for model in self.scene.scene.findAllMatches('**/objects/object*/acoustics/*'):
            if model.hasNetTag('acoustics-mode'):
                if model.getNetTag('acoustics-mode') != 'obstacle':
                    continue
            modelNodes.append(model)
        for model in self.scene.scene.findAllMatches('**/layouts/object*/acoustics/*'):
            if model.hasNetTag('acoustics-mode'):
                if model.getNetTag('acoustics-mode') != 'obstacle':
                    continue
            modelNodes.append(model)
                
        for model in modelNodes:
    
            materialId = int(model.getTag('materialId'))
    
            # Add polygons to EVERT engine
            #TODO: for bounding box approximations, we could reduce the number of triangles by
            #      half if each face of the box was modelled as a single rectangular polygon.
            polygons, triangles = getAcousticPolygonsFromModel(model)
            for polygon, triangle in zip(polygons,triangles):
                
                # Validate triangle (ignore if area is too small)
                dims = np.max(triangle, axis=0) - np.min(triangle, axis=0)
                s = np.sum(np.array(dims > self.minWidthThresholdPolygons, dtype=np.int))
                if s < 2:
                    continue
                
                polygon.setMaterialId(materialId)
                self.world.addPolygon(polygon, Vector3(1.0,1.0,1.0))
    
        self._updateSources()
        
        self.world.constructBSP()
        
        self.solutions.clear()
        for s in range(self.world.numSources()):
            for l in range(self.world.numListeners()):
                src = self.world.getSource(s)
                lst = self.world.getListener(l)
                solution = PathSolution(self.world, src, lst, self.maximumOrder)
                self.solutions[lst.getName()] = [solution, src.getName(), lst.getName()]
    
        self.geometryInitialized = True
    
    def _updateListeners(self):
    
        # Update positions of listeners
        for agent in self.agents:
            
            lst = None
            for i in range(self.world.numListeners()):
                listener = self.world.getListener(i)
                if listener.getName() == str(agent.getName()):
                    lst = listener
            assert lst is not None
            
            microphoneNp = agent.find('**/microphone*')
            netMat = microphoneNp.getNetTransform().getMat()
            lstNetPos = netMat.getRow3(3)
            lstNetMat = netMat.getUpper3()
            lst.setPosition(Vector3(lstNetPos.x * 1000.0, lstNetPos.y * 1000.0, lstNetPos.z * 1000.0)) # m to mm
            #FIXME: EVERT seems to work in Y-up coordinate system, not Z-up like Panda3d
            #yupTransformMat = Mat4.convertMat(CS_zup_right, CS_yup_right)
            #leftNetMat = (leftNetTans * yupTransformMat).getUpper3()
            lst.setOrientation(Matrix3(lstNetMat.getCell(0,0), lstNetMat.getCell(0,1), lstNetMat.getCell(0,2),
                                       lstNetMat.getCell(1,0), lstNetMat.getCell(1,1), lstNetMat.getCell(1,2),
                                       lstNetMat.getCell(2,0), lstNetMat.getCell(2,1), lstNetMat.getCell(2,2)))
        
            logger.debug('Agent %s: microphone at position (x=%f, y=%f, z=%f)' % (agent.getName(), lstNetPos.x, lstNetPos.y, lstNetPos.z))
    
    def calculateImpulseResponses(self):
        
        # Update each solution related to all pairs of source and listeners
        impulses = []
        for solution, srcName, lstName, in self.solutions.itervalues():
            solution.update()
            impulse = self._calculateImpulseResponse(solution, srcName, lstName)
            impulses.append(impulse)
    
        return impulses
    
    def step(self, dt):
        
        if not self.geometryInitialized:
            self.updateGeometry()
        
        self._updateListeners()
        
        impulses = self.calculateImpulseResponses()
            
        if self.debug:
            self._updateRenderedAcousticSolutions()
            
class EvertAudioSound(AudioSound):
            
    # Python implementation of AudioSound abstract class: 
    # https://www.panda3d.org/reference/1.9.4/python/panda3d.core.AudioSound
    
    def __init__(self, filename):
    
        self.name = os.path.basename(filename)
        self.filename = filename
        
        # Load sound as mono
        data, samplerate = sf.read(filename)
        self.data = data
        self.fs = samplerate
        
        self.t = 0.0
        self.isActive = True
        self.isLoop = False
        self.loopCount = 1
        self.priority = 0
        self.volume = 1.0
        self.playRate = 1.0
        self.balance = 0.0
        
        self.status = AudioSound.READY
        
    def configureFilters(self, config):
        raise NotImplementedError()
            
    def get3dMaxDistance(self):
        raise NotImplementedError()
 
    def get3dMinDistance(self):
        raise NotImplementedError()
 
    def getActive(self):
        return self.isActive
     
    def getBalance(self):
        return self.balance
 
    def getFinishedEvent(self):
        raise NotImplementedError()
 
    def getLoop(self):
        return self.isLoop
 
    def getLoopCount(self):
        return self.loopCount
 
    def getName(self):
        return self.name
            
    def getPlayRate(self):
        return self.playRate
       
    def getPriority(self):
        return self.priority
    
    def getSpeakerLevel(self, index):
        raise NotImplementedError()
             
    def getSpeakerMix(self, speaker):
        raise NotImplementedError()  

    def getTime(self):
        return self.t
             
    def getVolume(self):
        return self.volume
         
    def length(self):
        return len(self.data) / float(self.fs)
    
    def output(self, out):
        raise NotImplementedError()
        
    def play(self):
        self.status = AudioSound.PLAYING

    def set3dAttributes(self, px, py, pz, vx, vy, vz):
        raise NotImplementedError()
            
    def set3dMaxDistance(self, dist):
        raise NotImplementedError()  

    def set3dMinDistance(self, dist):
        raise NotImplementedError()
    
    def setActive(self, flag):
        self.isActive = flag
    
    def setBalance(self, balance_right):
        raise NotImplementedError()

    def setFinishedEvent(self, event):
        raise NotImplementedError()

    def setLoop(self, loop):
        self.isLoop = loop

    def setLoopCount(self, loop_count):
        self.loopCount = loop_count

    def setPlayRate(self, play_rate):
        raise NotImplementedError()

    def setPriority(self, priority):
        self.priority = priority
            
    def setSpeakerLevels(self, level1, level2, level3, level4, level5,
                         level6, level7, level8, level9):
        raise NotImplementedError()

    def setSpeakerMix(self, frontleft, frontright, center, sub, backleft, backright, sideleft, sideright):
        raise NotImplementedError()
    
    def setTime(self, start_time):
        assert start_time >= 0.0 and start_time <= 1.0
        self.t = start_time
            
    def setVolume(self, volume):
        self.volume = volume   
    
    def status(self):
        return self.status
          
    def stop(self):
        self.status = AudioSound.READY
    
    def write(self, out):
        raise NotImplementedError()
    

class EvertAudio3DManager(Audio3DManager):

    # Python implementation of the Audio3DManager class: 
    # https://www.panda3d.org/reference/1.9.4/python/direct.showbase.Audio3DManager.Audio3DManager

    def __init__(self, evertAcoustics, taskPriority=51):

        self.evertAcoustics = evertAcoustics
        self.root = self.evertAcoustics.scene.scene

        self.globalClock = ClockObject.getGlobalClock()
        self.sound_dict = {}

        taskMgr.add(self.update, "EvertAudio3DManager-updateTask", taskPriority)

    def loadSfx(self, name):
        """
        Use Audio3DManager.loadSfx to load a sound with 3D positioning enabled
        """
        sound = None
        if (name):
            sound = EvertAudioSound(name)
            
        return sound

    def setDistanceFactor(self, factor):
        """
        Control the scale that sets the distance units for 3D spacialized audio.
        Default is 1.0 which is adjust in panda to be feet.
        When you change this, don't forget that this effects the scale of setSoundMinDistance
        """
        raise NotImplementedError()

    def getDistanceFactor(self):
        """
        Control the scale that sets the distance units for 3D spacialized audio.
        Default is 1.0 which is adjust in panda to be feet.
        """
        raise NotImplementedError()

    def setDopplerFactor(self, factor):
        """
        Control the presence of the Doppler effect. Default is 1.0
        Exaggerated Doppler, use >1.0
        Diminshed Doppler, use <1.0
        """
        raise NotImplementedError()

    def getDopplerFactor(self):
        """
        Control the presence of the Doppler effect. Default is 1.0
        Exaggerated Doppler, use >1.0
        Diminshed Doppler, use <1.0
        """
        raise NotImplementedError()

    def setDropOffFactor(self, factor):
        """
        Exaggerate or diminish the effect of distance on sound. Default is 1.0
        Valid range is 0 to 10
        Faster drop off, use >1.0
        Slower drop off, use <1.0
        """
        raise NotImplementedError()

    def getDropOffFactor(self):
        """
        Exaggerate or diminish the effect of distance on sound. Default is 1.0
        Valid range is 0 to 10
        Faster drop off, use >1.0
        Slower drop off, use <1.0
        """
        raise NotImplementedError()

    def setSoundMinDistance(self, sound, dist):
        """
        Controls the distance (in units) that this sound begins to fall off.
        Also affects the rate it falls off.
        Default is 3.28 (in feet, this is 1 meter)
        Don't forget to change this when you change the DistanceFactor
        """
        sound.set3dMinDistance(dist)

    def getSoundMinDistance(self, sound):
        """
        Controls the distance (in units) that this sound begins to fall off.
        Also affects the rate it falls off.
        Default is 3.28 (in feet, this is 1 meter)
        """
        return sound.get3dMinDistance()

    def setSoundMaxDistance(self, sound, dist):
        """
        Controls the maximum distance (in units) that this sound stops falling off.
        The sound does not stop at that point, it just doesn't get any quieter.
        You should rarely need to adjust this.
        Default is 1000000000.0
        """
        sound.set3dMaxDistance(dist)

    def getSoundMaxDistance(self, sound):
        """
        Controls the maximum distance (in units) that this sound stops falling off.
        The sound does not stop at that point, it just doesn't get any quieter.
        You should rarely need to adjust this.
        Default is 1000000000.0
        """
        return sound.get3dMaxDistance()

    def setSoundVelocity(self, sound, velocity):
        """
        Set the velocity vector (in units/sec) of the sound, for calculating doppler shift.
        This is relative to the sound root (probably render).
        Default: VBase3(0, 0, 0)
        """
        raise NotImplementedError()

    def setSoundVelocityAuto(self, sound):
        """
        If velocity is set to auto, the velocity will be determined by the
        previous position of the object the sound is attached to and the frame dt.
        Make sure if you use this method that you remember to clear the previous
        transformation between frames.
        """
        raise NotImplementedError()

    def getSoundVelocity(self, sound):
        """
        Get the velocity of the sound.
        """
        if (sound in self.vel_dict):
            vel = self.vel_dict[sound]
            if (vel!=None):
                return vel
            else:
                for known_object in self.sound_dict.keys():
                    if self.sound_dict[known_object].count(sound):
                        return known_object.getPosDelta(self.root)/self.globalClock.getDt()
        return VBase3(0, 0, 0)

    def setListenerVelocity(self, velocity):
        """
        Set the velocity vector (in units/sec) of the listener, for calculating doppler shift.
        This is relative to the sound root (probably render).
        Default: VBase3(0, 0, 0)
        """
        raise NotImplementedError()

    def setListenerVelocityAuto(self):
        """
        If velocity is set to auto, the velocity will be determined by the
        previous position of the object the listener is attached to and the frame dt.
        Make sure if you use this method that you remember to clear the previous
        transformation between frames.
        """
        raise NotImplementedError()

    def getListenerVelocity(self):
        """
        Get the velocity of the listener.
        """
        if (self.listener_vel!=None):
            return self.listener_vel
        elif (self.listener_target!=None):
            return self.listener_target.getPosDelta(self.root)/self.globalClock.getDt()
        else:
            return VBase3(0, 0, 0)

    def attachSoundToObject(self, sound, object):
        """
        Sound will come from the location of the object it is attached to
        """
        # sound is an AudioSound
        # object is any Panda object with coordinates
        for known_object in self.sound_dict.keys():
            if self.sound_dict[known_object].count(sound):
                # This sound is already attached to something
                #return 0
                # detach sound
                self.sound_dict[known_object].remove(sound)
                if len(self.sound_dict[known_object]) == 0:
                    # if there are no other sounds, don't track
                    # the object any more
                    del self.sound_dict[known_object]

        if object not in self.sound_dict:
            self.sound_dict[object] = []

        self.sound_dict[object].append(sound)
        return 1


    def detachSound(self, sound):
        """
        sound will no longer have it's 3D position updated
        """
        for known_object in self.sound_dict.keys():
            if self.sound_dict[known_object].count(sound):
                self.sound_dict[known_object].remove(sound)
                if len(self.sound_dict[known_object]) == 0:
                    # if there are no other sounds, don't track
                    # the object any more
                    del self.sound_dict[known_object]
                return 1
        return 0


    def getSoundsOnObject(self, object):
        """
        returns a list of sounds attached to an object
        """
        if object not in self.sound_dict:
            return []
        sound_list = []
        sound_list.extend(self.sound_dict[object])
        return sound_list


    def attachListener(self, object):
        """
        Sounds will be heard relative to this object. Should probably be the camera.
        """
        raise NotImplementedError()


    def detachListener(self):
        """
        Sounds will be heard relative to the root, probably render.
        """
        raise NotImplementedError()


    def update(self, task=None):
        """
        Updates position of sounds in the 3D audio system. Will be called automatically
        in a task.
        """
        # Update the positions of all sounds based on the objects
        # to which they are attached
    
        return Task.cont
        
    def disable(self):
        """
        Detaches any existing sounds and removes the update task
        """
        taskMgr.remove("Audio3DManager-updateTask")
        self.detachListener()
        for object in self.sound_dict.keys():
            for sound in self.sound_dict[object]:
                self.detachSound(sound)
        
        