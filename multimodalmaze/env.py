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
#  - Neither the name of the NECOTIS research group nor the names of its contributors 
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

import heapq
import logging
import numpy as np

from multimodalmaze.core import Agent

from rendering import Panda3dRenderWorld
from physics import Panda3dBulletPhysicWorld
from multimodalmaze.acoustics import EvertAcousticWorld

logger = logging.getLogger(__name__)


def extractAllRegions(occupacyGrid):
    occupacyGrid = np.atleast_2d(occupacyGrid).astype(np.int)
    occupacyGrid[occupacyGrid != 0] = -1

    curRegionId = 1
    while np.count_nonzero(occupacyGrid == 0) > 0:

        # Apply Grassfire algorithm (also known as Wavefront or Brushfire algorithm)
        heap = []
        heapq.heapify(heap)
        visited = set()

        # Select a start point that is not yet labelled, add the heap queue
        start = tuple(np.argwhere(occupacyGrid == 0)[0])
        heapq.heappush(heap, start)
        visited.add(start)

        while len(heap) > 0:
            # Get cell from heap queue, assign to current region and add to visited set
            i, j = heapq.heappop(heap)
            occupacyGrid[i, j] = curRegionId

            # Add all 4 neighbors to heap queue, if not occupied
            for ai, aj in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:

                # Check bounds
                if (ai >= 0 and ai < occupacyGrid.shape[0] and aj >= 0 and aj < occupacyGrid.shape[1]):
                    # Check occupancy and redundancy
                    if occupacyGrid[ai, aj] == 0 and (ai, aj) not in visited:
                        heapq.heappush(heap, (ai, aj))
                        visited.add((ai, aj))

        curRegionId += 1

    assert np.all(occupacyGrid != 0)
    logger.debug('Number of regions found in occupancy map: %d' % (curRegionId - 1))

    return occupacyGrid


class Observation(object):
    def __init__(self, position, orientation, image, collision):
        self.position = position
        self.orientation = orientation
        self.image = image
        self.collision = collision

    def as_dict(self):
        return self.__dict__


class BasicEnvironment(object):
    def __init__(self, suncgDatasetRoot=None, size=(256, 256)):

        # Create default agent
        self.agent = Agent('agent')

        self.physicWorld = Panda3dBulletPhysicWorld(suncgDatasetRoot)
        self.physicWorld.addAgentToScene(self.agent, radius=0.1, height=1.6, mass=60.0, mode='capsule')

        self.renderWorld = Panda3dRenderWorld(size, shadowing=False, showCeiling=False)
        self.renderWorld.addDefaultLighting()
        self.renderWorld.addAgentToScene(self.agent)

        self.acousticWorld = EvertAcousticWorld(samplingRate=16000, maximumOrder=2, materialAbsorption=False, frequencyDependent=False, showCeiling=False)
        self.acousticWorld.addAgentToScene(self.agent)
        
        self.worlds = {
            "physics": self.physicWorld,
            "render": self.renderWorld,
            "acoustics": self.acousticWorld,
        }

        self.labeledNavMap = None
        self.occupancyMapCoord = None

    def destroy(self):
        if self.renderWorld is not None:
            self.renderWorld.destroy()

    def loadHouse(self, house):
        # FIXME: find out why the acoustic engine needs to be last, otherwise the positions are not synchronized
        self.worlds["physics"].addHouseToScene(house)
        self.worlds["render"].addHouseToScene(house)
        self.worlds["acoustics"].addHouseToScene(house)

    def generateOccupancyMap(self, minRegionArea=10.0, z=1.0, precision=0.1):

        cellArea = precision ** 2
        occupancyMap, self.occupancyMapCoord = self.physicWorld.calculate2dNavigationMap(self.agent, z=z,
                                                                                         precision=precision)
        self.labeledNavMap = extractAllRegions(occupancyMap)

        # Post-processing by removing regions smaller than the threshold
        curValidRegionId = 1
        nbRegions = int(np.max(self.labeledNavMap))
        regionAreas = []
        for r in range(1, nbRegions + 1):
            nbCells = np.count_nonzero(self.labeledNavMap == r)
            regionArea = nbCells * cellArea
            if regionArea < minRegionArea:
                self.labeledNavMap[self.labeledNavMap == r] = -1
            else:
                self.labeledNavMap[self.labeledNavMap == r] = curValidRegionId
                regionAreas.append(regionArea)
                curValidRegionId += 1

        assert np.all(self.labeledNavMap != 0)
        logger.debug('Number of valid regions found in occupancy map: %d' % (curValidRegionId - 1))

        return self.labeledNavMap, self.occupancyMapCoord

    def generateSpawnPositions(self, n, minRegionArea=10.0, z=1.0, precision=0.1):

        self.generateOccupancyMap(minRegionArea, z, precision)

        # Calculate region areas
        cellArea = precision ** 2
        nbRegions = int(np.max(self.labeledNavMap))
        regionAreas = []
        for r in range(1, nbRegions + 1):
            nbCells = np.count_nonzero(self.labeledNavMap == r)
            regionArea = nbCells * cellArea
            regionAreas.append(regionArea)
        regionAreas = np.array(regionAreas)

        # Calculate relative ratios between regions, and create a cummulative array
        regionAreas /= np.sum(regionAreas)
        s = np.cumsum(regionAreas)

        positions = []
        for _ in range(n):

            if len(s) > 1:
                # Select a random region
                r = np.random.random()
                idx = np.argwhere(s < r)[-1] + 1
                selRegionId = idx + 1
            else:
                selRegionId = 1

            # Select a random cell of the region
            coord = np.argwhere(self.labeledNavMap == selRegionId)
            r = np.random.randint(low=0, high=len(coord))

            i, j = coord[r]
            x, y = self.occupancyMapCoord[i, j]
            positions.append([x, y])

        occupancyMap = np.zeros(self.labeledNavMap.shape, dtype=np.float)
        occupancyMap[self.labeledNavMap == -1] = 1.0

        return occupancyMap, self.occupancyMapCoord, np.array(positions)

    def getObservation(self):
        position = self.agent.getPosition()
        orientation = self.agent.getOrientation()
        image = self.renderWorld.getRgbImage()
        collision = self.agent.isCollision()

        return Observation(position, orientation, image, collision)

    def step(self):
        # NOTE: we should always update the physics first
        self.worlds["physics"].step()
        self.worlds["render"].step()
        self.worlds["acoustics"].step()
