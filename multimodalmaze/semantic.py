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
import logging
import scipy.ndimage
import numpy as np

from string import digits

from panda3d.core import GeomVertexReader, GeomTristrips, GeomTriangles, ColorAttrib, TextureAttrib, TransparencyAttrib,\
    LVecBase4f, LVecBase3f
from panda3d.core import Loader, LoaderOptions, NodePath, Filename

from multimodalmaze.suncg import ModelCategoryMapping

logger = logging.getLogger(__name__)

def getSurfaceAreaFromGeom(geom):
    
    totalArea = 0.0
    for k in range(geom.getNumPrimitives()):
        prim = geom.getPrimitive(k)
        vdata = geom.getVertexData()
        vertex = GeomVertexReader(vdata, 'vertex')
        assert isinstance(prim, (GeomTristrips, GeomTriangles))
         
        # Decompose into triangles
        prim = prim.decompose()
        for p in range(prim.getNumPrimitives()):
            s = prim.getPrimitiveStart(p)
            e = prim.getPrimitiveEnd(p)
             
            triPts = []
            for i in range(s, e):
                vi = prim.getVertex(i)
                vertex.setRow(vi)
                v = vertex.getData3f()
                triPts.append([v.x, v.y, v.z])
            triPts = np.array(triPts)

            # calculate the semi-perimeter and area
            a = np.linalg.norm(triPts[0] - triPts[1], 2)
            b = np.linalg.norm(triPts[1] - triPts[2], 2)
            c = np.linalg.norm(triPts[2] - triPts[0], 2)
            s = (a + b + c) / 2
            area = (s*(s-a)*(s-b)*(s-c)) ** 0.5
            totalArea += area

    return totalArea

def getColorAttributesFromModel(model):
    areas = []
    rgbColors = []
    textures = []
    transparencies = []
    for nodePath in model.findAllMatches('**/+GeomNode'):
        geomNode = nodePath.node()
        
        for n in range(geomNode.getNumGeoms()):
            state = geomNode.getGeomState(n)
        
            rgbColor = None
            texture = None
            isTransparent = False
            if state.hasAttrib(TextureAttrib.getClassType()):
                texAttr = state.getAttrib(TextureAttrib.getClassType())
                tex = texAttr.getTexture()
                
                # Load texture image from file and compute average color
                texFilename = str(tex.getFullpath())
                img = scipy.ndimage.imread(texFilename)

                texture = os.path.splitext(os.path.basename(texFilename))[0]
                
                #TODO: handle black-and-white and RGBA texture
                assert img.dtype == np.uint8
                assert img.ndim == 3 and img.shape[-1] == 3
                
                rgbColor = (np.mean(img, axis=(0,1)) / 255.0).tolist()

            elif state.hasAttrib(ColorAttrib.getClassType()):
                colorAttr = state.getAttrib(ColorAttrib.getClassType())
                color = colorAttr.getColor()
                
                if isinstance(color, LVecBase4f):
                    rgbColor= [color[0], color[1], color[2]]
                    alpha = color[3]
                    
                    if state.hasAttrib(TransparencyAttrib.getClassType()):
                        transAttr = state.getAttrib(TransparencyAttrib.getClassType())
                        if transAttr.getMode() != TransparencyAttrib.MNone and alpha < 1.0:
                            isTransparent = True
                    elif alpha < 1.0:
                        isTransparent = True
                        
                elif isinstance(color, LVecBase3f):
                    rgbColor= [color[0], color[1], color[2]]
                else:
                    raise Exception('Unsupported color class type: %s' % (color.__class__.__name__))
                
            rgbColors.append(rgbColor)
            transparencies.append(isTransparent)
        
            geom = geomNode.getGeom(n)
            area = getSurfaceAreaFromGeom(geom)
            areas.append(area)
            textures.append(texture)
            
    areas = np.array(areas)
    areas /= np.sum(areas)
            
    return areas, rgbColors, transparencies, textures

class MaterialTable(object):
    
    materials = {
        "beam":"wood",
        "bricks":"bricks",
        "carpet":"carpet",
        "decorstone":"decoration stone",
        "facing_stone":"facing stone",
        #"grass":"grass",
        #"ground":"ground",
        "laminate":"laminate",
        "leather":"leather",
        "linen":"linen",
        "lnm":"linoleum",
        "paneling":"wood",
        "path":"stone",
        "potang":"potang",
        "prkt":"wood",
        "solo":"wood",
        "stone":"stone",
        "stucco":"stucco",
        "textile":"textile",
        "tile":"tile",
        "wallp":"wallpaper",
        "wood":"wood",
        }
    
    @staticmethod
    def getMaterialNameFromObject(obj, thresholdRelArea=0.2):
        
        # Load the model
        loader = Loader.getGlobalPtr()
        loaderOptions = LoaderOptions()
        node = loader.loadSync(Filename(obj.modelFilename), loaderOptions)
        if node is not None:
            nodePath = NodePath(node)
        else:
            raise IOError('Could not load model file: %s' % (obj.modelFilename))
        
        # Get the list of materials
        areas, _, _, textures = getColorAttributesFromModel(nodePath)

        # Get the most dominant material based on threshold on relative surface area
        materialDescriptions = []
        for area, texture in zip(areas, textures):
            if texture is None or not area >= thresholdRelArea: continue
            
            # Remove any digits
            texture = texture.translate(None, digits)
            
            # Remove trailing underscores
            texture = texture.rstrip("_")
            
            #NOTE: handle many variations of textile and wood in SUNCG texture names
            if "textile" in texture:
                texture = "textile"
            if "wood" in texture:
                texture = "wood"
                
            if texture in MaterialTable.materials:
                textureName = MaterialTable.materials[texture]
                materialDescriptions.append(textureName)
            else:
                logger.debug('Unsupported texture basename for material semantics: %s' % (texture))
        
        # Remove duplicates (if any)
        materialDescriptions = list(set(materialDescriptions))

        # Unload model
        nodePath.removeNode()
        
        return materialDescriptions
    

class MaterialColorTable(object):
    
    # From: http://www.rapidtables.com/web/color/RGB_Color.htm
    BasicColorTable = {
        "black":[0,0,0],
        "blue":[0,0,255],
        "cyan":[0,255,255],
        "gray":[128,128,128],
        "green":[0,128,0],
        "lime":[0,255,0],
        "magenta":[255,0,255],
        "maroon":[128,0,0],
        "navy":[0,0,128],
        "olive":[128,128,0],
        "purple":[128,0,128],
        "red":[255,0,0],
        "silver":[192,192,192],
        "teal":[0,128,128],
        "white":[255,255,255],
        "yellow":[255,255,0],
    }
    
    # From: http://www.rapidtables.com/web/color/RGB_Color.htm
    AdvancedColorTable = {
        "alice blue":[240,248,255],
        "antique white":[250,235,215],
        "aqua":[0,255,255],
        "aqua marine":[127,255,212],
        "azure":[240,255,255],
        "beige":[245,245,220],
        "bisque":[255,228,196],
        "black":[0,0,0],
        "blanched almond":[255,235,205],
        "blue":[0,0,255],
        "blue violet":[138,43,226],
        "brown":[165,42,42],
        "burly wood":[222,184,135],
        "cadet blue":[95,158,160],
        "chart reuse":[127,255,0],
        "chocolate":[210,105,30],
        "coral":[255,127,80],
        "corn flower blue":[100,149,237],
        "corn silk":[255,248,220],
        "crimson":[220,20,60],
        "cyan":[0,255,255],
        "dark blue":[0,0,139],
        "dark cyan":[0,139,139],
        "dark golden rod":[184,134,11],
        "dark gray":[169,169,169],
        "dark green":[0,100,0],
        "dark khaki":[189,183,107],
        "dark magenta":[139,0,139],
        "dark olive green":[85,107,47],
        "dark orange":[255,140,0],
        "dark orchid":[153,50,204],
        "dark red":[139,0,0],
        "dark salmon":[233,150,122],
        "dark sea green":[143,188,143],
        "dark slate blue":[72,61,139],
        "dark slate gray":[47,79,79],
        "dark turquoise":[0,206,209],
        "dark violet":[148,0,211],
        "deep pink":[255,20,147],
        "deep sky blue":[0,191,255],
        "dim gray":[105,105,105],
        "dodger blue":[30,144,255],
        "firebrick":[178,34,34],
        "floral white":[255,250,240],
        "forest green":[34,139,34],
        "gainsboro":[220,220,220],
        "ghost white":[248,248,255],
        "gold":[255,215,0],
        "golden rod":[218,165,32],
        "gray":[128,128,128],
        "green":[0,128,0],
        "green yellow":[173,255,47],
        "honeydew":[240,255,240],
        "hot pink":[255,105,180],
        "indian red":[205,92,92],
        "indigo":[75,0,130],
        "ivory":[255,255,240],
        "khaki":[240,230,140],
        "lavender":[230,230,250],
        "lavender blush":[255,240,245],
        "lawn green":[124,252,0],
        "lemon chiffon":[255,250,205],
        "light blue":[173,216,230],
        "light coral":[240,128,128],
        "light cyan":[224,255,255],
        "light golden rod yellow":[250,250,210],
        "light gray":[211,211,211],
        "light green":[144,238,144],
        "light pink":[255,182,193],
        "light salmon":[255,160,122],
        "light sea green":[32,178,170],
        "light sky blue":[135,206,250],
        "light slate gray":[119,136,153],
        "light steel blue":[176,196,222],
        "light yellow":[255,255,224],
        "lime":[0,255,0],
        "lime green":[50,205,50],
        "linen":[250,240,230],
        "magenta":[255,0,255],
        "maroon":[128,0,0],
        "medium aqua marine":[102,205,170],
        "medium blue":[0,0,205],
        "medium orchid":[186,85,211],
        "medium purple":[147,112,219],
        "medium sea green":[60,179,113],
        "medium slate blue":[123,104,238],
        "medium spring green":[0,250,154],
        "medium turquoise":[72,209,204],
        "medium violet red":[199,21,133],
        "midnight blue":[25,25,112],
        "mint cream":[245,255,250],
        "misty rose":[255,228,225],
        "moccasin":[255,228,181],
        "navajo white":[255,222,173],
        "navy":[0,0,128],
        "old lace":[253,245,230],
        "olive":[128,128,0],
        "olive drab":[107,142,35],
        "orange":[255,165,0],
        "orange red":[255,69,0],
        "orchid":[218,112,214],
        "pale golden rod":[238,232,170],
        "pale green":[152,251,152],
        "pale turquoise":[175,238,238],
        "pale violet red":[219,112,147],
        "papaya whip":[255,239,213],
        "peach puff":[255,218,185],
        "peru":[205,133,63],
        "pink":[255,192,203],
        "plum":[221,160,221],
        "powder blue":[176,224,230],
        "purple":[128,0,128],
        "red":[255,0,0],
        "rosy brown":[188,143,143],
        "royal blue":[65,105,225],
        "saddle brown":[139,69,19],
        "salmon":[250,128,114],
        "sandy brown":[244,164,96],
        "sea green":[46,139,87],
        "sea shell":[255,245,238],
        "sienna":[160,82,45],
        "silver":[192,192,192],
        "sky blue":[135,206,235],
        "slate blue":[106,90,205],
        "slate gray":[112,128,144],
        "snow":[255,250,250],
        "spring green":[0,255,127],
        "steel blue":[70,130,180],
        "tan":[210,180,140],
        "teal":[0,128,128],
        "thistle":[216,191,216],
        "tomato":[255,99,71],
        "turquoise":[64,224,208],
        "violet":[238,130,238],
        "wheat":[245,222,179],
        "white":[255,255,255],
        "white smoke":[245,245,245],
        "yellow":[255,255,0],
        "yellow green":[154,205,50],
    }
    
    # From: https://xkcd.com/color/rgb.txt
    XkcdColorTable = {
        "acid green":[143,254,9],
        "adobe":[189,108,72],
        "algae":[84,172,104],
        "algae green":[33,195,111],
        "almost black":[7,13,13],
        "amber":[254,179,8],
        "amethyst":[155,95,192],
        "apple":[110,203,60],
        "apple green":[118,205,38],
        "apricot":[255,177,109],
        "aqua":[19,234,201],
        "aqua blue":[2,216,233],
        "aqua green":[18,225,147],
        "aquamarine":[4,216,178],
        "aqua marine":[46,232,187],
        "army green":[75,93,22],
        "asparagus":[119,171,86],
        "aubergine":[61,7,52],
        "auburn":[154,48,1],
        "avocado":[144,177,52],
        "avocado green":[135,169,34],
        "azul":[29,93,236],
        "azure":[6,154,243],
        "baby blue":[162,207,254],
        "baby green":[140,255,158],
        "baby pink":[255,183,206],
        "baby poo":[171,144,4],
        "baby poop":[147,124,0],
        "baby poop green":[143,152,5],
        "baby puke green":[182,196,6],
        "baby purple":[202,155,247],
        "baby shit brown":[173,144,13],
        "baby shit green":[136,151,23],
        "banana":[255,255,126],
        "banana yellow":[250,254,75],
        "barbie pink":[254,70,165],
        "barf green":[148,172,2],
        "barney":[172,29,184],
        "barney purple":[160,4,152],
        "battleship grey":[107,124,133],
        "beige":[230,218,166],
        "berry":[153,15,75],
        "bile":[181,195,6],
        "black":[0,0,0],
        "bland":[175,168,139],
        "blood":[119,0,1],
        "blood orange":[254,75,3],
        "blood red":[152,0,2],
        "blue":[3,67,223],
        "blueberry":[70,65,150],
        "blue blue":[34,66,199],
        "bluegreen":[1,122,121],
        "blue/green":[15,155,142],
        "blue green":[19,126,109],
        "blue/grey":[117,141,163],
        "bluegrey":[133,163,178],
        "blue grey":[96,124,142],
        "blue purple":[87,41,206],
        "blue/purple":[90,6,239],
        "blue violet":[93,6,233],
        "blue with a hint of purple":[83,60,198],
        "bluey green":[43,177,121],
        "bluey grey":[137,160,176],
        "bluey purple":[98,65,199],
        "bluish":[41,118,187],
        "bluish green":[16,166,116],
        "bluish grey":[116,139,151],
        "bluish purple":[112,59,231],
        "blurple":[85,57,204],
        "blush":[242,158,142],
        "blush pink":[254,130,140],
        "booger":[155,181,60],
        "booger green":[150,180,3],
        "bordeaux":[123,0,44],
        "boring green":[99,179,101],
        "bottle green":[4,74,5],
        "brick":[160,54,35],
        "brick orange":[193,74,9],
        "brick red":[143,20,2],
        "bright aqua":[11,249,234],
        "bright blue":[1,101,252],
        "bright cyan":[65,253,254],
        "bright green":[1,255,7],
        "bright lavender":[199,96,255],
        "bright light blue":[38,247,253],
        "bright light green":[45,254,84],
        "bright lilac":[201,94,251],
        "bright lime":[135,253,5],
        "bright lime green":[101,254,8],
        "bright magenta":[255,8,232],
        "bright olive":[156,187,4],
        "bright orange":[255,91,0],
        "bright pink":[254,1,177],
        "bright purple":[190,3,253],
        "bright red":[255,0,13],
        "bright sea green":[5,255,166],
        "bright sky blue":[2,204,254],
        "bright teal":[1,249,198],
        "bright turquoise":[15,254,249],
        "bright violet":[173,10,253],
        "bright yellow":[255,253,1],
        "bright yellow green":[157,255,0],
        "british racing green":[5,72,13],
        "bronze":[168,121,0],
        "brown":[101,55,0],
        "brown green":[112,108,17],
        "brown grey":[141,132,104],
        "brownish":[156,109,87],
        "brownish green":[106,110,9],
        "brownish grey":[134,119,95],
        "brownish orange":[203,119,35],
        "brownish pink":[194,126,121],
        "brownish purple":[118,66,78],
        "brownish red":[158,54,35],
        "brownish yellow":[201,176,3],
        "brown orange":[185,105,2],
        "brown red":[146,43,5],
        "brown yellow":[178,151,5],
        "browny green":[111,108,10],
        "browny orange":[202,107,2],
        "bruise":[126,64,113],
        "bubblegum":[255,108,181],
        "bubblegum pink":[254,131,204],
        "bubble gum pink":[255,105,175],
        "buff":[254,246,158],
        "burgundy":[97,0,35],
        "burnt orange":[192,78,1],
        "burnt red":[159,35,5],
        "burnt siena":[183,82,3],
        "burnt sienna":[176,78,15],
        "burnt umber":[160,69,14],
        "burnt yellow":[213,171,9],
        "burple":[104,50,227],
        "butter":[255,255,129],
        "butterscotch":[253,177,71],
        "butter yellow":[255,253,116],
        "cadet blue":[78,116,150],
        "camel":[198,159,89],
        "camo":[127,143,78],
        "camo green":[82,101,37],
        "camouflage green":[75,97,19],
        "canary":[253,255,99],
        "canary yellow":[255,254,64],
        "candy pink":[255,99,233],
        "caramel":[175,111,9],
        "carmine":[157,2,22],
        "carnation":[253,121,143],
        "carnation pink":[255,127,167],
        "carolina blue":[138,184,254],
        "celadon":[190,253,183],
        "celery":[193,253,149],
        "cement":[165,163,145],
        "cerise":[222,12,98],
        "cerulean":[4,133,209],
        "cerulean blue":[5,110,238],
        "charcoal":[52,56,55],
        "charcoal grey":[60,65,66],
        "chartreuse":[193,248,10],
        "cherry":[207,2,52],
        "cherry red":[247,2,42],
        "chestnut":[116,40,2],
        "chocolate":[61,28,2],
        "chocolate brown":[65,25,0],
        "cinnamon":[172,79,6],
        "claret":[104,0,24],
        "clay":[182,106,80],
        "clay brown":[178,113,61],
        "clear blue":[36,122,253],
        "cloudy blue":[172,194,217],
        "cobalt":[30,72,143],
        "cobalt blue":[3,10,167],
        "cocoa":[135,95,66],
        "coffee":[166,129,76],
        "cool blue":[73,132,184],
        "cool green":[51,184,100],
        "cool grey":[149,163,166],
        "copper":[182,99,37],
        "coral":[252,90,80],
        "coral pink":[255,97,99],
        "cornflower":[106,121,247],
        "cornflower blue":[81,112,215],
        "cranberry":[158,0,58],
        "cream":[255,255,194],
        "creme":[255,255,182],
        "crimson":[140,0,15],
        "custard":[255,253,120],
        "cyan":[0,255,255],
        "dandelion":[254,223,8],
        "dark":[27,36,49],
        "dark aqua":[5,105,107],
        "dark aquamarine":[1,115,113],
        "dark beige":[172,147,98],
        "dark blue":[0,3,91],
        "darkblue":[3,7,100],
        "dark blue green":[0,82,73],
        "dark blue grey":[31,59,77],
        "dark brown":[52,28,2],
        "dark coral":[207,82,78],
        "dark cream":[255,243,154],
        "dark cyan":[10,136,138],
        "dark forest green":[0,45,4],
        "dark fuchsia":[157,7,89],
        "dark gold":[181,148,16],
        "dark grass green":[56,128,4],
        "dark green":[3,53,0],
        "darkgreen":[5,73,7],
        "dark green blue":[31,99,87],
        "dark grey":[54,55,55],
        "dark grey blue":[41,70,91],
        "dark hot pink":[217,1,102],
        "dark indigo":[31,9,84],
        "darkish blue":[1,65,130],
        "darkish green":[40,124,55],
        "darkish pink":[218,70,125],
        "darkish purple":[117,25,115],
        "darkish red":[169,3,8],
        "dark khaki":[155,143,85],
        "dark lavender":[133,103,152],
        "dark lilac":[156,109,165],
        "dark lime":[132,183,1],
        "dark lime green":[126,189,1],
        "dark magenta":[150,0,86],
        "dark maroon":[60,0,8],
        "dark mauve":[135,76,98],
        "dark mint":[72,192,114],
        "dark mint green":[32,192,115],
        "dark mustard":[168,137,5],
        "dark navy":[0,4,53],
        "dark navy blue":[0,2,46],
        "dark olive":[55,62,2],
        "dark olive green":[60,77,3],
        "dark orange":[198,81,2],
        "dark pastel green":[86,174,87],
        "dark peach":[222,126,93],
        "dark periwinkle":[102,95,209],
        "dark pink":[203,65,107],
        "dark plum":[63,1,44],
        "dark purple":[53,6,62],
        "dark red":[132,0,0],
        "dark rose":[181,72,93],
        "dark royal blue":[2,6,111],
        "dark sage":[89,133,86],
        "dark salmon":[200,90,83],
        "dark sand":[168,143,89],
        "dark seafoam":[31,181,122],
        "dark seafoam green":[62,175,118],
        "dark sea green":[17,135,93],
        "dark sky blue":[68,142,228],
        "dark slate blue":[33,71,97],
        "dark tan":[175,136,74],
        "dark taupe":[127,104,78],
        "dark teal":[1,77,78],
        "dark turquoise":[4,92,90],
        "dark violet":[52,1,63],
        "dark yellow":[213,182,10],
        "dark yellow green":[114,143,2],
        "deep aqua":[8,120,127],
        "deep blue":[4,2,115],
        "deep brown":[65,2,0],
        "deep green":[2,89,15],
        "deep lavender":[141,94,183],
        "deep lilac":[150,110,189],
        "deep magenta":[160,2,92],
        "deep orange":[220,77,1],
        "deep pink":[203,1,98],
        "deep purple":[54,1,63],
        "deep red":[154,2,0],
        "deep rose":[199,71,103],
        "deep sea blue":[1,84,130],
        "deep sky blue":[13,117,248],
        "deep teal":[0,85,90],
        "deep turquoise":[1,115,116],
        "deep violet":[73,6,72],
        "denim":[59,99,140],
        "denim blue":[59,91,146],
        "desert":[204,173,96],
        "diarrhea":[159,131,3],
        "dirt":[138,110,69],
        "dirt brown":[131,101,57],
        "dirty blue":[63,130,157],
        "dirty green":[102,126,44],
        "dirty orange":[200,118,6],
        "dirty pink":[202,123,128],
        "dirty purple":[115,74,101],
        "dirty yellow":[205,197,10],
        "dodger blue":[62,130,252],
        "drab":[130,131,68],
        "drab green":[116,149,81],
        "dried blood":[75,1,1],
        "duck egg blue":[195,251,244],
        "dull blue":[73,117,156],
        "dull brown":[135,110,75],
        "dull green":[116,166,98],
        "dull orange":[216,134,59],
        "dull pink":[213,134,157],
        "dull purple":[132,89,126],
        "dull red":[187,63,63],
        "dull teal":[95,158,143],
        "dull yellow":[238,220,91],
        "dusk":[78,84,129],
        "dusk blue":[38,83,141],
        "dusky blue":[71,95,148],
        "dusky pink":[204,122,139],
        "dusky purple":[137,91,123],
        "dusky rose":[186,104,115],
        "dust":[178,153,110],
        "dusty blue":[90,134,173],
        "dusty green":[118,169,115],
        "dusty lavender":[172,134,168],
        "dusty orange":[240,131,58],
        "dusty pink":[213,138,148],
        "dusty purple":[130,95,135],
        "dusty red":[185,72,78],
        "dusty rose":[192,115,122],
        "dusty teal":[76,144,133],
        "earth":[162,101,62],
        "easter green":[140,253,126],
        "easter purple":[192,113,254],
        "ecru":[254,255,202],
        "eggplant":[56,8,53],
        "eggplant purple":[67,5,65],
        "egg shell":[255,252,196],
        "eggshell":[255,255,212],
        "eggshell blue":[196,255,247],
        "electric blue":[6,82,255],
        "electric green":[33,252,13],
        "electric lime":[168,255,4],
        "electric pink":[255,4,144],
        "electric purple":[170,35,255],
        "emerald":[1,160,73],
        "emerald green":[2,143,30],
        "evergreen":[5,71,42],
        "faded blue":[101,140,187],
        "faded green":[123,178,116],
        "faded orange":[240,148,77],
        "faded pink":[222,157,172],
        "faded purple":[145,110,153],
        "faded red":[211,73,78],
        "faded yellow":[254,255,127],
        "fawn":[207,175,123],
        "fern":[99,169,80],
        "fern green":[84,141,68],
        "fire engine red":[254,0,2],
        "flat blue":[60,115,168],
        "flat green":[105,157,76],
        "fluorescent green":[8,255,8],
        "fluro green":[10,255,2],
        "foam green":[144,253,169],
        "forest":[11,85,9],
        "forest green":[6,71,12],
        "forrest green":[21,68,6],
        "french blue":[67,107,173],
        "fresh green":[105,216,79],
        "frog green":[88,188,8],
        "fuchsia":[237,13,217],
        "gold":[219,180,12],
        "golden":[245,191,3],
        "golden brown":[178,122,1],
        "golden rod":[249,188,8],
        "goldenrod":[250,194,5],
        "golden yellow":[254,198,21],
        "grape":[108,52,97],
        "grapefruit":[253,89,86],
        "grape purple":[93,20,81],
        "grass":[92,172,45],
        "grass green":[63,155,11],
        "grassy green":[65,156,3],
        "green":[21,176,26],
        "green apple":[94,220,31],
        "green/blue":[1,192,141],
        "greenblue":[35,196,139],
        "green blue":[6,180,139],
        "green brown":[84,78,3],
        "green grey":[119,146,111],
        "greenish":[64,163,104],
        "greenish beige":[201,209,121],
        "greenish blue":[11,139,135],
        "greenish brown":[105,97,18],
        "greenish cyan":[42,254,183],
        "greenish grey":[150,174,141],
        "greenish tan":[188,203,122],
        "greenish teal":[50,191,132],
        "greenish turquoise":[0,251,176],
        "greenish yellow":[205,253,2],
        "green teal":[12,181,119],
        "greeny blue":[66,179,149],
        "greeny brown":[105,96,6],
        "green/yellow":[181,206,8],
        "green yellow":[201,255,39],
        "greeny grey":[126,160,122],
        "greeny yellow":[198,248,8],
        "grey":[146,149,145],
        "grey/blue":[100,125,142],
        "grey blue":[107,139,164],
        "greyblue":[119,161,181],
        "grey brown":[127,112,83],
        "grey green":[120,155,115],
        "grey/green":[134,161,125],
        "greyish":[168,164,149],
        "greyish blue":[94,129,157],
        "greyish brown":[122,106,79],
        "greyish green":[130,166,125],
        "greyish pink":[200,141,148],
        "greyish purple":[136,113,145],
        "greyish teal":[113,159,145],
        "grey pink":[195,144,155],
        "grey purple":[130,109,140],
        "grey teal":[94,155,138],
        "gross green":[160,191,22],
        "gunmetal":[83,98,103],
        "hazel":[142,118,24],
        "heather":[164,132,172],
        "heliotrope":[217,79,245],
        "highlighter green":[27,252,6],
        "hospital green":[155,229,170],
        "hot green":[37,255,41],
        "hot magenta":[245,4,201],
        "hot pink":[255,2,141],
        "hot purple":[203,0,245],
        "hunter green":[11,64,8],
        "ice":[214,255,250],
        "ice blue":[215,255,254],
        "icky green":[143,174,34],
        "indian red":[133,14,4],
        "indigo":[56,2,130],
        "indigo blue":[58,24,177],
        "iris":[98,88,196],
        "irish green":[1,149,41],
        "ivory":[255,255,203],
        "jade":[31,167,116],
        "jade green":[43,175,106],
        "jungle green":[4,130,67],
        "kelley green":[0,147,55],
        "kelly green":[2,171,46],
        "kermit green":[92,178,0],
        "key lime":[174,255,110],
        "khaki":[170,166,98],
        "khaki green":[114,134,57],
        "kiwi":[156,239,67],
        "kiwi green":[142,229,63],
        "lavender":[199,159,239],
        "lavender blue":[139,136,248],
        "lavender pink":[221,133,215],
        "lawn green":[77,164,9],
        "leaf":[113,170,52],
        "leaf green":[92,169,4],
        "leafy green":[81,183,59],
        "leather":[172,116,52],
        "lemon":[253,255,82],
        "lemon green":[173,248,2],
        "lemon lime":[191,254,40],
        "lemon yellow":[253,255,56],
        "lichen":[143,182,123],
        "light aqua":[140,255,219],
        "light aquamarine":[123,253,199],
        "light beige":[255,254,182],
        "lightblue":[123,200,246],
        "light blue":[149,208,252],
        "light blue green":[126,251,179],
        "light blue grey":[183,201,226],
        "light bluish green":[118,253,168],
        "light bright green":[83,254,92],
        "light brown":[173,129,80],
        "light burgundy":[168,65,91],
        "light cyan":[172,255,252],
        "light eggplant":[137,69,133],
        "lighter green":[117,253,99],
        "lighter purple":[165,90,244],
        "light forest green":[79,145,83],
        "light gold":[253,220,92],
        "light grass green":[154,247,100],
        "lightgreen":[118,255,123],
        "light green":[150,249,123],
        "light green blue":[86,252,162],
        "light greenish blue":[99,247,180],
        "light grey":[216,220,214],
        "light grey blue":[157,188,212],
        "light grey green":[183,225,161],
        "light indigo":[109,90,207],
        "lightish blue":[61,122,253],
        "lightish green":[97,225,96],
        "lightish purple":[165,82,230],
        "lightish red":[254,47,74],
        "light khaki":[230,242,162],
        "light lavendar":[239,192,254],
        "light lavender":[223,197,254],
        "light light blue":[202,255,251],
        "light light green":[200,255,176],
        "light lilac":[237,200,255],
        "light lime":[174,253,108],
        "light lime green":[185,255,102],
        "light magenta":[250,95,247],
        "light maroon":[162,72,87],
        "light mauve":[194,146,161],
        "light mint":[182,255,187],
        "light mint green":[166,251,178],
        "light moss green":[166,200,117],
        "light mustard":[247,213,96],
        "light navy":[21,80,132],
        "light navy blue":[46,90,136],
        "light neon green":[78,253,84],
        "light olive":[172,191,105],
        "light olive green":[164,190,92],
        "light orange":[253,170,72],
        "light pastel green":[178,251,165],
        "light peach":[255,216,177],
        "light pea green":[196,254,130],
        "light periwinkle":[193,198,252],
        "light pink":[255,209,223],
        "light plum":[157,87,131],
        "light purple":[191,119,246],
        "light red":[255,71,76],
        "light rose":[255,197,203],
        "light royal blue":[58,46,254],
        "light sage":[188,236,172],
        "light salmon":[254,169,147],
        "light seafoam":[160,254,191],
        "light seafoam green":[167,255,181],
        "light sea green":[152,246,176],
        "light sky blue":[198,252,255],
        "light tan":[251,238,172],
        "light teal":[144,228,193],
        "light turquoise":[126,244,204],
        "light urple":[179,111,246],
        "light violet":[214,180,252],
        "light yellow":[255,254,122],
        "light yellow green":[204,253,127],
        "light yellowish green":[194,255,137],
        "lilac":[206,162,253],
        "liliac":[196,142,253],
        "lime":[170,255,50],
        "lime green":[137,254,5],
        "lime yellow":[208,254,29],
        "lipstick":[213,23,78],
        "lipstick red":[192,2,47],
        "macaroni and cheese":[239,180,53],
        "magenta":[194,0,120],
        "mahogany":[74,1,0],
        "maize":[244,208,84],
        "mango":[255,166,43],
        "manilla":[255,250,134],
        "marigold":[252,192,6],
        "marine":[4,46,96],
        "marine blue":[1,56,106],
        "maroon":[101,0,33],
        "mauve":[174,113,129],
        "medium blue":[44,111,187],
        "medium brown":[127,81,18],
        "medium green":[57,173,72],
        "medium grey":[125,127,124],
        "medium pink":[243,97,150],
        "medium purple":[158,67,162],
        "melon":[255,120,85],
        "merlot":[115,0,57],
        "metallic blue":[79,115,142],
        "mid blue":[39,106,179],
        "mid green":[80,167,71],
        "midnight":[3,1,45],
        "midnight blue":[2,0,53],
        "midnight purple":[40,1,55],
        "military green":[102,124,62],
        "milk chocolate":[127,78,30],
        "mint":[159,254,176],
        "mint green":[143,255,159],
        "minty green":[11,247,125],
        "mocha":[157,118,81],
        "moss":[118,153,88],
        "moss green":[101,139,56],
        "mossy green":[99,139,39],
        "mud":[115,92,18],
        "mud brown":[96,70,15],
        "muddy brown":[136,104,6],
        "muddy green":[101,116,50],
        "muddy yellow":[191,172,5],
        "mud green":[96,102,2],
        "mulberry":[146,10,78],
        "murky green":[108,122,14],
        "mushroom":[186,158,136],
        "mustard":[206,179,1],
        "mustard brown":[172,126,4],
        "mustard green":[168,181,4],
        "mustard yellow":[210,189,10],
        "muted blue":[59,113,159],
        "muted green":[95,160,82],
        "muted pink":[209,118,143],
        "muted purple":[128,91,135],
        "nasty green":[112,178,63],
        "navy":[1,21,62],
        "navy blue":[0,17,70],
        "navy green":[53,83,10],
        "neon blue":[4,217,255],
        "neon green":[12,255,12],
        "neon pink":[254,1,154],
        "neon purple":[188,19,254],
        "neon red":[255,7,58],
        "neon yellow":[207,255,4],
        "nice blue":[16,122,176],
        "night blue":[4,3,72],
        "ocean":[1,123,146],
        "ocean blue":[3,113,156],
        "ocean green":[61,153,115],
        "ocher":[191,155,12],
        "ochre":[191,144,5],
        "ocre":[198,156,4],
        "off blue":[86,132,174],
        "off green":[107,163,83],
        "off white":[255,255,228],
        "off yellow":[241,243,63],
        "old pink":[199,121,134],
        "old rose":[200,127,137],
        "olive":[110,117,14],
        "olive brown":[100,84,3],
        "olive drab":[111,118,50],
        "olive green":[103,122,4],
        "olive yellow":[194,183,9],
        "orange":[249,115,6],
        "orange brown":[190,100,0],
        "orangeish":[253,141,73],
        "orange pink":[255,111,82],
        "orange red":[253,65,30],
        "orangered":[254,66,15],
        "orangey brown":[177,96,2],
        "orange yellow":[255,173,1],
        "orangey red":[250,66,36],
        "orangey yellow":[253,185,21],
        "orangish":[252,130,74],
        "orangish brown":[178,95,3],
        "orangish red":[244,54,5],
        "orchid":[200,117,196],
        "pale":[255,249,208],
        "pale aqua":[184,255,235],
        "pale blue":[208,254,254],
        "pale brown":[177,145,110],
        "pale cyan":[183,255,250],
        "pale gold":[253,222,108],
        "pale green":[199,253,181],
        "pale grey":[253,253,254],
        "pale lavender":[238,207,254],
        "pale light green":[177,252,153],
        "pale lilac":[228,203,255],
        "pale lime":[190,253,115],
        "pale lime green":[177,255,101],
        "pale magenta":[215,103,173],
        "pale mauve":[254,208,252],
        "pale olive":[185,204,129],
        "pale olive green":[177,210,123],
        "pale orange":[255,167,86],
        "pale peach":[255,229,173],
        "pale pink":[255,207,220],
        "pale purple":[183,144,212],
        "pale red":[217,84,77],
        "pale rose":[253,193,197],
        "pale salmon":[255,177,154],
        "pale sky blue":[189,246,254],
        "pale teal":[130,203,178],
        "pale turquoise":[165,251,213],
        "pale violet":[206,174,250],
        "pale yellow":[255,255,132],
        "parchment":[254,252,175],
        "pastel blue":[162,191,254],
        "pastel green":[176,255,157],
        "pastel orange":[255,150,79],
        "pastel pink":[255,186,205],
        "pastel purple":[202,160,255],
        "pastel red":[219,88,86],
        "pastel yellow":[255,254,113],
        "pea":[164,191,32],
        "peach":[255,176,124],
        "peachy pink":[255,154,138],
        "peacock blue":[1,103,149],
        "pea green":[142,171,18],
        "pear":[203,248,95],
        "pea soup":[146,153,1],
        "pea soup green":[148,166,23],
        "periwinkle":[142,130,254],
        "periwinkle blue":[143,153,251],
        "perrywinkle":[143,140,231],
        "petrol":[0,95,106],
        "pig pink":[231,142,165],
        "pine":[43,93,52],
        "pine green":[10,72,30],
        "pink":[255,129,192],
        "pinkish":[212,106,126],
        "pinkish brown":[177,114,97],
        "pinkish grey":[200,172,169],
        "pinkish orange":[255,114,76],
        "pinkish purple":[214,72,215],
        "pinkish red":[241,12,69],
        "pinkish tan":[217,155,130],
        "pink purple":[219,75,218],
        "pink/purple":[239,29,231],
        "pink red":[245,5,79],
        "pinky":[252,134,170],
        "pinky purple":[201,76,190],
        "pinky red":[252,38,71],
        "piss yellow":[221,214,24],
        "pistachio":[192,250,139],
        "plum":[88,15,65],
        "plum purple":[78,5,80],
        "poison green":[64,253,20],
        "poo":[143,115,3],
        "poo brown":[136,95,1],
        "poop":[127,94,0],
        "poop brown":[122,89,1],
        "poop green":[111,124,0],
        "powder blue":[177,209,252],
        "powder pink":[255,178,208],
        "primary blue":[8,4,249],
        "prussian blue":[0,69,119],
        "puce":[165,126,82],
        "puke":[165,165,2],
        "puke brown":[148,119,6],
        "puke green":[154,174,7],
        "puke yellow":[194,190,14],
        "pumpkin":[225,119,1],
        "pumpkin orange":[251,125,7],
        "pure blue":[2,3,226],
        "purple":[126,30,156],
        "purple/blue":[93,33,208],
        "purple blue":[99,45,233],
        "purple brown":[103,58,63],
        "purple grey":[134,111,133],
        "purpleish":[152,86,141],
        "purpleish blue":[97,64,239],
        "purpleish pink":[223,78,200],
        "purple/pink":[215,37,222],
        "purple pink":[224,63,216],
        "purple red":[153,1,71],
        "purpley":[135,86,228],
        "purpley blue":[95,52,231],
        "purpley grey":[148,126,148],
        "purpley pink":[200,60,185],
        "purplish":[148,86,140],
        "purplish blue":[96,30,249],
        "purplish brown":[107,66,71],
        "purplish grey":[122,104,127],
        "purplish pink":[206,93,174],
        "purplish red":[176,5,75],
        "purply":[152,63,178],
        "purply blue":[102,26,238],
        "purply pink":[240,117,230],
        "putty":[190,174,138],
        "racing green":[1,70,0],
        "radioactive green":[44,250,31],
        "raspberry":[176,1,73],
        "raw sienna":[154,98,0],
        "raw umber":[167,94,9],
        "really light blue":[212,255,255],
        "red":[229,0,0],
        "red brown":[139,46,22],
        "reddish":[196,66,64],
        "reddish brown":[127,43,10],
        "reddish grey":[153,117,112],
        "reddish orange":[248,72,28],
        "reddish pink":[254,44,84],
        "reddish purple":[145,9,81],
        "reddy brown":[110,16,5],
        "red orange":[253,60,6],
        "red pink":[250,42,85],
        "red purple":[130,7,71],
        "red violet":[158,1,104],
        "red wine":[140,0,52],
        "rich blue":[2,27,249],
        "rich purple":[114,0,88],
        "robin egg blue":[138,241,254],
        "robin's egg":[109,237,253],
        "robin's egg blue":[152,239,249],
        "rosa":[254,134,164],
        "rose":[207,98,117],
        "rose pink":[247,135,154],
        "rose red":[190,1,60],
        "rosy pink":[246,104,142],
        "rouge":[171,18,57],
        "royal":[12,23,147],
        "royal blue":[5,4,170],
        "royal purple":[75,0,110],
        "ruby":[202,1,71],
        "russet":[161,57,5],
        "rust":[168,60,9],
        "rust brown":[139,49,3],
        "rust orange":[196,85,8],
        "rust red":[170,39,4],
        "rusty orange":[205,89,9],
        "rusty red":[175,47,13],
        "saffron":[254,178,9],
        "sage":[135,174,115],
        "sage green":[136,179,120],
        "salmon":[255,121,108],
        "salmon pink":[254,123,124],
        "sand":[226,202,118],
        "sand brown":[203,165,96],
        "sandstone":[201,174,116],
        "sandy":[241,218,122],
        "sandy brown":[196,166,97],
        "sand yellow":[252,225,102],
        "sandy yellow":[253,238,115],
        "sap green":[92,139,21],
        "sapphire":[33,56,171],
        "scarlet":[190,1,25],
        "sea":[60,153,146],
        "sea blue":[4,116,149],
        "seafoam":[128,249,173],
        "seafoam blue":[120,209,182],
        "seafoam green":[122,249,171],
        "sea green":[83,252,161],
        "seaweed":[24,209,123],
        "seaweed green":[53,173,107],
        "sepia":[152,94,43],
        "shamrock":[1,180,76],
        "shamrock green":[2,193,77],
        "shit":[127,95,0],
        "shit brown":[123,88,4],
        "shit green":[117,128,0],
        "shocking pink":[254,2,162],
        "sick green":[157,185,44],
        "sickly green":[148,178,28],
        "sickly yellow":[208,228,41],
        "sienna":[169,86,30],
        "silver":[197,201,199],
        "sky":[130,202,252],
        "sky blue":[117,187,253],
        "slate":[81,101,114],
        "slate blue":[91,124,153],
        "slate green":[101,141,109],
        "slate grey":[89,101,109],
        "slime green":[153,204,4],
        "snot":[172,187,13],
        "snot green":[157,193,0],
        "soft blue":[100,136,234],
        "soft green":[111,194,118],
        "soft pink":[253,176,192],
        "soft purple":[166,111,181],
        "spearmint":[30,248,118],
        "spring green":[169,249,113],
        "spruce":[10,95,56],
        "squash":[242,171,21],
        "steel":[115,133,149],
        "steel blue":[90,125,154],
        "steel grey":[111,130,138],
        "stone":[173,165,135],
        "stormy blue":[80,123,156],
        "straw":[252,246,121],
        "strawberry":[251,41,67],
        "strong blue":[12,6,247],
        "strong pink":[255,7,137],
        "sunflower":[255,197,18],
        "sunflower yellow":[255,218,3],
        "sunny yellow":[255,249,23],
        "sunshine yellow":[255,253,55],
        "sun yellow":[255,223,34],
        "swamp":[105,131,57],
        "swamp green":[116,133,0],
        "tan":[209,178,111],
        "tan brown":[171,126,76],
        "tangerine":[255,148,8],
        "tan green":[169,190,112],
        "taupe":[185,162,129],
        "tea":[101,171,124],
        "tea green":[189,248,163],
        "teal":[2,147,134],
        "teal blue":[1,136,159],
        "teal green":[37,163,111],
        "tealish":[36,188,168],
        "tealish green":[12,220,115],
        "terracota":[203,104,67],
        "terra cotta":[201,100,59],
        "terracotta":[202,102,65],
        "tiffany blue":[123,242,218],
        "tomato":[239,64,38],
        "tomato red":[236,45,1],
        "topaz":[19,187,175],
        "toupe":[199,172,125],
        "toxic green":[97,222,42],
        "tree green":[42,126,25],
        "true blue":[1,15,204],
        "true green":[8,148,4],
        "turquoise":[6,194,172],
        "turquoise blue":[6,177,196],
        "turquoise green":[4,244,137],
        "turtle green":[117,184,79],
        "twilight":[78,81,139],
        "twilight blue":[10,67,122],
        "ugly blue":[49,102,138],
        "ugly brown":[125,113,3],
        "ugly green":[122,151,3],
        "ugly pink":[205,117,132],
        "ugly purple":[164,66,160],
        "ugly yellow":[208,193,1],
        "ultramarine":[32,0,177],
        "ultramarine blue":[24,5,219],
        "umber":[178,100,0],
        "velvet":[117,8,81],
        "vermillion":[244,50,12],
        "very dark blue":[0,1,51],
        "very dark brown":[29,2,0],
        "very dark green":[6,46,3],
        "very dark purple":[42,1,52],
        "very light blue":[213,255,255],
        "very light brown":[211,182,131],
        "very light green":[209,255,189],
        "very light pink":[255,244,242],
        "very light purple":[246,206,252],
        "very pale blue":[214,255,254],
        "very pale green":[207,253,188],
        "vibrant blue":[3,57,248],
        "vibrant green":[10,221,8],
        "vibrant purple":[173,3,222],
        "violet":[154,14,234],
        "violet blue":[81,10,201],
        "violet pink":[251,95,252],
        "violet red":[165,0,85],
        "viridian":[30,145,103],
        "vivid blue":[21,46,255],
        "vivid green":[47,239,16],
        "vivid purple":[153,0,250],
        "vomit":[162,164,21],
        "vomit green":[137,162,3],
        "vomit yellow":[199,193,12],
        "warm blue":[75,87,219],
        "warm brown":[150,78,2],
        "warm grey":[151,138,132],
        "warm pink":[251,85,129],
        "warm purple":[149,46,143],
        "washed out green":[188,245,166],
        "water blue":[14,135,204],
        "watermelon":[253,70,89],
        "weird green":[58,229,127],
        "wheat":[251,221,126],
        "white":[255,255,255],
        "windows blue":[55,120,191],
        "wine":[128,1,63],
        "wine red":[123,3,35],
        "wintergreen":[32,249,134],
        "wisteria":[168,125,194],
        "yellow":[255,255,20],
        "yellow brown":[183,148,0],
        "yellowgreen":[187,249,15],
        "yellow green":[192,251,45],
        "yellow/green":[200,253,61],
        "yellowish":[250,238,102],
        "yellowish brown":[155,122,1],
        "yellowish green":[176,221,22],
        "yellowish orange":[255,171,15],
        "yellowish tan":[252,252,129],
        "yellow ochre":[203,157,6],
        "yellow orange":[252,176,1],
        "yellow tan":[255,227,110],
        "yellowy brown":[174,139,12],
        "yellowy green":[191,241,40],
    }

    @staticmethod
    def getBasicColorsFromObject(obj, mode='advanced', thresholdRelArea=0.2):
        
        # Load the model
        loader = Loader.getGlobalPtr()
        loaderOptions = LoaderOptions()
        node = loader.loadSync(Filename(obj.modelFilename), loaderOptions)
        if node is not None:
            nodePath = NodePath(node)
        else:
            raise IOError('Could not load model file: %s' % (obj.modelFilename))
        
        # Get the list of materials
        areas, colors, transparencies, _ = getColorAttributesFromModel(nodePath)

        if mode == 'basic':
            table = MaterialColorTable.BasicColorTable
        elif mode == 'advanced':
            table = MaterialColorTable.AdvancedColorTable
        elif mode == 'xkcd':
            table = MaterialColorTable.XkcdColorTable
        else:
            raise Exception('Unsupported color mode: %s' % (mode))
        
        # Get the most dominant colors based on threshold on relative surface area
        colorDescriptions = []
        for area, color, _ in zip(areas, colors, transparencies):
            if not area >= thresholdRelArea: continue
            
            #TODO: compare color in HSV or HSL domain instead of RGB?
            #hsvColor = colorsys.rgb_to_hsv(*color)
            
            # Find nearest color
            minDistance = np.Inf
            bestColorName = None
            for colorName, refColor in table.iteritems():
                dist = np.linalg.norm(np.array(refColor)/255.0 - np.array(color), ord=2)
                if dist < minDistance:
                    minDistance = dist
                    bestColorName = colorName
        
            colorDescriptions.append(bestColorName)
        
        # Remove duplicates (if any)
        colorDescriptions = list(set(colorDescriptions))
        
        # Unload model
        nodePath.removeNode()
        
        return colorDescriptions

class SemanticWorld(object):

    def __init__(self):
        pass

    def addObjectToScene(self, obj):
        pass
    
    def addRoomToScene(self, room):
        pass
    
    def addHouseToScene(self, house):
        pass

class SuncgSemanticWorld(SemanticWorld):
    
    def __init__(self, datasetRoot):
        self.categoryMapping = ModelCategoryMapping(os.path.join(datasetRoot, 'metadata', 'ModelCategoryMapping.csv'))
    
    def _describeObjectCategory(self, obj):
        category = self.categoryMapping.getFineGrainedCategoryForModelId(obj.modelId)
        desc = category.replace("_", " ")
        return desc
    
    def _describeObjectColor(self, obj):
        colors = MaterialColorTable.getBasicColorsFromObject(obj, mode='advanced')
        desc = ', '.join(colors)
        return desc
    
    def _describeObjectMaterial(self, obj):
        materials = MaterialTable.getMaterialNameFromObject(obj)
        desc = 'made of ' + ','.join(materials)
        return desc
    
    def describeObject(self, obj):
        
        items = []
        
        # TODO: color attribute of the main material RGB values
        colorDescription = self._describeObjectColor(obj)
        items.append(colorDescription)
        
        # TODO: category attribute from the SUNCG mapping
        categoryDescription = self._describeObjectCategory(obj)
        items.append(categoryDescription)
        
        # TODO: material attribute of the main textures
        materialDescription = self._describeObjectMaterial(obj)
        items.append(materialDescription)
        
        desc = " ".join(items)
        return desc
