import json
import cv2
import numpy as np

from torch.utils.data import Dataset
import random
from PIL import Image
import open_clip
import torch
import os
from ldm.modules.encoders.modules import FrozenCLIPEmbedder as CLIP

clip = CLIP()

ADE20K={
    1 : "wall",
    2 : "building edifice",
    3 : "sky",
    4 : "floor",
    5 : "tree",
    6 : "ceiling",
    7 : "road",
    8 : "bed",
    9 : "windowpane",
    10 : "grass",
    11 : "cabinet",
    12 : "sidewalk",
    13 : "person",
    14 : "earth ground",
    15 : "door",
    16 : "table",
    17 : "mountain",
    18 : "plant flora",
    19 : "curtain drapery mantle pall",
    20 : "chair",
    21 : "car",
    22 : "water",
    23 : "painting picture",
    24 : "sofa lounge",
    25 : "shelf",
    26 : "house",
    27 : "sea",
    28 : "mirror",
    29 : "carpet",
    30 : "field",
    31 : "armchair",
    32 : "seat",
    33 : "fence",
    34 : "desk",
    35 : "rock stone",
    36 : "wardrobe closet",
    37 : "lamp",
    38 : "bathtub",
    39 : "railing",
    40 : "cushion",
    41 : "base pedestal stand",
    42 : "box",
    43 : "pillar",
    44 : "signboard sign",
    45 : "chest bureau dresser",
    46 : "counter",
    47 : "sand",
    48 : "sink",
    49 : "skyscraper",
    50 : "fireplace",
    51 : "refrigerator",
    52 : "grandstand covered stand",
    53 : "path",
    54 : "stairs",
    55 : "runway",
    56 : "case showcase vitrine",
    57 : "pool table billiard table snooker table",                                                                                                                                                                   
    58 : "pillow",
    59 : "screen door",
    60 : "stairway",
    61 : "river",
    62 : "bridge",
    63 : "bookcase",
    64 : "blind screen",
    65 : "coffee table cocktail table",
    66 : "toilet can commode potty",
    67 : "flower",
    68 : "book",
    69 : "hill",
    70 : "bench",
    71 : "countertop",
    72 : "kitchen range cooking stove",
    73 : "palm tree",                                                                                                                                                                                                 
    74 : "kitchen island",
    75 : "computer",
    76 : "swivel chair",
    77 : "boat",
    78 : "bar",
    79 : "arcade machine",
    80 : "hovel shack shanty",
    81 : "autobus motorbus omnibus",
    82 : "towel",
    83 : "light",
    84 : "truck",
    85 : "tower",
    86 : "chandelier pendant",
    87 : "awning sunblind",
    88 : "streetlight",
    89 : "booth cubicle kiosk",
    90 : "television tv set idiot box boob tube telly goggle box",
    91 : "airplane",                                                                                                                                                                                              
    92 : "dirt track",
    93 : "apparel",
    94 : "pole",
    95 : "land ground soil",
    96 : "balustrade handrail",
    97 : "escalator",
    98 : "ottoman pouf hassock",
    99 : "bottle",
    100 : "buffet counter sideboard",
    101 : "poster placard notice card",
    102 : "stage",
    103 : "van",
    104 : "ship",
    105 : "fountain",
    106 : "conveyor belt transporter",
    107 : "canopy",                                                                                                                                                         
    108 : "washing machine",
    109 : "toy",
    110 : "swimming pool natatorium",
    111 : "stool",
    112 : "barrel",
    113 : "basket handbasket",
    114 : "waterfall",
    115 : "tent",
    116 : "bag",
    117 : "motorbike",
    118 : "cradle",
    119 : "oven",
    120 : "ball",
    121 : "food",
    122 : "stair",
    123 : "storage tank",
    124 : "brand marque",
    125 : "microwave oven",
    126 : "flowerpot",
    127 : "animal fauna",
    128 : "bicycle",
    129 : "lake",
    130 : "dishwasher",
    131 : "screen silver screen projection screen",
    132 : "blanket",
    133 : "sculpture",
    134 : "exhaust hood",
    135 : "sconce",
    136 : "vase",
    137 : "traffic light",
    138 : "tray",
    139 : "ashcan trash can dustbin",
    140 : "fan",
    141 : "pier wharfage dock",
    142 : "crt screen",
    143 : "plate",
    144 : "monitoring device",
    145 : "notice board",
    146 : "shower",
    147 : "radiator",
    148 : "drinking glass",
    149 : "clock",
    150 : "flag"  
}

COCO={
    1 : "person",
    2 : "bicycle",
    3 : "car",
    4 : "motorcycle",
    5 : "airplane",
    6 : "bus",
    7 : "train",
    8 : "truck",
    9 : "boat",
    10 : "traffic light",
    11 : "fire hydrant",
    12 : "street sign",
    13 : "stop sign",
    14 : "parking meter",
    15 : "bench",
    16 : "bird",
    17 : "cat",
    18 : "dog",
    19 : "horse",
    20 : "sheep",
    21 : "cow",
    22 : "elephant",
    23 : "bear",
    24 : "zebra",
    25 : "giraffe",
    26 : "hat",
    27 : "backpack",
    28 : "umbrella",
    29 : "shoe",
    30 : "eye glasses",
    31 : "handbag",
    32 : "tie",
    33 : "suitcase",
    34 : "frisbee",
    35 : "skis",
    36 : "snowboard",
    37 : "sports ball",
    38 : "kite",
    39 : "baseball bat",
    40 : "baseball glove",
    41 : "skateboard",
    42 : "surfboard",
    43 : "tennis racket",
    44 : "bottle",
    45 : "plate",
    46 : "wine glass",
    47 : "cup",
    48 : "fork",
    49 : "knife",
    50 : "spoon",
    51 : "bowl",
    52 : "banana",
    53 : "apple",
    54 : "sandwich",
    55 : "orange",
    56 : "broccoli",
    57 : "carrot",
    58 : "hot dog",
    59 : "pizza",
    60 : "donut",
    61 : "cake",
    62 : "chair",
    63 : "couch",
    64 : "potted plant",
    65 : "bed",
    66 : "mirror",
    67 : "dining table",
    68 : "window",
    69 : "desk",
    70 : "toilet",
    71 : "door",
    72 : "tv",
    73 : "laptop",
    74 : "mouse",
    75 : "remote",
    76 : "keyboard",
    77 : "cell phone",
    78 : "microwave",
    79 : "oven",
    80 : "toaster",
    81 : "sink",
    82 : "refrigerator",
    83 : "blender",
    84 : "book",
    85 : "clock",
    86 : "vase",
    87 : "scissors",
    88 : "teddy bear",
    89 : "hair drier",
    90 : "toothbrush",
    91 : "hair brush",
    92 : "banner",
    93 : "blanket",
    94 : "branch",
    95 : "bridge",
    96 : "building",
    97 : "bush",
    98 : "cabinet",
    99 : "cage",
    100 : "cardboard",
    101 : "carpet",
    102 : "ceiling",
    103 : "tile ceiling",
    104 : "cloth",
    105 : "clothes",
    106 : "clouds",
    107 : "counter",
    108 : "cupboard",
    109 : "curtain",
    110 : "desk",
    111 : "dirt",
    112 : "door",
    113 : "fence",
    114 : "marble floor",
    115 : "floor",
    116 : "stone floor",
    117 : "tile floor",
    118 : "wood floor",
    119 : "flower",
    120 : "fog",
    121 : "food",
    122 : "fruit",
    123 : "furniture",
    124 : "grass",
    125 : "gravel",
    126 : "ground",
    127 : "hill",
    128 : "house",
    129 : "leaves",
    130 : "light",
    131 : "mat",
    132 : "metal",
    133 : "mirror",
    134 : "moss",
    135 : "mountain",
    136 : "mud",
    137 : "napkin",
    138 : "net",
    139 : "paper",
    140 : "pavement",
    141 : "pillow",
    142 : "plant",
    143 : "plastic",
    144 : "platform",
    145 : "playingfield",
    146 : "railing",
    147 : "railroad",
    148 : "river",
    149 : "road",
    150 : "rock",
    151 : "roof",
    152 : "rug",
    153 : "salad",
    154 : "sand",
    155 : "sea",
    156 : "shelf",
    157 : "sky",
    158 : "skyscraper",
    159 : "snow",
    160 : "solid",
    161 : "stairs",
    162 : "stone",
    163 : "straw",
    164 : "structural",
    165 : "table",
    166 : "tent",
    167 : "textile",
    168 : "towel",
    169 : "tree",
    170 : "vegetable",
    171 : "brick wall",
    172 : "concrete wall",
    173 : "wall",
    174 : "panel wall",
    175 : "stone wall",
    176 : "tile wall",
    177 : "wood wall",
    178 : "water",
    179 : "waterdrops",
    180 : "blind window",
    181 : "window",
    182 : "wood"
}

def gettks(tkss):
    newtks = []
    for i in range(77):
        if tkss[0,i]==49407:
            break
        elif tkss[0,i]==49406:
            continue
        elif tkss[0,i]==267:
            continue
        else:
            newtks.append(int(tkss[0,i]))
    return newtks

class ADE20KDataset(Dataset):
    def __init__(self, data_root, phase='Val'):
        self.data = []
        path = data_root
        prefix = 'validation' if phase=='Val' else 'training'
        files = os.listdir(path+'/images/'+prefix+'/') if phase=='Train' else os.listdir(path+'/images/'+prefix+'/')
        for fn in files:
            self.data.append({'target':path+'/images/'+prefix+'/'+fn,'source':path+'/annotations/'+prefix+'/'+fn.replace('.jpg','.png')})

        self.labeldic={}
        self.namedic={}
        for k in ADE20K.keys():
            self.namedic[int(k)-1] = ADE20K[k]

            batch_encoding = clip.tokenizer(ADE20K[k], truncation=True, max_length=clip.max_length, return_length=True,
                 return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            tokens = batch_encoding["input_ids"]
            corr_tks = gettks(tokens)

            self.labeldic[int(k)-1] = corr_tks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']

        source = np.array(Image.open(source_filename).resize((512,512),Image.NEAREST),dtype=np.float)
        target = np.array(Image.open(target_filename).resize((512,512),Image.BICUBIC).convert('RGB'))

        tokens = np.full((77),49407)
        tokens[0] = 49406
        token_list = []
        tokens_cls = np.full((77),49407)
        tokens_cls[0] = 49406
        tokens_cls_list = []
        source_minus_1 = source - 1
        prompt = ''
        for lb in np.unique(source_minus_1):
            if lb==-1:
                continue
            token_list+=self.labeldic[lb]
            tokens_cls_list += [lb]*len(self.labeldic[lb])
            prompt += self.namedic[lb] + ','
        tokens[1:len(token_list)+1] = np.array(token_list)
        tokens_cls[1:len(token_list)+1] = np.array(tokens_cls_list)

        # Normalize source images to [0, 1].
        viewsource = np.zeros((512,512,3))
        viewsource[:,:,0] = source
        viewsource[:,:,1] = source
        viewsource[:,:,2] = source
        viewsource = np.array(viewsource,dtype=np.uint8)
        viewsource = viewsource.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        source = source_minus_1[:,:,np.newaxis]

        assert viewsource.shape==target.shape, str(viewsource.shape)+' '+str(target.shape) + ' '+item['target']+' ' + item['source']

        return dict(jpg=target, txt=prompt, tks=tokens, hint=source, targetpath=item['target'], sourcepath=item['source'], viewcontrol=viewsource, tokens_cls=tokens_cls)

class COCODataset(Dataset):
    def __init__(self, data_root, phase='Val'):
        self.data = []
        path = data_root
        prefix = 'val2017' if phase=='Val' else 'train2017'
        files = os.listdir(path+prefix)
        for fn in files:
            self.data.append({'target':path+'/'+prefix+'/'+fn,'source':path+'/stuffthingmaps_trainval2017/'+prefix+'/'+fn.replace('.jpg','.png')})

        self.labeldic={}
        self.namedic={}
        for k in COCO.keys():
            self.namedic[int(k)-1] = COCO[k]

            batch_encoding = clip.tokenizer(COCO[k], truncation=True, max_length=clip.max_length, return_length=True,
                 return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            tokens = batch_encoding["input_ids"]
            corr_tks = gettks(tokens)

            self.labeldic[int(k)-1] = corr_tks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']

        source = np.array(Image.open(source_filename).resize((512,512),Image.NEAREST),dtype=np.float)
        source = np.where(source==255,-1,source)
        source = source + 1
        
        target = np.array(Image.open(target_filename).resize((512,512),Image.BILINEAR).convert('RGB'))

        tokens = np.full((77),49407)
        tokens[0] = 49406
        token_list = []
        tokens_cls = np.full((77),49407)
        tokens_cls[0] = 49406
        tokens_cls_list = []
        source_minus_1 = source - 1
        prompt = ''
        for lb in np.unique(source_minus_1):
            if lb==-1:
                continue
            token_list+=self.labeldic[lb]
            tokens_cls_list += [lb]*len(self.labeldic[lb])
            prompt += self.namedic[lb] + ','
        tokens[1:len(token_list)+1] = np.array(token_list)
        tokens_cls[1:len(token_list)+1] = np.array(tokens_cls_list)

        # Normalize source images to [0, 1].
        viewsource = np.zeros((512,512,3))
        viewsource[:,:,0] = source
        viewsource[:,:,1] = source
        viewsource[:,:,2] = source
        viewsource = np.array(viewsource,dtype=np.uint8)
        viewsource = viewsource.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        source = source_minus_1[:,:,np.newaxis]

        assert viewsource.shape==target.shape, str(viewsource.shape)+' '+str(target.shape) + ' '+item['target']+' ' + item['source']

        return dict(jpg=target, txt=prompt, tks=tokens, hint=source, targetpath=item['target'], sourcepath=item['source'], viewcontrol=viewsource, tokens_cls=tokens_cls)