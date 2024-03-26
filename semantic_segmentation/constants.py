class Losses:
    CROSS_ENTROPY = "xentropy"
    MSE = "mse"
    NLL = "nll"
    ALEATORIC = "aleatoric"


IGNORE_INDEX = {"cityscapes": 19, "potsdam": 0, "flightmare": 9, "shapenet": None, "pascal_voc": 0}


LABELS = {
    "cityscapes": {
        "road": {"color": (128, 64, 128), "id": 0},
        "sidewalk": {"color": (244, 35, 232), "id": 1},
        "building": {"color": (70, 70, 70), "id": 2},
        "wall": {"color": (102, 102, 156), "id": 3},
        "fence": {"color": (190, 153, 153), "id": 4},
        "pole": {"color": (153, 153, 153), "id": 5},
        "traffic light": {"color": (250, 170, 30), "id": 6},
        "traffic sign": {"color": (220, 220, 0), "id": 7},
        "vegetation": {"color": (107, 142, 35), "id": 8},
        "terrain": {"color": (152, 251, 152), "id": 9},
        "sky": {"color": (70, 130, 180), "id": 10},
        "person": {"color": (220, 20, 60), "id": 11},
        "rider": {"color": (255, 0, 0), "id": 12},
        "car": {"color": (0, 0, 142), "id": 13},
        "truck": {"color": (0, 0, 70), "id": 14},
        "bus": {"color": (0, 60, 100), "id": 15},
        "train": {"color": (0, 80, 100), "id": 16},
        "motorcycle": {"color": (0, 0, 230), "id": 17},
        "bicycle": {"color": (119, 11, 32), "id": 18},
        "void": {"color": (0, 0, 0), "id": 19},
    },
    "potsdam": {
        "boundary line": {"color": (0, 0, 0), "id": 0},
        "imprevious surfaces": {"color": (255, 255, 255), "id": 1},
        "building": {"color": (0, 0, 255), "id": 2},
        "low vegetation": {"color": (0, 255, 255), "id": 3},
        "tree": {"color": (0, 255, 0), "id": 4},
        "car": {"color": (255, 255, 0), "id": 5},
        "clutter/background": {"color": (255, 0, 0), "id": 6},
    },
    "flightmare": {
        "background": {"color": (0, 0, 0), "id": 0},
        "floor": {"color": (2, 73, 9), "id": 1},
        "hangar": {"color": (32, 73, 65), "id": 2},
        "fence": {"color": (6, 73, 72), "id": 3},
        "road": {"color": (36, 73, 8), "id": 4},
        "tank": {"color": (2, 73, 128), "id": 5},
        "pipe": {"color": (32, 9, 201), "id": 6},
        "container": {"color": (6, 9, 193), "id": 7},
        "misc": {"color": (36, 9, 129), "id": 8},
        "boundary": {"color": (255, 255, 255), "id": 9},
    },
    "shapenet": {
        "background": {"color": (255, 255, 255), "id": 0},
        "car": {"color": (102, 102, 102), "id": 1},
        "chair": {"color": (0, 0, 255), "id": 2},
        "table": {"color": (0, 255, 255), "id": 3},
        "sofa": {"color": (255, 0, 0), "id": 4},
        "airplane": {"color": (102, 0, 204), "id": 5},
        "camera": {"color": (0, 102, 0), "id": 6},
        # "birdhouse": {"color": (255, 153, 204), "id": 7},
    },
    "pascal": {
        "background": {"color": (0, 0, 0), "id": 0},
        "aeroplane": {"color": (128, 0, 0), "id": 1},
        "bicycle": {"color": (0, 128, 0), "id": 2},
        "bird": {"color": (128, 128, 0), "id": 3},
        "boat": {"color": (0, 0, 128), "id": 4},
        "bottle": {"color": (128, 0, 128), "id": 5},
        "bus": {"color": (0, 128, 128), "id": 6},
        "car": {"color": (128, 128, 128), "id": 7},
        "cat": {"color": (64, 0, 0), "id": 8},
        "chair": {"color": (192, 0, 0), "id": 9},
        "cow": {"color": (64, 128, 0), "id": 10},
        "diningtable": {"color": (192, 128, 0), "id": 11},
        "dog": {"color": (64, 0, 128), "id": 12},
        "horse": {"color": (192, 0, 128), "id": 13},
        "motorbike": {"color": (64, 128, 128), "id": 14},
        "person": {"color": (192, 128, 128), "id": 15},
        "pottedplant": {"color": (0, 64, 0), "id": 16},
        "sheep": {"color": (128, 64, 0), "id": 17},
        "sofa": {"color": (0, 192, 0), "id": 18},
        "train": {"color": (128, 192, 0), "id": 19},
        "tvmonitor": {"color": (0, 64, 128), "id": 20},
    },

}

THEMES = {
    "cityscapes": "cityscapes",
    "potsdam": "potsdam",
    "flightmare": "flightmare",
    "shapenet": "shapenet",
}
