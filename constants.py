class Losses:
    CROSS_ENTROPY = "xentropy"
    MSE = "mse"
    NLL = "nll"
    ALEATORIC = "aleatoric"


IGNORE_INDEX = {"cityscapes": 19, "potsdam": 0, "flightmare": 9, "shapenet": None}
