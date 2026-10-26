from .w2gg import W2GG
ARCHITECTURES = {"W2GG": W2GG}

def build_model(cfg):
    return ARCHITECTURES[cfg.MODEL.ARCHITECTURE](cfg)
