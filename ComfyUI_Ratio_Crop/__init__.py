from .nodes import RatioCropNode, RatioMergeNode, RatioBatchLoader, RatioAutoCropSAM

NODE_CLASS_MAPPINGS = {
    "RatioCropNode": RatioCropNode,
    "RatioMergeNode": RatioMergeNode,
    "RatioBatchLoader": RatioBatchLoader,
    "RatioAutoCropSAM": RatioAutoCropSAM
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RatioCropNode": "Ratio Crop Image",
    "RatioMergeNode": "Ratio Merge Image",
    "RatioBatchLoader": "Ratio Batch Loader",
    "RatioAutoCropSAM": "Ratio Auto Crop (SAM)"
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
