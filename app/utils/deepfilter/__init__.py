# from livekit.agents import Plugin
# import logging

# from .noise_suppressor import DeepFilterNoiseSuppressor

# logger = logging.getLogger(__name__)

# class DeepFilterPlugin(Plugin):
#     def __init__(self):
#         super().__init__(
#             title="DeepFilterNet",
#             version="0.3.0",
#             package="livekit-plugins-deepfilter",
#             logger=logger,
#         )

#     def download_files(self):
#         # DeepFilterNet handles model downloading internally via init_df()
#         pass

# def noise_suppression(**kwargs) -> DeepFilterNoiseSuppressor:
#     """Create a DeepFilterNoiseSuppressor instance."""
#     return DeepFilterNoiseSuppressor(**kwargs)

# Plugin.register_plugin(DeepFilterPlugin())

# __all__ = ["DeepFilterNoiseSuppressor", "noise_suppression"]