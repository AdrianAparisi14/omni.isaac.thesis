# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from omni.isaac.thesis.base_sample import BaseSampleExtension
from omni.isaac.thesis.force_dataset import SceneForceDataset


class HelloWorldExtension(BaseSampleExtension):
    def on_startup(self, ext_id: str):
        super().on_startup(ext_id)
        super().start_extension(
            menu_name="Force Torque Dataset",
            submenu_name="",
            name="Create Force Torque Dataset",
            title="Manipulator Example",
            doc_link="https://docs.omniverse.nvidia.com/isaacsim/latest/tutorial_core_adding_manipulator.html",
            overview="Extension to create a dataset with forces and torques during an assembly process.",
            file_path=os.path.abspath(__file__),
            sample=SceneForceDataset(),
        )

        return