from collections import OrderedDict
from typing import List, Optional

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.utils.common import random_choice
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim.utils.sapien_utils import vectorize_pose

from .base_env import CustomSceneEnv, CustomOtherObjectsInSceneEnv
from .grasp_single_in_scene import GraspSingleCustomOrientationInSceneEnv, GraspSingleCustomInSceneEnv, GraspSingleInSceneEnv

@register_env("AltGraspSpongeDistractorInSceneEnv-v0", max_episode_steps=80)
class AltGraspSpongeDistractorInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    def __init__(self, distractor_config="less", **kwargs):
        kwargs.pop("model_ids", None)
        kwargs["model_ids"] = ["sponge"]
        if distractor_config == "less":
            self.distractor_model_ids = [
                "bridge_spoon_generated_modified",
                "opened_coke_can"
            ]
        elif distractor_config == "more":
            self.distractor_model_ids = [
                "opened_7up_can",
                "opened_sprite_can",
                "orange",
                "opened_fanta_can",
                "bridge_spoon_generated_modified",
                "opened_coke_can",
            ]
        else:
            raise NotImplementedError()
        kwargs["distractor_model_ids"] = self.distractor_model_ids
        super().__init__(**kwargs)

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()
        options["distractor_model_ids"] = self.distractor_model_ids

        return super().reset(seed=seed, options=options)

    def get_language_instruction(self, **kwargs):
        obj_name = self._get_instruction_obj_name(self.obj.name)
        task_description = f"I want to clean the table. Pick a suitable tool for me." #  The tool is sponge
        return task_description

@register_env("AltGraspOrangeDistractorInSceneEnv-v0", max_episode_steps=80)
class AltGraspOrangeDistractorInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    def __init__(self, distractor_config="less", **kwargs):
        kwargs.pop("model_ids", None)
        kwargs["model_ids"] = ["orange"]
        if distractor_config == "less":
            self.distractor_model_ids = [
                "bridge_spoon_generated_modified",
                "opened_coke_can",
            ]
        elif distractor_config == "more":
            self.distractor_model_ids = [
                "opened_7up_can",
                "opened_sprite_can",
                "opened_fanta_can",
                "bridge_spoon_generated_modified",
                "opened_coke_can",
            ]
        else:
            raise NotImplementedError()
        kwargs["distractor_model_ids"] = self.distractor_model_ids
        super().__init__(**kwargs)

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()
        options["distractor_model_ids"] = self.distractor_model_ids

        return super().reset(seed=seed, options=options)

    def get_language_instruction(self, **kwargs):
        obj_name = self._get_instruction_obj_name(self.obj.name)
        task_description = f"I am thirsty but I do not want drinks. Please grab something for me." # The fruit is orange
        return task_description

@register_env("AltGraspFantaDistractorInSceneEnv-v0", max_episode_steps=80)
class AltGraspFantaDistractorInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    def __init__(self, distractor_config="less", **kwargs):
        kwargs.pop("model_ids", None)
        kwargs["model_ids"] = ["opened_fanta_can"]
        if distractor_config == "less":
            self.distractor_model_ids = [
                "opened_coke_can",
                "orange",
            ]
        elif distractor_config == "more":
            self.distractor_model_ids = [
                "opened_7up_can",
                "opened_sprite_can",
                "orange",
                "bridge_spoon_generated_modified",
                "opened_coke_can",
            ]
        else:
            raise NotImplementedError()
        kwargs["distractor_model_ids"] = self.distractor_model_ids
        super().__init__(**kwargs)

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()
        options["distractor_model_ids"] = self.distractor_model_ids

        return super().reset(seed=seed, options=options)

    def get_language_instruction(self, **kwargs):
        obj_name = self._get_instruction_obj_name(self.obj.name)
        task_description = f"I am thirsty and I want the orange taste drink. Please grab something for me." # The fruit is orange
        return task_description

@register_env("AltGraspOrange2DistractorInSceneEnv-v0", max_episode_steps=80)
class AltGraspOrange2DistractorInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    def __init__(self, distractor_config="less", **kwargs):
        kwargs.pop("model_ids", None)
        kwargs["model_ids"] = ["orange"]
        if distractor_config == "less":
            self.distractor_model_ids = [
                "bridge_spoon_generated_modified",
                "opened_coke_can",
            ]
        elif distractor_config == "more":
            self.distractor_model_ids = [
                "opened_7up_can",
                "opened_sprite_can",
                "eggplant",
                "bridge_spoon_generated_modified",
                "opened_coke_can",
            ]
        else:
            raise NotImplementedError()
        kwargs["distractor_model_ids"] = self.distractor_model_ids
        super().__init__(**kwargs)

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()
        options["distractor_model_ids"] = self.distractor_model_ids

        return super().reset(seed=seed, options=options)

    def get_language_instruction(self, **kwargs):
        obj_name = self._get_instruction_obj_name(self.obj.name)
        task_description = f"Can you grab the fruit before preparing the salad?"
        return task_description

@register_env("AltGraspEggplantDistractorInSceneEnv-v0", max_episode_steps=80)
class AltGraspEggplantDistractorInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    def __init__(self, distractor_config="less", **kwargs):
        kwargs.pop("model_ids", None)
        kwargs["model_ids"] = ["eggplant"]
        if distractor_config == "less":
            self.distractor_model_ids = [
                "bridge_spoon_generated_modified",
                "opened_coke_can",
                "orange",
            ]
        elif distractor_config == "more":
            self.distractor_model_ids = [
                "opened_7up_can",
                "opened_fanta_can",
                "bridge_spoon_generated_modified",
                "opened_coke_can",
                "orange",
            ]
        else:
            raise NotImplementedError()
        kwargs["distractor_model_ids"] = self.distractor_model_ids
        super().__init__(**kwargs)


    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()
        if "model_scale" not in options:
            options["model_scale"] = 1.5
        options["distractor_model_ids"] = self.distractor_model_ids

        return super().reset(seed=seed, options=options)

    def get_language_instruction(self, **kwargs):
        obj_name = self._get_instruction_obj_name(self.obj.name)
        task_description = f"I want a health vegetable. Grab it for me."
        return task_description

@register_env("AltGraspMugDistractorInSceneEnv-v0", max_episode_steps=80)
class AltGraspMugDistractorInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_v1.json"
    def __init__(self, distractor_config="less", **kwargs):
        kwargs.pop("model_ids", None)
        kwargs["model_ids"] = ["coffee_mug"]
        if distractor_config == "less":
            self.distractor_model_ids = [
                "bridge_spoon_generated_modified",
                "opened_coke_can",
                "orange",
            ]
        elif distractor_config == "more":
            self.distractor_model_ids = [
                "opened_7up_can",
                "opened_fanta_can",
                "bridge_spoon_generated_modified",
                "opened_coke_can",
                "orange",
            ]
        else:
            raise NotImplementedError()
        kwargs["distractor_model_ids"] = self.distractor_model_ids
        super().__init__(**kwargs)


    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()
        if "model_scale" not in options:
            options["model_scale"] = 1.5
        options["distractor_model_ids"] = self.distractor_model_ids

        return super().reset(seed=seed, options=options)

    def get_language_instruction(self, **kwargs):
        task_description = f"I want to drink the coffee. Grab the tool for me."
        return task_description

@register_env("AltGraspMugDistractorInSceneEnv-v1", max_episode_steps=80)
class AltGraspMugDistractorInSceneEnvV1(AltGraspMugDistractorInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_v1.json"
    
    def get_language_instruction(self, **kwargs):
        obj_name = self._get_instruction_obj_name(self.obj.name)
        task_description = f"pick up the water container."
        return task_description


from .open_drawer_in_scene import CloseDrawerInSceneEnv, OpenDrawerCustomInSceneEnv
from .place_in_closed_drawer_in_scene import PlaceObjectInClosedDrawerInSceneEnv, PlaceIntoClosedDrawerCustomInSceneEnv
from mani_skill2_real2sim.utils.sapien_utils import get_entity_by_name
from mani_skill2_real2sim.utils.sapien_utils import (
    get_pairwise_contacts,
    compute_total_impulse,
)


@register_env("AltCloseDrawerInLongStep2HorizonInSceneEnv-v0", max_episode_steps=200)
class AltCloseDrawerInLongHorizonInSceneEnv(PlaceObjectInClosedDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    def __init__(self, **kwargs):
        self.DEFAULT_ASSET_ROOT = "{ASSET_DIR}/custom"
        self.DEFAULT_SCENE_ROOT = "{ASSET_DIR}/hab2_bench_assets"
        self.DEFAULT_MODEL_JSON = "info_pick_custom_v1.json"
        self.drawer_ids = ["top"]
        super().__init__(**kwargs)
    
    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        if "obj_init_options" not in options:
            options["obj_init_options"] = dict()
        if "cabinet_init_qpos" not in options["obj_init_options"]:
            options["obj_init_options"]["cabinet_init_qpos"] = 0.2
        return super().reset(seed=seed, options=options)

    def evaluate(self, **kwargs):
        qpos = self.art_obj.get_qpos()[self.joint_idx]
        self.episode_stats["qpos"] = "{:.3f}".format(qpos)
        return dict(success=qpos <= 0.05, qpos=qpos, episode_stats=self.episode_stats)

    def get_language_instruction(self):
        model_name = self._get_instruction_obj_name(self.model_id)
        return f"After grab the {model_name} from the {self.drawer_ids[0]} drawer, please close the top drawer"


@register_env("AltCloseMiddleDrawerInLongStep2HorizonInSceneEnv-v0", max_episode_steps=200)
class AltCloseMiddleDrawerInLongHorizonInSceneEnv(PlaceObjectInClosedDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    def __init__(self, **kwargs):
        self.DEFAULT_ASSET_ROOT = "{ASSET_DIR}/custom"
        self.DEFAULT_SCENE_ROOT = "{ASSET_DIR}/hab2_bench_assets"
        self.DEFAULT_MODEL_JSON = "info_pick_custom_v0.json"
        self.drawer_ids = ["middle"]
        super().__init__(**kwargs)
    
    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        if "obj_init_options" not in options:
            options["obj_init_options"] = dict()
        if "cabinet_init_qpos" not in options["obj_init_options"]:
            options["obj_init_options"]["cabinet_init_qpos"] = 0.2
        return super().reset(seed=seed, options=options)

    def evaluate(self, **kwargs):
        qpos = self.art_obj.get_qpos()[self.joint_idx]
        self.episode_stats["qpos"] = "{:.3f}".format(qpos)
        return dict(success=qpos <= 0.05, qpos=qpos, episode_stats=self.episode_stats)

    def get_language_instruction(self):
        model_name = self._get_instruction_obj_name(self.model_id)
        return f"Pick the {model_name} from the {self.drawer_ids[0]} drawer, then close the {self.drawer_ids[0]} drawer"


@register_env("AltCloseDrawerInLongStep1HorizonInSceneEnv-v0", max_episode_steps=200)
class AltCloseDrawerInLongStep12HorizonInSceneEnv(PlaceObjectInClosedDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    def __init__(self, **kwargs):
        self.DEFAULT_ASSET_ROOT = "{ASSET_DIR}/custom"
        self.DEFAULT_SCENE_ROOT = "{ASSET_DIR}/hab2_bench_assets"
        self.DEFAULT_MODEL_JSON = "info_pick_custom_v0.json"
        self.drawer_ids = ["top"]
        super().__init__(**kwargs)
    
    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()
        # ========================= reset drawer ==========================
        reconfigure = options.get("reconfigure", False)
        self.set_episode_rng(seed)
        self.drawer_id = self._episode_rng.choice(self.drawer_ids)

        if self.prepackaged_config:
            _reconfigure = self._additional_prepackaged_config_reset(options)
            reconfigure = reconfigure or _reconfigure

        options["reconfigure"] = reconfigure

        self._initialize_episode_stats()

        obs, info = super().reset(seed=self._episode_seed, options=options) # articulations are loaded here
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")

        # setup cabinet qpos
        obj_init_options = options.get("obj_init_options", {})
        obj_init_options = obj_init_options.copy()
        cabinet_init_qpos = obj_init_options.get("cabinet_init_qpos", None)
        if cabinet_init_qpos is not None:
            if isinstance(cabinet_init_qpos, float):
                # set qpos for target cabinet joint
                tmp = [0.0] * self.art_obj.dof
                tmp[self.joint_idx] = cabinet_init_qpos
                cabinet_init_qpos = tmp
            self.art_obj.set_qpos(cabinet_init_qpos)
        else:
            self.art_obj.set_qpos([0.15,0,0]) # ensure that the top drawer is open
        # ========================= reset object ==========================
        # set objects
        self.obj_init_options = options.get("obj_init_options", {})
        model_scale = options.get("model_scale", None)
        model_id = options.get("model_id", None)
        reconfigure = options.get("reconfigure", False)
        _reconfigure = self._set_model(model_id, model_scale)
        reconfigure = _reconfigure or reconfigure
        options["reconfigure"] = reconfigure

        self.drawer_link: sapien.Link = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        self.drawer_collision = self.drawer_link.get_collision_shapes()[2]

        # ========================= reset object ==========================

        obs = self.get_obs()

        info.update(
            {
                "drawer_pose_wrt_robot_base": self.agent.robot.pose.inv()
                * self.drawer_obj.pose,
                "cabinet_pose_wrt_robot_base": self.agent.robot.pose.inv()
                * self.art_obj.pose,
                "station_name": self.station_name,
                "light_mode": self.light_mode,
            }
        )
        return obs, info

    def evaluate(self, **kwargs):
        qpos = self.art_obj.get_qpos()[self.joint_idx]
        obj = self.obj
        self.episode_stats["qpos"] = "{:.3f}".format(qpos)

        # Check whether the object contacts with the drawer
        contact_infos = get_pairwise_contacts(
            self._scene.get_contacts(),
            self.obj,
            self.drawer_link,
            collision_shape1=self.drawer_collision,
        )
        total_impulse = compute_total_impulse(contact_infos)
        has_contact = np.linalg.norm(total_impulse) > 1e-6
        self.episode_stats["has_contact"] += has_contact

        success = self.episode_stats["has_contact"] >= 1

        return dict(success=success, qpos=qpos, episode_stats=self.episode_stats)

    def get_language_instruction(self):
        model_name = self._get_instruction_obj_name(self.model_id)
        return f"Please grab the {model_name} into the top drawer, then close it"



@register_env("AltPlaceIntoCompleteLongHorizonInSceneEnv-v0", max_episode_steps=200)
class AltPlaceIntoCompleteLongHorizonInSceneEnv(PlaceIntoClosedDrawerCustomInSceneEnv):
    drawer_ids = ["top"]
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def get_language_instruction(self):
        model_name = self._get_instruction_obj_name(self.model_id)
        return f"Open the {self.drawer_id} drawer and place {model_name} into it"

# @register_env("AltPlaceIntoInSceneEnv-v0", max_episode_steps=113)
# class AltPlaceIntoInSceneEnv(PlaceIntoClosedDrawerCustomInSceneEnv):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#     # Open the drawer at the beginning
#     def reset(self, seed=None, options=None):
#         if options is None:
#             options = dict()
#         if "obj_init_options" not in options:
#             options["obj_init_options"] = dict()
#         if "cabinet_init_qpos" not in options["obj_init_options"]:
#             options["obj_init_options"]["cabinet_init_qpos"] = 0.2
#         return super().reset(seed=seed, options=options)
#     def get_language_instruction(self):
#         model_name = self._get_instruction_obj_name(self.model_id)
#         return f"Can you take the {model_name} from the table and store it in the {self.drawer_id} drawer below?"


@register_env("AltOpenBottomDrawerCustomInSceneEnv-v0", max_episode_steps=113)
class AltOpenBottomDrawerCustomInSceneEnv(OpenDrawerCustomInSceneEnv):
    drawer_ids = ["bottom"]
    success = False
    # we define success by judge whether the drawer is opened once
    def evaluate(self, **kwargs):
        qpos = self.art_obj.get_qpos()[self.joint_idx]
        self.episode_stats["qpos"] = "{:.3f}".format(qpos)
        self.success = qpos >= 0.15 or self.success
        return dict(success=self.success, qpos=qpos, episode_stats=self.episode_stats)
    def get_language_instruction(self, **kwargs):
        return f"Please check if there are any items in the {self.drawer_ids[0]} drawer"

@register_env("AltOpenTopDrawerCustomInSceneEnv-v0", max_episode_steps=113)
class AltOpenTopDrawerCustomInSceneEnv(AltOpenBottomDrawerCustomInSceneEnv):
    drawer_ids = ["top"]

@register_env("AltOpenMiddleDrawerCustomInSceneEnv-v0", max_episode_steps=113)
class AltOpenMiddleDrawerCustomInSceneEnv(AltOpenBottomDrawerCustomInSceneEnv):
    drawer_ids = ["middle"]

@register_env("AltOpenTopDrawerCustomInSceneEnv2-v0", max_episode_steps=113)
class AltOpenTopDrawer2CustomInSceneEnv(OpenDrawerCustomInSceneEnv):
    drawer_ids = ["top"]
    def get_language_instruction(self, **kwargs):
        return f"I need you to retrieve some utensils from the top drawer"


@register_env("AltOpenMiddleDrawerCustomInSceneEnv2-v0", max_episode_steps=113)
class AltOpenMiddleDrawer2CustomInSceneEnv(OpenDrawerCustomInSceneEnv):
    drawer_ids = ["middle"]
    def get_language_instruction(self, **kwargs):
        return f"I need you to retrieve some utensils from the middle drawer"

@register_env("AltOpenBottomDrawerCustomInSceneEnv2-v0", max_episode_steps=113)
class AltOpenBottomDrawer2CustomInSceneEnv(OpenDrawerCustomInSceneEnv):
    drawer_ids = ["bottom"]
    def get_language_instruction(self, **kwargs):
        return f"I need you to retrieve some utensils from the bottom drawer"


from .move_near_in_scene import MoveNearGoogleInSceneEnv

@register_env("AltPlaceAppleNearOrangeCanEnv-v0", max_episode_steps=80)
class AltPlaceAppleNearOrangeCanEnv(MoveNearGoogleInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_baked_tex_v0.json"
    _main_rng = 42
    _episode_seed = 42
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_obj_configs()

    def _setup_obj_configs(self):
        self.triplets = [
            ("baked_apple", "baked_opened_pepsi_can", "baked_sponge"),

        ]

        self._source_obj_ids = [0]  # apple
        self._target_obj_ids = [1]

        self._xy_config_per_triplet = [
            ([-0.33, 0.04], [-0.33, 0.34], [-0.13, 0.19]),
            ([-0.33, 0.34], [-0.33, 0.04], [-0.13, 0.19]),
            ([-0.33, 0.34], [-0.13, 0.19], [-0.33, 0.04]),

            ([-0.33, 0.04], [-0.13, 0.19], [-0.33, 0.34]),
            ([-0.13, 0.19], [-0.33, 0.34], [-0.33, 0.04]),
            ([-0.13, 0.19], [-0.33, 0.34], [-0.33, 0.04]),

            ([-0.13, 0.04], [-0.33, 0.19], [-0.13, 0.34]),
            ([-0.33, 0.19], [-0.13, 0.04], [-0.13, 0.34]),
            ([-0.33, 0.19], [-0.13, 0.34], [-0.13, 0.04]),

            ([-0.13, 0.04], [-0.13, 0.34], [-0.33, 0.19]),
            ([-0.13, 0.34], [-0.33, 0.19], [-0.13, 0.04]),
            ([-0.13, 0.34], [-0.33, 0.19], [-0.13, 0.04]),
        ]

        self.obj_init_quat_dict = {
            "baked_apple": [1.0, 0.0, 0.0, 0.0],
            "baked_opened_pepsi_can": euler2quat(np.pi / 2, 0, 0),
            "baked_sponge": euler2quat(0, 0, np.pi / 2),
        }

        self.special_density_dict = {"baked_apple": 200}

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        obj_init_options = options.get("obj_init_options", {})
        obj_init_options = obj_init_options.copy()

        obj_init_options = {
            "episode_id": obj_init_options.get(
                "episode_id", 0
                ), 
            "source_obj_id": 0,  # apple
            "target_obj_id": 1,  # orange can
            "init_xys": self._xy_config_per_triplet[0],
            "init_rot_quats": [
                self.obj_init_quat_dict["baked_apple"],
                self.obj_init_quat_dict["baked_opened_pepsi_can"],
                self.obj_init_quat_dict["baked_sponge"],
            ],
        }

        options["model_ids"] = self.triplets[0]
        options["obj_init_options"] = obj_init_options

        obs, info = super().reset(seed=self._episode_seed, options=options)
        return obs, info

    def get_language_instruction(self, **kwargs):
        return "After I finish my drink, can you place the fruit near it for a snack?"


@register_env("AltPlaceBottleNearSpongeEnv-v0", max_episode_steps=80)
class AltPlaceBottleNearSpongeEnv(MoveNearGoogleInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_baked_tex_v0.json"
    _main_rng = 42
    _episode_seed = 42
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_obj_configs()

    def _setup_obj_configs(self):
        self.triplets = [
            ("baked_sponge", "blue_plastic_bottle", "orange"),
        ]

        self._source_obj_ids = [1]  # apple
        self._target_obj_ids = [0]

        self._xy_config_per_triplet = [
            ([-0.33, 0.04], [-0.33, 0.34], [-0.13, 0.19]),
            ([-0.33, 0.34], [-0.33, 0.04], [-0.13, 0.19]),
            ([-0.33, 0.34], [-0.13, 0.19], [-0.33, 0.04]),

            ([-0.33, 0.04], [-0.13, 0.19], [-0.33, 0.34]),
            ([-0.13, 0.19], [-0.33, 0.34], [-0.33, 0.04]),
            ([-0.13, 0.19], [-0.33, 0.34], [-0.33, 0.04]),

            ([-0.13, 0.04], [-0.33, 0.19], [-0.13, 0.34]),
            ([-0.33, 0.19], [-0.13, 0.04], [-0.13, 0.34]),
            ([-0.33, 0.19], [-0.13, 0.34], [-0.13, 0.04]),

            ([-0.13, 0.04], [-0.13, 0.34], [-0.33, 0.19]),
            ([-0.13, 0.34], [-0.33, 0.19], [-0.13, 0.04]),
            ([-0.13, 0.34], [-0.33, 0.19], [-0.13, 0.04]),
        ]

        self.obj_init_quat_dict = {
            "blue_plastic_bottle": euler2quat(np.pi / 2, 0, np.pi / 2),
            "orange": euler2quat(0, 0, np.pi / 2),
            "baked_sponge": euler2quat(0, 0, np.pi / 2),
        }

        self.special_density_dict = {"orange": 200}

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.set_episode_rng(seed)

        obj_init_options = options.get("obj_init_options", {})
        obj_init_options = obj_init_options.copy()

        obj_init_options = {
            "episode_id": obj_init_options.get(
                "episode_id", 0
                ), 
            "source_obj_id": 1,  # apple
            "target_obj_id": 0,  # orange can
            "init_xys": self._xy_config_per_triplet[0],
            "init_rot_quats": [
                self.obj_init_quat_dict["baked_sponge"],
                self.obj_init_quat_dict["blue_plastic_bottle"],
                self.obj_init_quat_dict["orange"],
            ],
        }

        options["model_ids"] = self.triplets[0]
        options["obj_init_options"] = obj_init_options

        obs, info = super().reset(seed=self._episode_seed, options=options)
        return obs, info

    def get_language_instruction(self, **kwargs):
        return "Please move the water bottle to wet the sponge."


@register_env("AltPlaceBottleNearOrangeEnv-v0", max_episode_steps=80)
class AltPlaceBottleNearOrangeEnv(MoveNearGoogleInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_baked_tex_v0.json"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _setup_obj_configs(self):
        self.triplets = [
            ("baked_sponge", "blue_plastic_bottle", "orange"),
        ]

        self._source_obj_ids = [1]
        self._target_obj_ids = [2]

        self._xy_config_per_triplet = [
            ([-0.33, 0.04], [-0.33, 0.34], [-0.13, 0.19]),
            ([-0.33, 0.34], [-0.33, 0.04], [-0.13, 0.19]),
            ([-0.33, 0.34], [-0.13, 0.19], [-0.33, 0.04]),

            ([-0.33, 0.04], [-0.13, 0.19], [-0.33, 0.34]),
            ([-0.13, 0.19], [-0.33, 0.34], [-0.33, 0.04]),
            ([-0.13, 0.19], [-0.33, 0.34], [-0.33, 0.04]),

            ([-0.13, 0.04], [-0.33, 0.19], [-0.13, 0.34]),
            ([-0.33, 0.19], [-0.13, 0.04], [-0.13, 0.34]),
            ([-0.33, 0.19], [-0.13, 0.34], [-0.13, 0.04]),

            ([-0.13, 0.04], [-0.13, 0.34], [-0.33, 0.19]),
            ([-0.13, 0.34], [-0.33, 0.19], [-0.13, 0.04]),
            ([-0.13, 0.34], [-0.33, 0.19], [-0.13, 0.04]),
        ]

        self.obj_init_quat_dict = {
            "blue_plastic_bottle": euler2quat(np.pi / 2, 0, np.pi / 2),
            "orange": euler2quat(0, 0, np.pi / 2),
            "baked_sponge": euler2quat(0, 0, np.pi / 2),
        }

        self.special_density_dict = {"orange": 200}

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.set_episode_rng(seed)

        obj_init_options = options.get("obj_init_options", {})
        obj_init_options = obj_init_options.copy()

        obj_init_options = {
            "episode_id": obj_init_options.get(
                "episode_id", 0
                ), 
            "source_obj_id": 1,
            "target_obj_id": 2,
            "init_xys": self._xy_config_per_triplet[0],
            "init_rot_quats": [
                self.obj_init_quat_dict["baked_sponge"],
                self.obj_init_quat_dict["blue_plastic_bottle"],
                self.obj_init_quat_dict["orange"],
            ],
        }

        options["model_ids"] = self.triplets[0]
        options["obj_init_options"] = obj_init_options

        obs, info = super().reset(seed=self._episode_seed, options=options)
        return obs, info

    def get_language_instruction(self, **kwargs):
        return "I want to wash the orange with water. Please move the correct water-containing object to the orange."


# =============================== bridge ===============================

from .put_on_in_scene import PutEggplantInBasketScene, PutOnBridgeInSceneEnv

@register_env("AltDryEgglpantInSceneEnv-v0", max_episode_steps=120)
class AltDryEgglpantInSceneEnv(PutEggplantInBasketScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def get_language_instruction(self):
        return "dry the eggplant in yellow basket after wash it."

@register_env("AltDryOrangeInSceneEnv-v0", max_episode_steps=120)
class AltDryOrangeInSceneEnv(PutOnBridgeInSceneEnv):
    def __init__(self, **kwargs):
        self.DEFAULT_ASSET_ROOT = "{ASSET_DIR}/custom"
        self.DEFAULT_SCENE_ROOT = "{ASSET_DIR}/hab2_bench_assets"
        self.DEFAULT_MODEL_JSON = "info_bridge_custom_v0.json"
        
        source_obj_name = 'orange'
        target_obj_name = "dummy_sink_target_plane"  # invisible

        target_xy = np.array([-0.125, 0.025])
        xy_center = [-0.105, 0.206]

        half_span_x = 0.01
        half_span_y = 0.015
        num_x = 2
        num_y = 4

        grid_pos = []
        for x in np.linspace(-half_span_x, half_span_x, num_x):
            for y in np.linspace(-half_span_y, half_span_y, num_y):
                grid_pos.append(np.array([x + xy_center[0], y + xy_center[1]]))

        xy_configs = [np.stack([pos, target_xy], axis=0) for pos in grid_pos]

        quat_configs = [
            np.array([
                euler2quat(0, 0, 0, 'sxyz'),
                [1, 0, 0, 0]
            ]),
            np.array([
                euler2quat(0, 0, 1 * np.pi / 4, 'sxyz'),
                [1, 0, 0, 0]
            ]),
            np.array([
                euler2quat(0, 0, -1 * np.pi / 4, 'sxyz'),
                [1, 0, 0, 0]
            ]),
        ]

        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            rgb_always_overlay_objects=['sink', 'dummy_sink_target_plane'],
            **kwargs,
        )

    def get_language_instruction(self, **kwargs):
        return "After wash the orange in the sink, I want you to dry it in the yellow basket."

    def _load_model(self):
        super()._load_model()
        self.sink_id = 'sink'
        self.sink = self._build_actor_helper(
            self.sink_id,
            self._scene,
            density=self.model_db[self.sink_id].get("density", 1000),
            physical_material=self._scene.create_physical_material(
                static_friction=self.obj_static_friction, dynamic_friction=self.obj_dynamic_friction, restitution=0.0
            ),
            root_dir=self.asset_root,
        )
        self.sink.name = self.sink_id

    def _initialize_actors(self):
        # Move the robot far away to avoid collision
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        self.sink.set_pose(sapien.Pose(
            [-0.16, 0.13, 0.88],
            [1, 0, 0, 0]
        ))
        self.sink.lock_motion()

        super()._initialize_actors()

    def evaluate(self, *args, **kwargs):
        return super().evaluate(success_require_src_completely_on_target=False, 
                                z_flag_required_offset=0.06,
                                *args, **kwargs)

    def _setup_prepackaged_env_init_config(self):
        ret = super()._setup_prepackaged_env_init_config()
        ret["robot"] = "widowx_sink_camera_setup"
        ret["scene_name"] = "bridge_table_1_v2"
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/bridge_sink.png"
        )
        return ret

    def _additional_prepackaged_config_reset(self, options):
        # use prepackaged robot evaluation configs under visual matching setup
        options["robot_init_options"] = {
            "init_xy": [0.127, 0.06],
            "init_rot_quat": [0, 0, 0, 1],
        }
        return False # in env reset options, no need to reconfigure the environment

    def _setup_lighting(self):
        if self.bg_name is not None:
            return

        shadow = self.enable_shadow

        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        self._scene.add_directional_light(
            [0, 0, -1],
            [0.3, 0.3, 0.3],
            position=[0, 0, 1],
            shadow=shadow,
            scale=5,
            shadow_map_size=2048,
        )

# =============================== bridge v2 ===============================

@register_env("AltPutSponeOnPlateInScene-v0", max_episode_steps=60)
class AltPutSponeOnPlateInScene(PutOnBridgeInSceneEnv):
    def __init__(self,
        source_obj_name: str = None,
        target_obj_name: str = None,
        other_obj_names: List[str] = None,
        xy_configs: List[np.ndarray] = None,
        quat_configs: List[np.ndarray] = None,
        **kwargs,
    ):
        self.DEFAULT_ASSET_ROOT = "{ASSET_DIR}/custom"
        self.DEFAULT_SCENE_ROOT = "{ASSET_DIR}/hab2_bench_assets"
        self.DEFAULT_MODEL_JSON = "info_bridge_custom_v0.json"

        if source_obj_name is None: # skip init object
        
            source_obj_name = 'bridge_spoon_generated_modified'
            target_obj_name = "bridge_plate_objaverse_larger"

            additional_obj_name = ["orange", "table_cloth_generated_shorter"]

            if other_obj_names is None:
                self._other_obj_names = additional_obj_name
            else:
                self._other_obj_names = other_obj_names

        # Define positions for all objects

        if xy_configs is None:

            xy_center = np.array([-0.18, 0.08])
            half_edge_length_xs = [0.075, 0.1]
            half_edge_length_ys = [0.1, 0.12]
            xy_configs = []

            for (half_edge_length_x, half_edge_length_y) in zip(
                half_edge_length_xs, half_edge_length_ys
            ):
                grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
                grid_pos = (
                    grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
                    + xy_center[None]
                )

                for i, grid_pos_1 in enumerate(grid_pos):
                    for j, grid_pos_2 in enumerate(grid_pos):
                        if i != j:
                            additional_positions = [grid_pos[k] for k in range(len(grid_pos)) if k != i and k != j]
                            xy_config = np.array([grid_pos_1, grid_pos_2] + additional_positions) # size: 4 x 2
                            xy_configs.append(xy_config)

            quat_configs = [ # no rotation for orange
                np.array([euler2quat(0, 0, np.pi), [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]),
            ]
        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def _set_model(self, model_ids, model_scales):
        """Set the model id and scale. If not provided, choose a triplet randomly from self.model_ids."""
        self.episode_model_ids = [self._source_obj_name, self._target_obj_name] + self._other_obj_names
        # model scales
        reconfigure = False
        if model_scales is None:
            model_scales = []
            for model_id in self.episode_model_ids:
                this_available_model_scales = self.model_db[model_id].get(
                    "scales", None
                )
                if this_available_model_scales is None:
                    model_scales.append(1.0)
                else:
                    model_scales.append(
                        random_choice(this_available_model_scales, self._episode_rng)
                    )
        if not self._list_equal(model_scales, self.episode_model_scales):
            self.episode_model_scales = model_scales
            reconfigure = True

        # model bbox sizes
        model_bbox_sizes = []
        for model_id, model_scale in zip(
            self.episode_model_ids, self.episode_model_scales
        ):
            model_info = self.model_db[model_id]
            if "bbox" in model_info:
                bbox = model_info["bbox"]
                bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
                model_bbox_sizes.append(bbox_size * model_scale)
            else:
                raise ValueError(f"Model {model_id} does not have bbox info.")
        self.episode_model_bbox_sizes = model_bbox_sizes

        return reconfigure

    def _load_model(self):
        self.episode_objs = []
        for (model_id, model_scale) in zip(
            self.episode_model_ids, self.episode_model_scales
        ):
            density = self.model_db[model_id].get("density", 1000)

            obj = self._build_actor_helper(
                model_id,
                self._scene,
                scale=model_scale,
                density=density,
                physical_material=self._scene.create_physical_material(
                    static_friction=self.obj_static_friction,
                    dynamic_friction=self.obj_dynamic_friction,
                    restitution=0.0,
                ),
                root_dir=self.asset_root,
            )
            obj.name = model_id
            self.episode_objs.append(obj)

    def get_language_instruction(self, **kwargs):
        return "put the tool that can be used to feed a baby on the plate"

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.set_episode_rng(seed)

        obj_init_options = options.get("obj_init_options", {})
        obj_init_options = obj_init_options.copy()
        episode_id = obj_init_options.get(
            "episode_id",
            self._episode_rng.randint(len(self._xy_configs) * len(self._quat_configs)),
        )
        xy_config = self._xy_configs[
            (episode_id % (len(self._xy_configs) * len(self._quat_configs)))
            // len(self._quat_configs)
        ]
        quat_config = self._quat_configs[episode_id % len(self._quat_configs)]
        # make sure the source is always at 0 and the target is always at 1
        options["model_ids"] = [self._source_obj_name, self._target_obj_name] + self._other_obj_names
        obj_init_options["source_obj_id"] = 0
        obj_init_options["target_obj_id"] = 1
        obj_init_options["init_xys"] = xy_config # size: 4 x 2
        obj_init_options["init_rot_quats"] = quat_config # size: 4 x 4
        options["obj_init_options"] = obj_init_options

        obs, info = super().reset(seed=self._episode_seed, options=options)
        info.update({"episode_id": episode_id})
        return obs, info

@register_env("AltPutOrangeOnPlateInScene-v0", max_episode_steps=60)
class AltPutOrangeOnPlateInScene(AltPutSponeOnPlateInScene):
    def __init__(self,
        source_obj_name: str = None,
        target_obj_name: str = None,
        other_obj_names: List[str] = None,
        xy_configs: List[np.ndarray] = None,
        quat_configs: List[np.ndarray] = None,
        **kwargs,
    ):
        self.DEFAULT_ASSET_ROOT = "{ASSET_DIR}/custom"
        self.DEFAULT_SCENE_ROOT = "{ASSET_DIR}/hab2_bench_assets"
        self.DEFAULT_MODEL_JSON = "info_bridge_custom_v0.json"
        
        source_obj_name = "orange"
        target_obj_name = "bridge_plate_objaverse_larger"

        additional_obj_name = ["bridge_spoon_generated_modified", "opened_coke_can"]

        if other_obj_names is None:
            self._other_obj_names = additional_obj_name
        else:
            self._other_obj_names = other_obj_names

        # Define positions for all objects

        xy_center = np.array([-0.18, 0.08])
        half_edge_length_xs = [0.075, 0.1]
        half_edge_length_ys = [0.1, 0.12]
        xy_configs = []

        for (half_edge_length_x, half_edge_length_y) in zip(
            half_edge_length_xs, half_edge_length_ys
        ):
            grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
            grid_pos = (
                grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
                + xy_center[None]
            )

            for i, grid_pos_1 in enumerate(grid_pos):
                for j, grid_pos_2 in enumerate(grid_pos):
                    if i != j:
                        additional_positions = [grid_pos[k] for k in range(len(grid_pos)) if k != i and k != j]
                        xy_config = np.array([grid_pos_1, grid_pos_2] + additional_positions) # size: 4 x 2
                        xy_configs.append(xy_config)

        quat_configs = [ # no rotation for orange
            np.array([euler2quat(0, 0, np.pi), [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]),
        ]
        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def get_language_instruction(self, **kwargs):
        return "I do not want the drink, please put the fruit on the plate"