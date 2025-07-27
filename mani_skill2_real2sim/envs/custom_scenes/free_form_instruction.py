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
from .grasp_single_in_scene import GraspSingleCustomOrientationInSceneEnv, GraspSingleCustomInSceneEnv

@register_env("FreeGraspSingleBananaInScene-v0", max_episode_steps=80)
class FreeGraspSingleBananaInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_v1.json"
    def __init__(self, **kwargs):
        kwargs.pop("model_ids", None)
        kwargs["model_ids"] = ["custom_banana"]
        super().__init__(**kwargs)

    def get_language_instruction(self, **kwargs):
        return "grab the banana"

@register_env("FreeGraspSingleBananaInScene-v1", max_episode_steps=80)
class FreeGraspSingleBananaInSceneEnvV1(FreeGraspSingleBananaInSceneEnv):
    def get_language_instruction(self, **kwargs):
        return "Pick up the yellow fruit on the table."


@register_env("FreeGraspSingleBananaInScene-v2", max_episode_steps=80)
class FreeGraspSingleBananaInSceneEnvV2(FreeGraspSingleBananaInSceneEnv):
    def get_language_instruction(self, **kwargs):
        return "saisis la banane"

        
@register_env("FreeGraspSinglePineappleInScene-v0", max_episode_steps=80)
class FreeGraspSinglePineappleInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_v1.json"
    def __init__(self, **kwargs):
        kwargs.pop("model_ids", None)
        kwargs["model_ids"] = ["custom_pineapple"]
        super().__init__(**kwargs)

    def get_language_instruction(self, **kwargs):
        obj_name = self._get_instruction_obj_name(self.obj.name)
        task_description = f"grab the pineapple"
        return task_description


@register_env("FreeGraspSinglePearInScene-v0", max_episode_steps=80)
class FreeGraspSinglePearInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_v1.json"
    orientations_dict = {
            "upright": euler2quat(np.pi / 2, 0, 0),
            "laid_vertically": euler2quat(0, 0, np.pi / 2),
            "lr_switch": euler2quat(0, 0, np.pi),
        }
    def __init__(self, **kwargs):
        kwargs.pop("model_ids", None)
        kwargs["model_ids"] = ["custom_pear"]
        super().__init__(**kwargs)

    def get_language_instruction(self, **kwargs):
        obj_name = self._get_instruction_obj_name(self.obj.name)
        task_description = f"pick the pear"
        return task_description

    def reset(self, seed=None, options=None):

        if 'obj_init_options' in options:
            options['obj_init_options']["orientation"] = "upright"
        obs, info = super().reset(seed=seed, options=options)
        return obs, info


@register_env("FreeGraspSinglePearInScene-v1", max_episode_steps=80)
class FreeGraspSinglePearInSceneEnvV1(FreeGraspSinglePearInSceneEnv):
    def get_language_instruction(self, **kwargs):
        return f"Please retrieve the green teardrop-shaped fruit from the surface."

@register_env("FreeGraspSinglePearInScene-v2", max_episode_steps=80)
class FreeGraspSinglePearInSceneEnvV2(FreeGraspSinglePearInSceneEnv):
    def get_language_instruction(self, **kwargs):
        return f"Veuillez récupérer le fruit en forme de goutte verte de la surface."

@register_env("FreeGraspSinglePlumInScene-v0", max_episode_steps=80)
class FreeGraspSinglePlumInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_v1.json"
    orientations_dict = {
            "upright": euler2quat(np.pi / 2, 0, 0),
            "laid_vertically": euler2quat(0, 0, np.pi / 2),
            "lr_switch": euler2quat(0, 0, np.pi),
        }
    def __init__(self, **kwargs):
        kwargs.pop("model_ids", None)
        kwargs["model_ids"] = ["custom_plum"]
        super().__init__(**kwargs)

    def get_language_instruction(self, **kwargs):
        obj_name = self._get_instruction_obj_name(self.obj.name)
        task_description = f"give me the purple fruit on the tabel"
        return task_description

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options['model_scale'] = 1.3
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

@register_env("FreeGraspSinglePlumInScene-v1", max_episode_steps=80)
class FreeGraspSinglePlumInSceneEnvV1(FreeGraspSinglePlumInSceneEnv):
    def get_language_instruction(self, **kwargs):
        return f"Pick up the small round purple fruit"


@register_env("FreeGraspSinglePumpkinInScene-v0", max_episode_steps=80)
class FreeGraspSinglePumpkinInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_v1.json"
    orientations_dict = {
            "upright": euler2quat(np.pi / 2, 0, 0),
            "laid_vertically": euler2quat(0, 0, np.pi / 2),
            "lr_switch": euler2quat(0, 0, np.pi),
        }
    def __init__(self, **kwargs):
        kwargs.pop("model_ids", None)
        kwargs["model_ids"] = ["custom_pumpkin"]
        super().__init__(**kwargs)

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options['model_scale'] = 0.6
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def get_language_instruction(self, **kwargs):
        obj_name = self._get_instruction_obj_name(self.obj.name)
        task_description = f"give me the pumpkin"
        return task_description


@register_env("FreeGraspSinglePumpkinInScene-v1", max_episode_steps=80)
class FreeGraspSinglePumpkinInSceneEnvV1(FreeGraspSinglePumpkinInSceneEnv):
    def get_language_instruction(self, **kwargs):
        return f"Grab the orange squash and bring it to me."

@register_env("FreeGraspSingleLemonInScene-v0", max_episode_steps=80)
class FreeGraspSingleLemonInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_v1.json"
    orientations_dict = {
            "upright": euler2quat(np.pi / 2, 0, 0),
            "laid_vertically": euler2quat(0, 0, np.pi / 2),
            "lr_switch": euler2quat(0, 0, np.pi),
        }
    def __init__(self, **kwargs):
        kwargs.pop("model_ids", None)
        kwargs["model_ids"] = ["custom_lemon"]
        super().__init__(**kwargs)

    def get_language_instruction(self, **kwargs):
        obj_name = self._get_instruction_obj_name(self.obj.name)
        task_description = f"lift the lemon"
        return task_description

@register_env("FreeGraspSingleLemonInScene-v1", max_episode_steps=80)
class FreeGraspSingleLemonInSceneEnvV1(FreeGraspSingleLemonInSceneEnv):
    def get_language_instruction(self, **kwargs):
        return f"Lift the sour yellow fruit sitting on the table."

@register_env("FreeGraspSingleLemonInScene-v2", max_episode_steps=80)
class FreeGraspSingleLemonInSceneEnvV2(FreeGraspSingleLemonInSceneEnv):
    def get_language_instruction(self, **kwargs):
        return f"lève le citron"


@register_env("FreeGraspSingleOrangeInScene-v0", max_episode_steps=80)
class FreeGraspSingleOrangeInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_v1.json"
    orientations_dict = {
            "upright": euler2quat(np.pi / 2, 0, 0),
            "laid_vertically": euler2quat(0, 0, np.pi / 2),
            "lr_switch": euler2quat(0, 0, np.pi),
        }
    def __init__(self, **kwargs):
        kwargs.pop("model_ids", None)
        kwargs["model_ids"] = ["custom_green_orange"]
        super().__init__(**kwargs)

    def get_language_instruction(self, **kwargs):
        obj_name = self._get_instruction_obj_name(self.obj.name)
        task_description = f"give me the green orange"
        return task_description

@register_env("FreeGraspSingleOrangeInScene-v1", max_episode_steps=80)
class FreeGraspSingleOrangeInSceneEnvV1(FreeGraspSingleOrangeInSceneEnv):
    def get_language_instruction(self, **kwargs):
        return f"Pick up the citrus fruit with green skin and hand it to me."

@register_env("FreeGraspSingleOpenedPepsiCanInScene-v0", max_episode_steps=80)
class FreeGraspSingleOpenedPepsiCanInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop("model_ids", None)
        kwargs["model_ids"] = ["opened_pepsi_can"]
        super().__init__(**kwargs)
    
    def get_language_instruction(self, **kwargs):
        obj_name = self._get_instruction_obj_name(self.obj.name)
        task_description = f"I want you to lift the can before serve the drink"
        return task_description

@register_env("FreeGraspSingleOpenedPepsiCanInScene-v1", max_episode_steps=80)
class FreeGraspSingleOpenedPepsiCanInSceneEnvV1(FreeGraspSingleOpenedPepsiCanInSceneEnv):
    def get_language_instruction(self, **kwargs):
        return f"Lift the already opened soda can so we can clean up."


@register_env("FreeGraspSingle7upCanInScene-v0", max_episode_steps=80)
class FreeGraspSingle7upCanInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_v1.json"
    def __init__(self, **kwargs):
        kwargs.pop("model_ids", None)
        kwargs["model_ids"] = ["7up_can"]
        self.distractor_model_ids = [
            "opened_pepsi_can",
            "custom_green_orange"
            ]
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
        task_description = f"pick the green and white can"
        return task_description

@register_env("FreeGraspSingle7upCanInScene-v1", max_episode_steps=80)
class FreeGraspSingle7upCanInSceneEnvV1(FreeGraspSingle7upCanInSceneEnv):
    def get_language_instruction(self, **kwargs):
        return f"Grab the green and white fizzy drink can from the workspace."

@register_env("FreeGraspSingle7upCanInScene-v2", max_episode_steps=80)
class FreeGraspSingle7upCanInSceneEnvV2(FreeGraspSingle7upCanInSceneEnv):
    def get_language_instruction(self, **kwargs):
        return f"toma la lata verde y blanca"



@register_env("FreeGraspSingleOpened7upCanInScene-v0", max_episode_steps=80)
class FreeGraspSingleOpened7upCanInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop("model_ids", None)
        kwargs["model_ids"] = ["opened_7up_can"]

        self.distractor_model_ids = [
            "blue_plastic_bottle",
            "orange"
            ]
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
        task_description = f"recycle the opened can"
        return task_description

@register_env("FreeGraspSingleOpened7upCanInScene-v1", max_episode_steps=80)
class FreeGraspSingleOpened7upCanInSceneEnvV1(FreeGraspSingleOpened7upCanInSceneEnv):
    def get_language_instruction(self, **kwargs):
        return "Recicla la lata abierta"

@register_env("FreeGraspSingleSpriteCanInScene-v0", max_episode_steps=80)
class FreeGraspSingleSpriteCanInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop("model_ids", None)
        kwargs["model_ids"] = ["sprite_can"]
        super().__init__(**kwargs)
    
    def get_language_instruction(self, **kwargs):
        obj_name = self._get_instruction_obj_name(self.obj.name)
        task_description = f"give me the drink"
        return task_description

@register_env("FreeGraspSingleSpriteCanInScene-v1", max_episode_steps=80)
class FreeGraspSingleSpriteCanInSceneEnvV1(FreeGraspSingleSpriteCanInSceneEnv):
    def get_language_instruction(self, **kwargs):
        return f"Bring me the lime-green drink can from the table."


@register_env("FreeGraspSingleSpriteCanInScene-v2", max_episode_steps=80)
class FreeGraspSingleSpriteCanInSceneEnvV2(FreeGraspSingleSpriteCanInSceneEnv):
    def get_language_instruction(self, **kwargs):
        return f"Trae la lata de bebida verde lima de la mesa."


from .open_drawer_in_scene import OpenDrawerCustomInSceneEnv, CloseDrawerCustomInSceneEnv



@register_env("FreeOpenTopDrawerCustomInScene-v0", max_episode_steps=113)
class FreeOpenTopDrawerCustomInSceneEnv(OpenDrawerCustomInSceneEnv):
    drawer_ids = ["top"]

    def get_language_instruction(self, **kwargs):
        return f"Pull open the upper drawer"


@register_env("FreeOpenMiddleDrawerCustomInScene-v0", max_episode_steps=113)
class FreeOpenMiddleDrawerCustomInSceneEnv(OpenDrawerCustomInSceneEnv):
    drawer_ids = ["middle"]

    def get_language_instruction(self, **kwargs):
        return f"Pull the center drawer open"


@register_env("FreeOpenBottomDrawerCustomInScene-v0", max_episode_steps=113)
class FreeOpenBottomDrawerCustomInSceneEnv(OpenDrawerCustomInSceneEnv):
    drawer_ids = ["bottom"]

    def get_language_instruction(self, **kwargs):
        return f"Please access the drawer that is positioned at the bottom"

@register_env("FreeCloseTopDrawerCustomInScene-v0", max_episode_steps=113)
class FreeCloseTopDrawerCustomInSceneEnv(CloseDrawerCustomInSceneEnv):
    drawer_ids = ["top"]
    
    def get_language_instruction(self, **kwargs):
        return f"Close the topmost drawer."


@register_env("FreeCloseMiddleDrawerCustomInScene-v0", max_episode_steps=113)
class FreeCloseMiddleDrawerCustomInSceneEnv(CloseDrawerCustomInSceneEnv):
    drawer_ids = ["middle"]
    
    def get_language_instruction(self, **kwargs):
        return f"Shut the central drawer."


@register_env("FreeCloseBottomDrawerCustomInScene-v0", max_episode_steps=113)
class FreeCloseBottomDrawerCustomInSceneEnv(CloseDrawerCustomInSceneEnv):
    drawer_ids = ["bottom"]
    
    def get_language_instruction(self, **kwargs):
        return f"Make sure the bottom drawer is closed."



@register_env("FreeOpenTopDrawerCustomInScene-v1", max_episode_steps=113)
class FreeOpenTopDrawerCustomInSceneEnvV1(OpenDrawerCustomInSceneEnv):
    drawer_ids = ["top"]

    def get_language_instruction(self, **kwargs):
        return f"Slide out the top drawer."


@register_env("FreeOpenMiddleDrawerCustomInScene-v1", max_episode_steps=113)
class FreeOpenMiddleDrawerCustomInSceneEnvV1(OpenDrawerCustomInSceneEnv):
    drawer_ids = ["middle"]

    def get_language_instruction(self, **kwargs):
        return f"Open the drawer located in the middle section."


@register_env("FreeOpenBottomDrawerCustomInScene-v1", max_episode_steps=113)
class FreeOpenBottomDrawerCustomInSceneEnvV1(OpenDrawerCustomInSceneEnv):
    drawer_ids = ["bottom"]

    def get_language_instruction(self, **kwargs):
        return f"Pull out the lowest drawer from the cabinet."


@register_env("FreeCloseTopDrawerCustomInScene-v1", max_episode_steps=113)
class FreeCloseTopDrawerCustomInSceneEnvV1(CloseDrawerCustomInSceneEnv):
    drawer_ids = ["top"]
    
    def get_language_instruction(self, **kwargs):
        return f"Gently push the upper drawer to shut it."


@register_env("FreeCloseMiddleDrawerCustomInScene-v1", max_episode_steps=113)
class FreeCloseMiddleDrawerCustomInSceneEnvV1(CloseDrawerCustomInSceneEnv):
    drawer_ids = ["middle"]
    
    def get_language_instruction(self, **kwargs):
        return f"Securely close the drawer that's in the middle."


@register_env("FreeCloseBottomDrawerCustomInScene-v1", max_episode_steps=113)
class FreeCloseBottomDrawerCustomInSceneEnvV1(CloseDrawerCustomInSceneEnv):
    drawer_ids = ["bottom"]
    
    def get_language_instruction(self, **kwargs):
        return f"Ensure the drawer at the base is completely closed."


from .move_near_in_scene import MoveNearGoogleInSceneEnv

@register_env("FreePlaceLemonNearPearEnv-v0", max_episode_steps=80)
class FreePlaceLemonNearPearEnv(MoveNearGoogleInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_v1.json"
    _main_rng = 42
    _episode_seed = 42
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_obj_configs()

    def _setup_obj_configs(self):
        self.triplets = [
            ("custom_lemon", "custom_pear", "baked_sponge_v2"),

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
            ([-0.13, 0.34], [-0.13, 0.04], [-0.33, 0.19]),
        ]

        self.obj_init_quat_dict = {
            "custom_lemon": euler2quat(np.pi / 2, 0, 0),
            "custom_pear": euler2quat(np.pi / 2, 0, 0),
            "baked_sponge_v2": euler2quat(0, 0, np.pi / 2),
        }

        self.special_density_dict = {"custom_lemon": 200}

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
                self.obj_init_quat_dict["custom_lemon"],
                self.obj_init_quat_dict["custom_pear"],
                self.obj_init_quat_dict["baked_sponge_v2"],
            ],
        }

        options["model_ids"] = self.triplets[0]
        options["obj_init_options"] = obj_init_options

        obs, info = super().reset(seed=self._episode_seed, options=options)
        return obs, info

    def get_language_instruction(self, **kwargs):
        return "Can you place the lemon near pear?"

@register_env("FreePlaceLemonNearPearEnv-v1", max_episode_steps=80)
class FreePlaceLemonNearPearEnvV1(FreePlaceLemonNearPearEnv):
    def get_language_instruction(self, **kwargs):
        return "Can you place the sourest fruit to the less sour one?"


@register_env("FreePlaceLemonNearPearEnv-v2", max_episode_steps=80)
class FreePlaceLemonNearPearEnvV2(FreePlaceLemonNearPearEnv):
    def get_language_instruction(self, **kwargs):
        return "¿Puedes colocar la fruta más ácida junto a la menos ácida?"

@register_env("FreePlaceLNearVEnv-v0", max_episode_steps=80)
class FreePlaceLNearVEnv(MoveNearGoogleInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_v1.json"
    _main_rng = 42
    _episode_seed = 42
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_obj_configs()

    def _setup_obj_configs(self):
        self.triplets = [
            ("V", "L", "A"),
        ]

        self._source_obj_ids = [1]  # apple
        self._target_obj_ids = [0]

        self._xy_config_per_triplet = [
            ([-0.33, 0.04], [-0.33, 0.34], [-0.13, 0.19]),
            ([-0.33, 0.34], [-0.33, 0.04], [-0.13, 0.19]),
            ([-0.33, 0.34], [-0.13, 0.19], [-0.33, 0.04]),

            ([-0.33, 0.04], [-0.13, 0.19], [-0.33, 0.34]),
            ([-0.13, 0.19], [-0.33, 0.34], [-0.33, 0.04]),
            ([-0.13, 0.19], [-0.33, 0.04], [-0.33, 0.34]),

            ([-0.13, 0.04], [-0.33, 0.19], [-0.13, 0.34]),
            ([-0.33, 0.19], [-0.13, 0.04], [-0.13, 0.34]),
            ([-0.33, 0.19], [-0.13, 0.34], [-0.13, 0.04]),

            ([-0.13, 0.04], [-0.13, 0.34], [-0.33, 0.19]),
            ([-0.13, 0.34], [-0.33, 0.19], [-0.13, 0.04]),
            ([-0.13, 0.34], [-0.13, 0.04], [-0.33, 0.19]),
        ]

        self.obj_init_quat_dict = {
            "L": euler2quat(0, 0, np.pi / 2),
            "A": euler2quat(0, 0, np.pi / 2),
            "V": euler2quat(0, 0, np.pi / 2),
        }

        self.special_density_dict = {"A": 200}

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
                self.obj_init_quat_dict["V"],
                self.obj_init_quat_dict["L"],
                self.obj_init_quat_dict["A"],
            ],
        }

        options["model_ids"] = self.triplets[0]
        options["obj_init_options"] = obj_init_options

        obs, info = super().reset(seed=self._episode_seed, options=options)
        return obs, info

    def get_language_instruction(self, **kwargs):
        return "Please move the 'L' to 'V'"



@register_env("FreePlaceLNearAEnv-v0", max_episode_steps=80)
class FreePlaceLNearAEnv(MoveNearGoogleInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_v1.json"
    _main_rng = 42
    _episode_seed = 42
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_obj_configs()

    def _setup_obj_configs(self):
        self.triplets = [
            ("A", "L", "V"),
        ]

        self._source_obj_ids = [1]  # apple
        self._target_obj_ids = [0]

        self._xy_config_per_triplet = [
            ([-0.33, 0.04], [-0.33, 0.34], [-0.13, 0.19]),
            ([-0.33, 0.34], [-0.33, 0.04], [-0.13, 0.19]),
            ([-0.33, 0.34], [-0.13, 0.19], [-0.33, 0.04]),

            ([-0.33, 0.04], [-0.13, 0.19], [-0.33, 0.34]),
            ([-0.13, 0.19], [-0.33, 0.34], [-0.33, 0.04]),
            ([-0.13, 0.19], [-0.33, 0.04], [-0.33, 0.34]),

            ([-0.13, 0.04], [-0.33, 0.19], [-0.13, 0.34]),
            ([-0.33, 0.19], [-0.13, 0.04], [-0.13, 0.34]),
            ([-0.33, 0.19], [-0.13, 0.34], [-0.13, 0.04]),

            ([-0.13, 0.04], [-0.13, 0.34], [-0.33, 0.19]),
            ([-0.13, 0.34], [-0.33, 0.19], [-0.13, 0.04]),
            ([-0.13, 0.34], [-0.13, 0.04], [-0.33, 0.19]),
        ]

        self.obj_init_quat_dict = {
            "L": euler2quat(0, 0, np.pi / 2),
            "A": euler2quat(0, 0, np.pi / 2),
            "V": euler2quat(0, 0, np.pi / 2),
        }

        self.special_density_dict = {"A": 200}

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
                self.obj_init_quat_dict["V"],
                self.obj_init_quat_dict["L"],
                self.obj_init_quat_dict["A"],
            ],
        }

        options["model_ids"] = self.triplets[0]
        options["obj_init_options"] = obj_init_options

        obs, info = super().reset(seed=self._episode_seed, options=options)
        return obs, info

    def get_language_instruction(self, **kwargs):
        return "Please move the 'L' to 'A'"


@register_env("FreePlaceBottleNearOrangeEnv-v0", max_episode_steps=80)
class FreePlaceBottleNearOrangeEnv(MoveNearGoogleInSceneEnv):
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
            ([-0.13, 0.19], [-0.33, 0.04], [-0.33, 0.34]),

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
        return "pick the plastic bottle and then place it near the orange."


@register_env("FreePlaceMugNearPlaystationEnv-v0", max_episode_steps=80)
class FreePlaceMugNearPlaystationEnv(MoveNearGoogleInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_v1.json"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _setup_obj_configs(self):
        self.triplets = [
            ("pot", "coffee_mug", "play_station"),
        ]

        self._source_obj_ids = [1]
        self._target_obj_ids = [2]

        self._xy_config_per_triplet = [
            ([-0.33, 0.04], [-0.33, 0.34], [-0.13, 0.19]),
            ([-0.33, 0.34], [-0.33, 0.04], [-0.13, 0.19]),
            ([-0.33, 0.34], [-0.13, 0.19], [-0.33, 0.04]),

            ([-0.33, 0.04], [-0.13, 0.19], [-0.33, 0.34]),
            ([-0.13, 0.19], [-0.33, 0.34], [-0.33, 0.04]),
            ([-0.13, 0.19], [-0.33, 0.04], [-0.33, 0.34]),

            ([-0.13, 0.04], [-0.33, 0.19], [-0.13, 0.34]),
            ([-0.33, 0.19], [-0.13, 0.04], [-0.13, 0.34]),
            ([-0.33, 0.19], [-0.13, 0.34], [-0.13, 0.04]),

            ([-0.13, 0.04], [-0.13, 0.34], [-0.33, 0.19]),
            ([-0.13, 0.34], [-0.33, 0.19], [-0.13, 0.04]),
            ([-0.13, 0.34], [-0.33, 0.19], [-0.13, 0.04]),
        ]

        self.obj_init_quat_dict = {
            "coffee_mug": [0,0,0,1], # euler2quat(np.pi / 2, 0, np.pi / 2),
            "play_station":  euler2quat(np.pi/2, -np.pi/2, 0),
            "pot":  [0,0,0,1], # euler2quat(0, 0, np.pi / 2),
        }

        self.special_density_dict = {"play_station": 200}

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
                self.obj_init_quat_dict["pot"],
                self.obj_init_quat_dict["coffee_mug"],
                self.obj_init_quat_dict["play_station"],
            ],
        }

        options["model_ids"] = self.triplets[0]
        options["obj_init_options"] = obj_init_options

        obs, info = super().reset(seed=self._episode_seed, options=options)
        return obs, info

    def get_language_instruction(self, **kwargs):
        return "pick the mug and then place it near the play station."


@register_env("FreePlaceMugNearPlaystationEnv-v1", max_episode_steps=80)
class FreePlaceMugNearPlaystationEnvV1(FreePlaceMugNearPlaystationEnv):
    def get_language_instruction(self, **kwargs):
        return "grab the mug and place it next to the play station."

from .place_in_closed_drawer_in_scene import PlaceIntoClosedTopDrawerCustomInSceneEnv

@register_env("FreePlaceIntoTopDrawerEnv-v0", max_episode_steps=200)
class FreePlaceIntoTopDrawerEnv(PlaceIntoClosedTopDrawerCustomInSceneEnv):
    def get_language_instruction(self, **kwargs):
        if self.cur_subtask_id == 0:
            return f"Pull out the {self.drawer_id} drawer"
        else:
            model_name = self._get_instruction_obj_name(self.model_id)
            return f"place {model_name} into the open drawer"

@register_env("FreePlaceIntoTopDrawerEnv-v1", max_episode_steps=200)
class FreePlaceIntoTopDrawerEnvV1(PlaceIntoClosedTopDrawerCustomInSceneEnv):
    def get_language_instruction(self, **kwargs):
        if self.cur_subtask_id == 0:
            return f"Pull the {self.drawer_id} drawer out"
        else:
            model_name = self._get_instruction_obj_name(self.model_id)
            return f"place the {model_name} into the drawer you just opened"


@register_env("FreePlaceIntoTopDrawerEnv-v2", max_episode_steps=200)
class FreePlaceIntoTopDrawerEnvV2(PlaceIntoClosedTopDrawerCustomInSceneEnv):
    def get_language_instruction(self, **kwargs):
        if self.cur_subtask_id == 0:
            return f"Slide open the {self.drawer_id} drawer"
        else:
            model_name = self._get_instruction_obj_name(self.model_id)
            return f"put the {model_name} into the open drawer"

