## TO RUN THIS FILE, RUN THIS COMMAND: 
# python -m interactive_gym.examples.cogrid.overcooked_human_ai_sp_crampedroom  --port 5703      
from __future__ import annotations

import eventlet

eventlet.monkey_patch()

import argparse
import copy
import json
import math
from pathlib import Path

from interactive_gym.server import app
from interactive_gym.scenes import scene
from interactive_gym.scenes import static_scene
from interactive_gym.scenes import stager
from interactive_gym.examples.cogrid.pyodide_overcooked import (
    scenes as oc_scenes,
)
from interactive_gym.examples.cogrid import (
    overcooked_utils,
)

from interactive_gym.configurations import experiment_config

end_survey_scene = (
    static_scene.ScalesAndTextBox(
        pre_scale_header="Please answer the following questions about your experience:",
        scale_questions=[
            "On a scale from 1-7, with 1 being detrimental and 7 being beneficial to your success, how effective was your partner as a teammate?",
            "On a scale from 1-7, with 1 being not at all and 7 being very much, rate how much you enjoyed playing the game with your partner.",
            "On a scale of 1-7, rate how much you think that your partner contributed to the success of your team. With 1 meaning they made your team worse off and 7 being that they made a very positive contribution.",
            "On a scale of 1-7, rate how much you think that you contributed to the success of your team. With 1 meaning you made your team worse off and 7 being you made a very positive contribution.",
            "On a scale from 1 to 7, where 1 is definitely a bot, 4 is unsure, and 7 is definitely a human, indicate how likely you think that your partner is a human or a bot build to play this game?",
        ],
        scale_labels=["1", "2", "3", "4", "5", "6", "7"],
        scale_size=7,
        text_box_header="Please provide any additional feedback you would like to share.",
    )
    .scene(scene_id="cramped_room_options_scene_0", experiment_config={})
    .display(scene_subheader="Partner Feedback")
)

def create_cramped_room_episode(episode_num: int) -> scene.Scene:
    """Create a cramped room scene with the given episode number."""
    base_scene = (
        copy.deepcopy(oc_scenes.cramped_room_frame_skip)
        .scene(
            scene_id=f"cramped_room_frame_skip_round_{episode_num + 1:02d}",
            should_export_metadata=True,
        )
        .gameplay(num_episodes=1)
    )
    
    if episode_num == 0:
        # First episode
        scene_body = "<center><p>" + "You'll now play with a partner for 20 rounds. " + "<br><br> " + "You will be playing on the layout pictured below. " + '<center><img src="static/assets/overcooked/cramped_room.png" alt="Annotated Overcooked environment." height="270" width="315"></center>' + "When the button activates, click it to begin. " + "</p></center>"
    else:
        # Subsequent episodes
        scene_body = "<center><p>" + "You'll now play another round on the same layout. " + "</p></center>"
    
    scene_obj = base_scene.user_experience(
        scene_header="Overcooked",
        scene_body=scene_body,
        game_page_html_fn=overcooked_utils.overcooked_game_page_header_fn,
        in_game_scene_body="""
        <center>
        <p>
        Use the arrow keys <img src="static/assets/keys/arrow_keys_2.png" alt="Keyboard arrow keys" height="24" width="20" style="vertical-align:middle;"> 
        to control your chef <img src="static/assets/overcooked/blue_chef.png" alt="Blue Chef" height="24" width="24" style="vertical-align:middle;"> 
        and press <img src="static/assets/keys/icons8-w-key-50.png" alt="W key" height="24" width="24" style="vertical-align:middle;"> to pick up and 
        drop objects. Try to deliver as many dishes as possible by combining onions in the pot, plating the cooked onions, 
        and delivering them to the grey delivery zone.
        </p>
        </center>
        <br><br>
        """,
    )
    
    # Store the episode number in the scene object
    scene_obj.episode_num = episode_num
    
    return scene_obj

def create_cramped_room_scenes_with_frame_skip(frame_skip_value: int):
    """Create the tutorial and 20 episodes with one frame skip value."""
    tutorial = copy.deepcopy(oc_scenes.frame_skip_tutorial_gym_scene)
    tutorial.policies(frame_skip=frame_skip_value)

    episodes = [create_cramped_room_episode(episode_num) for episode_num in range(20)]
    for episode_scene in episodes:
        episode_scene.policies(frame_skip=frame_skip_value)

    return scene.SceneWrapper(scenes=[tutorial, *episodes])


speed_data_path = (
    Path(__file__).resolve().parents[2] / "scenes" / "subject_mean_speeds.json"
)
with speed_data_path.open(encoding="utf-8") as speed_data_file:
    subject_mean_speeds = json.load(speed_data_file)

# Each JSON entry gets one slot, so repeated frame skips retain their frequency.
frame_skip_values = [math.ceil(speeds[1]) for speeds in subject_mean_speeds.values()]
frameskip_scene_wrappers = {
    frame_skip: create_cramped_room_scenes_with_frame_skip(frame_skip)
    for frame_skip in sorted(set(frame_skip_values))
}

stager = stager.Stager(
    scenes=[
        oc_scenes.start_scene,
        scene.RandomizeOrder(
            scenes=[
                frameskip_scene_wrappers[frame_skip] for frame_skip in frame_skip_values
            ],
            keep_n=1,
        ),
        end_survey_scene,
        oc_scenes.end_scene,
    ]
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", type=int, default=5703, help="Port number to listen on"
    )
    parser.add_argument(
        "--data-dir", default="data", help="Directory for participant data"
    )
    parser.add_argument(
        "--debug", type=bool, default=False, help="Enable debug mode to print episode information"
    )
    args = parser.parse_args()

    # Store debug flag in the experiment config so it can be accessed by the scene
    import interactive_gym.server.app as app_module
    app_module.DEBUG_MODE = args.debug

    experiment_config = (
        experiment_config.ExperimentConfig()
        .experiment(
            stager=stager,
            experiment_id="overcooked_test",
            data_dir=args.data_dir,
            scene_data_directories={
                "overcooked_frame_skip_tutorial": "overcooked_tutorial",
                **{
                    f"cramped_room_frame_skip_round_{episode_num + 1:02d}": str(
                        Path("cramped_room_frame_skip")
                        / f"cramped_room_frame_skip_round_{episode_num + 1:02d}"
                    )
                    for episode_num in range(20)
                },
            },
        )
        .hosting(port=args.port, host="0.0.0.0")
    )

    app.run(experiment_config)
