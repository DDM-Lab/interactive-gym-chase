from __future__ import annotations

import eventlet

eventlet.monkey_patch()

import argparse
import copy

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
    base_scene = copy.deepcopy(oc_scenes.cramped_room_sp_0)
    
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
    """
    Create a complete set of cramped room scenes (20 episodes) with the specified frame skip value.
    Returns a SceneWrapper containing all episodes for that frame skip.
    """
    # Create all 20 episodes
    episodes = [create_cramped_room_episode(episode_num) for episode_num in range(20)]
    
    # Update frame_skip for each episode's policies
    for episode_scene in episodes:
        # The scene object has a policies method that we need to override the frame_skip
        episode_scene.policies(frame_skip=frame_skip_value)
    
    # Create a SceneWrapper with a descriptive scene_id
    return scene.SceneWrapper(
        scenes=episodes
    )

# Create scene wrapper with fixed frame skip value of 5
cramped_room_scenes = create_cramped_room_scenes_with_frame_skip(5)

stager = stager.Stager(
    scenes=[
        oc_scenes.start_scene,
        oc_scenes.tutorial_gym_scene,
        cramped_room_scenes,
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
        "--debug", type=bool, default=False, help="Enable debug mode to print episode information"
    )
    args = parser.parse_args()

    # Store debug flag in the experiment config so it can be accessed by the scene
    import interactive_gym.server.app as app_module
    app_module.DEBUG_MODE = args.debug

    experiment_config = (
        experiment_config.ExperimentConfig()
        .experiment(stager=stager, experiment_id="overcooked_test")
        .hosting(port=args.port, host="0.0.0.0")
    )

    app.run(experiment_config)
