import numpy as np
from vizdoom import *


class Environment():
    def __init__(self):
        # setting up doom as specified here: https://github.com/awjuliani/DeepRL-Agents/blob/master/a3c_doom-Doom.ipynb
        game = DoomGame()
        game.set_doom_scenario_path(
            "environments/doom/basic.wad")  # This corresponds to the simple task we will pose our agent
        game.set_doom_map("map01")
        game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_screen_format(ScreenFormat.GRAY8)
        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.add_available_button(Button.MOVE_LEFT)
        game.add_available_button(Button.MOVE_RIGHT)
        game.add_available_button(Button.ATTACK)
        game.add_available_game_variable(GameVariable.AMMO2)
        game.add_available_game_variable(GameVariable.POSITION_X)
        game.add_available_game_variable(GameVariable.POSITION_Y)
        game.set_episode_timeout(300)
        game.set_episode_start_time(10)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_living_reward(-1)
        game.set_mode(Mode.PLAYER)
        game.init()
        # End Doom set-up
        self.env = game

        self.actions = np.eye(3, dtype=bool).tolist()
        self.num_actions = len(self.actions)

    def reset(self):
        self.env.new_episode()

    def render(self):
        return self.env.get_state().screen_buffer

    def step(self, action):
        # internally, the action is not just a number, vizdoom expects a list
        action = self.actions[action]
        reward = self.env.make_action(action)
        done = self.env.is_episode_finished()

        if not done:
            observation = self.render()
        else:
            # this is the shape of doom screen capture
            # TODO: remove this magic number
            observation = np.zeros((120, 160))
        return observation, reward, done

    def sample_action(self):
        # random.choice could work directly on the list
        # but I want np.seed(0) to make this globally deterministic
        # well, as far as possible
        return np.random.choice(self.num_actions)
