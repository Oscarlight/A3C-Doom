from vizdoom import *

class Game:
	def __init__(self):
		self.game = DoomGame()

	def basic(self):
		game = self.game
		game.set_doom_scenario_path("basic.wad") #This corresponds to the simple task we will pose our agent
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
		return game