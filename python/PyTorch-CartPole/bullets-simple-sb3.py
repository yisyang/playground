import click
import gym
from pynput import keyboard
from pynput.keyboard import Key
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
# from stable_baselines3 import PPO
# from stable_baselines3.ppo import MlpPolicy
# from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env


@click.command()
@click.option('-m', '--mode', default='ai', help='Select execution mode: ai, train, human, check.')
@click.option('-n', '--n-envs', default=1, help='Number of parallel envs to train with.')
@click.option('-ts', '--training-steps', default=50000, help='Number of time steps to train.')
@click.option('-ds', '--delayed-start', default=0, help='Requires additional key press to start.')
def start(mode, n_envs, training_steps, delayed_start):
    ai_play_mode = 'ai'
    training_mode = 'train'
    human_mode = 'human'
    env_check_mode = 'check'

    env_name = 'rj_gym_envs:bullets-simple-v0'

    if mode in [training_mode, ai_play_mode]:
        if n_envs > 1:
            # Parallel envs
            env = make_vec_env(env_name, n_envs=n_envs)
        else:
            # Single env
            env = gym.make(env_name)

        model = A2C(MlpPolicy, env, learning_rate=0.005, verbose=1)
        if mode == ai_play_mode:
            model = A2C(MlpPolicy, env, verbose=1)
            model.load("net/ppo_bullets_simple")

            obs = env.reset()
            steps = 0
            env.render()

            if delayed_start:
                prompt_any_key()

            done = False
            while steps < 10000 and not done:
                steps += 1
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                env.render()
                print(f'Step: {steps}  Player HP: {env.player_ship.hp}  Boss HP: {env.boss_ship.hp}')
        else:
            model.learn(total_timesteps=training_steps)
            model.save("net/ppo_bullets_simple")
            print('Training complete.')

    elif mode == human_mode:
        env = gym.make(env_name)

        action_state = ActionState()
        handle_input(action_state)

        env.reset()
        steps = 0
        env.render()

        if delayed_start:
            prompt_any_key()

        done = False
        while steps < 10000 and not action_state.terminate and not done:
            steps += 1
            obs, rewards, done, info = env.step(action_state.to_array())
            env.render()
            print(f'Step: {steps}  Player HP: {env.player_ship.hp}  Boss HP: {env.boss_ship.hp}')

    elif mode == env_check_mode:
        env = gym.make(env_name)
        check_env(env, warn=True, skip_render_check=False)
    else:
        raise Exception('Undefined mode. Please refer to script source code for available modes.')

    print('Done')


def handle_input(action_state):
    def on_press(key):
        try:
            key_char = key.char
        except AttributeError:
            key_char = ''
        if key_char == 'a' or key == Key.left:
            action_state.x_left = 1
            action_state.x_action = 1
        elif key_char == 'd' or key == Key.right:
            action_state.x_right = 1
            action_state.x_action = 2
        elif key_char == 'w' or key == Key.up:
            action_state.y_up = 1
            action_state.y_action = 1
        elif key_char == 's' or key == Key.down:
            action_state.y_down = 1
            action_state.y_action = 2

    def on_release(key):
        try:
            key_char = key.char
        except AttributeError:
            key_char = ''
        if key_char == 'a' or key == Key.left:
            action_state.x_left = 0
            if action_state.x_right == 1:
                action_state.x_action = 2
            else:
                action_state.x_action = 0
        elif key_char == 'd' or key == Key.right:
            action_state.x_right = 0
            if action_state.x_left == 1:
                action_state.x_action = 1
            else:
                action_state.x_action = 0
        elif key_char == 'w' or key == Key.up:
            action_state.y_up = 0
            if action_state.y_down == 1:
                action_state.y_action = 2
            else:
                action_state.y_action = 0
        elif key_char == 's' or key == Key.down:
            action_state.y_down = 0
            if action_state.y_up == 1:
                action_state.y_action = 1
            else:
                action_state.y_action = 0
        if key == keyboard.Key.esc:
            # End game
            print('End')
            action_state.terminate = True
            return False

    # Collect events until released
    # with keyboard.Listener(
    #         on_press=on_press,
    #         on_release=on_release) as listener:
    #     listener.join()

    # ...or, in a non-blocking fashion
    listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release)
    listener.start()


def prompt_any_key():
    print('Press any key in game window to continue.')

    def on_press(key):
        return

    def on_release(key):
        return False

    # Collect events until released
    with keyboard.Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()


class ActionState:
    """
    Actions:
        Type:   Discrete(9)
        Representation                      Details
        XY-Direction Acceleration           NOOP[0], U[1], UL[2], L[3], DL[4], D[5], DR[6], R[7], UR[8]
    """
    def __init__(self):
        self.x_action = 0
        self.y_action = 0
        self.xy_action = 0
        self.x_left = 0
        self.x_right = 0
        self.y_up = 0
        self.y_down = 0
        self.terminate = False

    def to_array(self):
        # Going left
        if self.x_action == 1:
            # Going up
            if self.y_action == 1:
                self.xy_action = 2
            # Going down
            elif self.y_action == 2:
                self.xy_action = 4
            else:
                self.xy_action = 3
        # Going right
        elif self.x_action == 2:
            # Going up
            if self.y_action == 1:
                self.xy_action = 8
            # Going down
            elif self.y_action == 2:
                self.xy_action = 6
            else:
                self.xy_action = 7
        else:
            # Going up
            if self.y_action == 1:
                self.xy_action = 1
            # Going down
            elif self.y_action == 2:
                self.xy_action = 5
            else:
                self.xy_action = 0
        return self.xy_action


if __name__ == '__main__':
    start()
