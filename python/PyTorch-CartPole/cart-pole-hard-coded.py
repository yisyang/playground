import gym
import time

SCORES = []
RENDER_ON_SOLVED = True
MAX_STEPS = 1000
SOLVED_SCORE = 996

CYCLES = 1000


# noinspection PyPep8Naming,DuplicatedCode
class HardCodedSolver:
    @staticmethod
    def predict(observation):
        # Observations:
        # 0: Cart position (-4.8 ~ 4.8)
        # 1: Cart velocity
        # 2: Pole angle (-24 ~ 24deg, approx -0.209 ~ 0.209)
        # 3: Pole tip velocity

        # Actions:
        # 0: Push cart to the left
        # 1: Push cart to the right

        # Life and death
        # Pole tip velocity is too high.
        if observation[3] < -0.2:
            return 0
        elif observation[3] > 0.2:
            return 1

        # Pole is leaning too much, and not enough velocity going the other way.
        if observation[2] < -0.01 and observation[3]/observation[2] < -0.8:
            return 0
        elif observation[2] > 0.01 and observation[3]/observation[2] < -0.8:
            return 1

        # Pole tip velocity is still a bit high.
        if observation[3] < -0.14:
            return 0
        elif observation[3] > 0.14:
            return 1

        # Long-term Stability
        # If cart is too far to one side and in the same direction as the pole angle,
        # we will try to move it the other way by moving it more towards that side first.
        if abs(observation[2]) < 0.005 or observation[0] / observation[2] > 0:
            if observation[0] < -0.15:
                return 0
            elif observation[0] > 0.15:
                return 1

        # Try to make things even more boring by further stabling the tip velocity.
        if observation[3] < 0:
            return 0
        else:
            return 1


def cart_pole():
    env = gym.make('gym_cartpole_foo:cartpole-foo-v0')
    env._max_episode_steps = MAX_STEPS

    solver = HardCodedSolver()

    fps = env.metadata.get('video.frames_per_second') or 60

    run = 0
    rendering = False
    dimmed_score = 0
    while run < CYCLES:
        # Reset environment and record the starting observation.
        observation = env.reset()
        step = 0

        while step < MAX_STEPS:
            if not rendering and RENDER_ON_SOLVED and dimmed_score > SOLVED_SCORE:
                rendering = True

            if rendering:
                env.render()

            # Pick action.
            action = solver.predict(observation)

            # Step through environment using chosen action.
            observation, reward, done, info = env.step(action)

            step_next = step + 1
            if done or step_next == MAX_STEPS:
                dimmed_score = 0.8 * dimmed_score + 0.2 * step_next
                print(f"Run: {run}, score: {step_next}, dimmed_score: {dimmed_score}")
                SCORES.append(step)
                break

            step += 1

            if rendering:
                time.sleep(1.0/fps)

        run += 1


if __name__ == "__main__":
    cart_pole()
