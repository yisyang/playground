"""
Plane vs. bullets env by Scott Yang.
Based originally on gym cart-pole system implemented by Rich Sutton et al.
"""

import math
import gym
from gym import spaces, logger
from gym.envs.classic_control import rendering as rend
# from gym.utils import seeding
import numpy as np
# import pyglet
# from pyglet import gl
import random

STATE_W = 100
STATE_H = 100
WINDOW_MARGIN = 10          # Invisible from display, bullets in margin are kept alive.
WINDOW_DISPLAY_SCALE = 4    # Zoom, must be integer. Actual displayed window width is (STATE_W - MARGIN) * DISPLAY_SCALE
FPS = 50                    # Frames per second


class BulletsEnv(gym.Env):
    """
    Description:
        An enemy boss ship is shooting bullets of constant velocity at the plane.
        The player must dodge all bullets and at the same time destroy the boss ship.
        The player ship only shoots forward.
        Enemy boss ship can have powerful weapons firing in all directions.

    Observation:
        Type:   Box(STATE_W, STATE_H, 6)  dtype=int8
        Relevance to training:                                  Dodge   Aim
        Box[0]: X                                               x       x
        Box[1]: Y                                               x       x
        Box[2]: Integer, as described below:
        Index   Representation              Min         Max
        0       Player Hit Box                0           1     x
        1       Enemy Hit Box                 0           1             x
        2       Player Bullet HP              0           1             x
        3       Player Bullet Damage          0           8             x
        4       Enemy Bullet HP               0          40     x
        5       Enemy Bullet Damage           0          40     x

    Actions:
        Type:   MultiDiscrete([9, 2, 2])
        Index   Representation                    Details
        0       XY-Direction Acceleration         NOOP[0], U[1], UL[2], L[3], DL[4], D[5], DR[6], R[7], UR[8]
        1       Charge Weapon                     NOOP[0], CHARGE_WEAPON[1]
        2       Charge Shield                     NOOP[0], CHARGE_SHIELD[1]

        Note:
            Normal weapon deals 1 damage and fires every N frames.
            Shield can be charged over 6+ normal weapon firings, to absorb all damage lasting for
                [0.25, 0.5, 1, 2, 3, 4] weapon firings.
            Weapon can be supercharged over 3+ normal firings, to deal
                [2, 4, 8] damage in a single firing.
                [2, 22, 2222]

    Reward: (When acting as player)
        Points  Awarded For
        +1.0    For each survived step.             (If player max HP = 1)
        -1.0    For each damage taken by player.    (If player max HP > 1)
        +1.0    For each HP damage on boss.

    Starting State:
        Player at the bottom and center of the screen. Enemy at the top center.

    Episode Termination:
        Player HP <= 0.
        Enemy boss HP <= 0.
        Solved Requirements:
        Considered solved when the average reward is greater than or equal to
        495.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second': FPS
    }

    def __init__(self):
        self.action_space = spaces.MultiDiscrete([9, 2, 2])

        self.observation_space = spaces.Box(low=0, high=40, shape=(STATE_W, STATE_H, 6), dtype=np.int8)

        # self.seed()
        self.state = None
        self.viewer = None
        self.score_label = None
        self.player_ship_transform = None
        self.boss_ship_transform = None

        self.steps_taken = 0
        self.steps_beyond_done = None
        self.reward_twenty = 0

        self.player_ship = PlayerShip(int((STATE_W - 1)/2), 9)
        self.boss_ship = BossShipSkullyTrident(int((STATE_W - 1)/2), STATE_H - 11, -1)
        self.bullet_engine = BulletEngine(1, -1)

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def step(self, action):
        """
        :param action:
        :type action: [int]
        :return:
        """
        boss_ship_damage = 0
        if action is not None:
            # Firstly, let's obtain the player input and use it to move the player ship.
            self.player_ship.steer(action[0])

            # Move boss ship randomly for now.
            if random.randint(0, 4) == 4:
                movements = [0, 3, 7]
                self.boss_ship.steer(movements[random.randint(0, 2)])
            else:
                self.boss_ship.steer(self.boss_ship.prev_accel)

            # After ship movements are performed, we will move all existing bullets and remove dead ones.
            self.bullet_engine.move_bullets()

            # After all ship and bullet movements, we will charge/fire weapons and shields.
            self.player_ship.charge_and_shoot(action[1], action[2], self.bullet_engine)
            self.boss_ship.charge_and_shoot(self.bullet_engine)

            # Finally, now that all the ships and bullets are in position, compute collision.
            player_ship_damage = self.bullet_engine.compute_player_ship_collision(self.player_ship)
            boss_ship_damage = self.bullet_engine.compute_boss_ship_collision(self.boss_ship)

            # Deduct hp from ships.
            if player_ship_damage > 0:
                if self.player_ship.shield_duration == 0:
                    self.player_ship.hp = max(0, self.player_ship.hp - player_ship_damage)

            if boss_ship_damage > 0:
                self.boss_ship.hp = max(0, self.boss_ship.hp - boss_ship_damage)

            # Also collide and eliminate targetable bullets.
            self.bullet_engine.collide_targetable_bullets()

        lose = self.player_ship.hp == 0
        win = self.boss_ship.hp == 0
        done = win or lose

        if not done:
            # reward = boss_ship_damage - player_ship_damage
            reward = 1 + 5 * boss_ship_damage
        elif self.steps_beyond_done is None:
            # Just done
            self.steps_beyond_done = 0
            if win:
                reward = 10
            else:
                reward = -10
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        self.steps_taken += 1
        self.reward_twenty = self.reward_twenty / 20 + reward

        logger.info(f'Steps taken: {self.steps_taken} Reward moving: {self.reward_twenty}')

        self.state = self.render("state_pixels")

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.player_ship.reset()
        self.boss_ship.reset()
        self.bullet_engine.reset()
        self.state = self.render("state_pixels")
        self.steps_taken = 0
        self.steps_beyond_done = None
        self.reward_twenty = 0
        return np.array(self.state)

    def render(self, mode='human'):
        assert mode in ['human', 'rgb_array', 'state_pixels']
        if self.viewer is None:
            self.viewer = rend.Viewer(STATE_W * WINDOW_DISPLAY_SCALE, STATE_H * WINDOW_DISPLAY_SCALE)
            self.boss_ship_transform = rend.Transform(
                    translation=(0, 0),
                    scale=(1, 1))
            self.player_ship_transform = rend.Transform(
                    translation=(0, 0),
                    scale=(1, 1))
            # TODO: Add score label
            # self.score_label = pyglet.text.Label('0000', font_size=36,
            #                                      x=20, y=WINDOW_H * 2.5 / 40.00, anchor_x='left', anchor_y='center',
            #                                      color=(255, 255, 255, 255))

            # Add ships to renderer
            boss_ship = self.boss_ship.get_poly_render(scale=WINDOW_DISPLAY_SCALE)
            boss_ship.add_attr(self.boss_ship_transform)
            boss_ship.set_color(0.8, 0.4, 0)
            player_ship = self.player_ship.get_poly_render(scale=WINDOW_DISPLAY_SCALE)
            player_ship.add_attr(self.player_ship_transform)
            player_ship.set_color(0, 0.6, 1.0)
            self.viewer.add_geom(boss_ship)
            self.viewer.add_geom(player_ship)

        # Adjust rendered ship position to ship geoms.
        self.boss_ship_transform.set_translation(self.boss_ship.x * WINDOW_DISPLAY_SCALE,
                                                 self.boss_ship.y * WINDOW_DISPLAY_SCALE)
        self.player_ship_transform.set_translation(self.player_ship.x * WINDOW_DISPLAY_SCALE,
                                                   self.player_ship.y * WINDOW_DISPLAY_SCALE)

        # Render player ship shield if necessary.
        if self.player_ship.shield_duration > 0:
            shield_geom = rend.make_circle(radius=5, filled=False)
            shield_geom.add_attr(
                    rend.Transform(
                        translation=tuple(np.array([self.player_ship.x, self.player_ship.y]) * WINDOW_DISPLAY_SCALE),
                        scale=(WINDOW_DISPLAY_SCALE, WINDOW_DISPLAY_SCALE)))
            if self.player_ship.shield_duration > 0.5 * self.player_ship.weapon_delay:
                shield_geom.set_color(0, 0.5, 1.0)
            else:
                shield_geom.set_color(1.0, 0, 0)
            self.viewer.add_onetime(shield_geom)

        # Render play ship weapon charge status.
        if self.player_ship.weapon_charging == 1:
            weapon_level = min(3, math.floor(self.player_ship.weapon_charged / self.player_ship.weapon_delay))
            charging_geom = rend.make_circle(radius=1, filled=True)
            charging_geom.add_attr(rend.Transform(
                    translation=tuple(
                            np.array([self.player_ship.x,
                                      self.player_ship.y + 2 * self.player_ship.y_direction]) * WINDOW_DISPLAY_SCALE),
                    scale=(WINDOW_DISPLAY_SCALE, WINDOW_DISPLAY_SCALE)))
            charging_geom.set_color(max(0, 1 - weapon_level), weapon_level / 3, weapon_level / 3)
            self.viewer.add_onetime(charging_geom)
        elif self.player_ship.shield_charging == 1:
            shield_level = min(6, math.floor(self.player_ship.shield_charged / self.player_ship.weapon_delay))
            charging_geom = rend.make_circle(radius=1, filled=True)
            charging_geom.add_attr(rend.Transform(
                translation=tuple(np.array([self.player_ship.x, self.player_ship.y]) * WINDOW_DISPLAY_SCALE),
                scale=(WINDOW_DISPLAY_SCALE, WINDOW_DISPLAY_SCALE)))
            charging_geom.set_color(max(0, 1 - shield_level), shield_level / 6, 0)
            self.viewer.add_onetime(charging_geom)

        # Add bullets to renderer.
        for bullet in self.bullet_engine.boss_bullets:
            bullet_geom = rend.make_circle(radius=1, filled=True)
            bullet_geom.add_attr(
                    rend.Transform(translation=tuple(np.array([bullet.x, bullet.y]) * WINDOW_DISPLAY_SCALE),
                                   scale=(WINDOW_DISPLAY_SCALE, WINDOW_DISPLAY_SCALE)))
            bullet_geom.set_color(0.8, 0.2, 0)
            self.viewer.add_onetime(bullet_geom)
        for bullet in self.bullet_engine.player_bullets:
            bullet_geom = rend.make_circle(radius=1, filled=True)
            bullet_geom.add_attr(
                    rend.Transform(translation=tuple(np.array([bullet.x, bullet.y]) * WINDOW_DISPLAY_SCALE),
                                   scale=(WINDOW_DISPLAY_SCALE, WINDOW_DISPLAY_SCALE)))
            bullet_geom.set_color(0, 0.2, 1.0)
            self.viewer.add_onetime(bullet_geom)

        if mode != 'state_pixels':
            return self.viewer.render(return_rgb_array=mode == 'rgb_array')

        # Initialize state pixels
        state_pixels = np.zeros(shape=(STATE_W, STATE_H, 6), dtype=np.int8)

        # Add player ship presence to state
        ship_xy_data = self.player_ship.get_xy_positions()
        for xy in ship_xy_data:
            state_pixels[xy[0]][xy[1]][0] = 1

        # Add boss ship presence to state
        ship_xy_data = self.boss_ship.get_xy_positions()
        for xy in ship_xy_data:
            state_pixels[xy[0]][xy[1]][1] = 1

        # Add player bullet hp to state
        for bullet in self.bullet_engine.player_bullets:
            state_pixels[bullet.x][bullet.y][2] = bullet.hp
            state_pixels[bullet.x][bullet.y][3] = bullet.damage

        # Add boss bullet hp to state
        for bullet in self.bullet_engine.boss_bullets:
            state_pixels[bullet.x][bullet.y][4] = bullet.hp
            state_pixels[bullet.x][bullet.y][5] = bullet.damage

        return state_pixels

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class Bullet:
    """
    This is a 1 px bullet. (*)
    """
    def __init__(self, x, y, damage_ratio=1, speed_ratio=10, flying_pattern=10, targetable=False, hp=1, ttl=1000):
        """
        :param x: Initial x position
        :param y: Initial y position
        :param speed_ratio: (0-20) Change in linear pixels (x or y) per 20 steps
        :param damage_ratio: (1-100) Amount of damage on enemy per hp of bullet
        :param flying_pattern: See FLYING_PATTERN_ constants.
        :param targetable: Whether or not bullet interacts with enemy bullet
        :param hp: HP of bullet, if it is destructible. Otherwise 1.
        :param ttl: Steps to stay alive
        :type x: int
        :type y: int
        :type speed_ratio: int
        :type damage_ratio: int
        :type ttl: int
        """
        self.x_actual = x
        self.y_actual = y
        self.x = round(self.x_actual)
        self.y = round(self.y_actual)
        self.speed_ratio = min(40, speed_ratio)
        self.damage_ratio = damage_ratio
        self.flying_pattern = flying_pattern
        self.targetable = targetable
        self.hp = hp
        self.damage = math.ceil(self.hp * self.damage_ratio)
        self.ttl = ttl
        self.steps = 0


class BulletEngine:
    # Simple straight patterns
    FLYING_PATTERN_STRAIGHT_LEFT = 4        # Note left is relative to the straight direction of traveling.
    FLYING_PATTERN_STRAIGHT_L_75_DEG = 5
    FLYING_PATTERN_STRAIGHT_L_60_DEG = 6
    FLYING_PATTERN_STRAIGHT_L_45_DEG = 7
    FLYING_PATTERN_STRAIGHT_L_30_DEG = 8
    FLYING_PATTERN_STRAIGHT_L_15_DEG = 9
    FLYING_PATTERN_STRAIGHT = 10
    FLYING_PATTERN_STRAIGHT_R_15_DEG = 11
    FLYING_PATTERN_STRAIGHT_R_30_DEG = 12
    FLYING_PATTERN_STRAIGHT_R_45_DEG = 13
    FLYING_PATTERN_STRAIGHT_R_60_DEG = 14
    FLYING_PATTERN_STRAIGHT_R_75_DEG = 15
    FLYING_PATTERN_STRAIGHT_RIGHT = 16

    # Advanced patterns
    FLYING_PATTERN_ACCEL = 60
    FLYING_PATTERN_WAVY = 70
    FLYING_PATTERN_SPREAD_5 = 80
    FLYING_PATTERN_SPREAD_7 = 81
    FLYING_PATTERN_SPREAD_9 = 82
    FLYING_PATTERN_HOMING = 90

    def __init__(self, player_ship_y_direction=1, boss_ship_y_direction=-1):
        """
        :type self.player_bullets: [Bullet]
        :type self.boss_bullets: [Bullet]
        """
        self.player_bullets = []
        self.boss_bullets = []
        self.player_ship_y_direction = player_ship_y_direction
        self.boss_ship_y_direction = boss_ship_y_direction

    @staticmethod
    def compute_bullet_collisions(bullets_list_a, bullets_list_b):
        bullets_list_a_remaining = []
        bullets_list_b_remaining = []

        # Eliminate targetable bullets in list a.
        for la_b in bullets_list_a:
            if not la_b.targetable:
                bullets_list_a_remaining.append(la_b)
                continue

            # Targetable bullet in list a found.
            # Iterate against all bullets in list b.
            for lb_b in bullets_list_b:
                if lb_b.x == la_b.x and lb_b.y == la_b.y:
                    # Collision found against bullet in list b.
                    la_b.hp = la_b.hp - lb_b.damage
                    lb_b.hp = lb_b.hp - la_b.damage

                    if lb_b.hp > 0:
                        lb_b.damage = math.ceil(lb_b.hp * lb_b.damage_ratio)
                        bullets_list_b_remaining.append(lb_b)

                    if la_b.hp <= 0:
                        # Targetable list a bullet completely destroyed, proceed to next list a bullet.
                        break

                    # Compute list a bullet's remaining damage.
                    la_b.damage = math.ceil(la_b.hp * la_b.damage_ratio)
                else:
                    # No collision, put list b bullet back.
                    bullets_list_b_remaining.append(lb_b)

            # List a bullet survived past all collisions.
            if la_b.hp > 0:
                bullets_list_a_remaining.append(la_b)

            # Update list b for iteration against next targetable list a bullet.
            bullets_list_b = bullets_list_b_remaining

        return [bullets_list_a_remaining, bullets_list_b]

    @staticmethod
    def compute_ship_collision(ship, bullets):
        """
        Calculate damage to ship and remaining bullets based on their coords.

        :param ship:
        :param bullets:
        :type ship: Ship
        :type bullets: [Bullet]
        :return:
        """
        ship_damage = 0
        bullets_remaining = []
        ship_xy_data = ship.get_xy_positions()
        for bullet in bullets:
            bullet_hit = False
            for xy in ship_xy_data:
                # Collide given ship against given bullets
                if bullet.x == xy[0] and bullet.y == xy[1]:
                    ship_damage += bullet.damage
                    bullet_hit = True
                    break
            if not bullet_hit:
                bullets_remaining.append(bullet)

        return [ship_damage, bullets_remaining]

    @staticmethod
    def create_bullet(x, y, damage_ratio=1, speed_ratio=10, flying_pattern=FLYING_PATTERN_STRAIGHT, targetable=False,
                      hp=1, ttl=1000):
        return Bullet(x, y, damage_ratio, speed_ratio, flying_pattern, targetable, hp, ttl)

    @staticmethod
    def get_moved_bullets(bullets_list, y_direction=1):
        bullets_list_moved = []
        for bullet in bullets_list:
            # Remove bullets past TTL.
            if bullet.steps >= bullet.ttl:
                continue

            # Calculate new x/y.
            bullet.steps += 1
            if bullet.flying_pattern == BulletEngine.FLYING_PATTERN_STRAIGHT:
                bullet.y_actual += y_direction * bullet.speed_ratio / 20
                bullet.y = round(bullet.y_actual)
            else:
                quit('Not implemented.')

            # Remove bullets out of bounds.
            if bullet.x < 0 or bullet.x >= STATE_W:
                continue
            if bullet.y < 0 or bullet.y >= STATE_H:
                continue

            bullets_list_moved.append(bullet)

        return bullets_list_moved

    def add_player_bullets(self, bullets):
        if len(bullets) > 0:
            self.player_bullets += bullets

    def add_boss_bullets(self, bullets):
        if len(bullets) > 0:
            self.boss_bullets += bullets

    def compute_player_ship_collision(self, player_ship):
        [ship_damage, self.boss_bullets] = self.compute_ship_collision(player_ship, self.boss_bullets)
        return ship_damage

    def compute_boss_ship_collision(self, boss_ship):
        [ship_damage, self.player_bullets] = self.compute_ship_collision(boss_ship, self.player_bullets)
        return ship_damage

    def collide_targetable_bullets(self):
        """
        Calculate bullet cancellations between player and boss bullets.

        :return:
        """

        # First eliminate targetable boss bullets.
        [self.boss_bullets, self.player_bullets] = self.compute_bullet_collisions(self.boss_bullets,
                                                                                  self.player_bullets)
        # Next eliminate targetable player bullets.
        [self.player_bullets, self.boss_bullets] = self.compute_bullet_collisions(self.player_bullets,
                                                                                  self.boss_bullets)

    def move_bullets(self):
        self.player_bullets = self.get_moved_bullets(self.player_bullets, self.player_ship_y_direction)
        self.boss_bullets = self.get_moved_bullets(self.boss_bullets, self.boss_ship_y_direction)

    def reset(self):
        self.player_bullets = []
        self.boss_bullets = []


class Ship:
    def __init__(self, x, y, y_direction=1):
        assert y_direction in [1, -1]
        self.x_init = x
        self.y_init = y
        self.y_direction = y_direction
        self.x_actual = self.x_init
        self.y_actual = self.y_init
        self.x = round(self.x_actual)
        self.y = round(self.y_actual)
        self.x_velocity = 0.0
        self.y_velocity = 0.0
        self.prev_accel = 0
        self.weapon_delay = 30
        self.weapon_cooldown = 0
        self.max_hp = 10000
        self.hp = self.max_hp
        self.ship_width = 5
        self.ship_height = 5
        self.ship_width_clearance = 10
        self.ship_height_clearance = 10

    def reset(self):
        self.x_actual = self.x_init
        self.y_actual = self.y_init
        self.x = round(self.x_actual)
        self.y = round(self.y_actual)
        self.x_velocity = 0.0
        self.y_velocity = 0.0
        self.prev_accel = 0
        self.weapon_cooldown = 0
        self.hp = self.max_hp

    def steer(self, action_accel):
        """control: steer

        Args:
            action_accel: NOOP[0], U[1], UL[2], L[3], DL[4], D[5], DR[6], R[7], UR[8]
        """
        self.prev_accel = action_accel
        x_acceleration = 0
        y_acceleration = 0
        if action_accel == 1:
            y_acceleration = 1
        elif action_accel == 2:
            x_acceleration = -1
            y_acceleration = 1
        elif action_accel == 3:
            x_acceleration = -1
        elif action_accel == 4:
            x_acceleration = -1
            y_acceleration = -1
        elif action_accel == 5:
            y_acceleration = -1
        elif action_accel == 6:
            x_acceleration = 1
            y_acceleration = -1
        elif action_accel == 7:
            x_acceleration = 1
        elif action_accel == 8:
            x_acceleration = 1
            y_acceleration = 1

        self.x_velocity = max(-1.0, min(1.0, self.x_velocity * 0.2 + 0.5 * x_acceleration))
        self.y_velocity = max(-1.0, min(1.0, self.y_velocity * 0.2 + 0.5 * y_acceleration))
        self.x_actual += self.x_velocity
        self.y_actual += self.y_velocity

        self.x = round(self.x_actual)
        self.y = round(self.y_actual)

        # Prevent ship from going off of the screen.
        if self.x < self.ship_width_clearance:
            self.x_actual = self.ship_width_clearance
            self.x_velocity = 0
        if self.x > STATE_W - self.ship_width_clearance - 1:
            self.x_actual = STATE_W - self.ship_width_clearance - 1
            self.x_velocity = 0
        if self.y < self.ship_height_clearance:
            self.y_actual = self.ship_height_clearance
            self.y_velocity = 0
        if self.y > STATE_H - self.ship_height_clearance - 1:
            self.y_actual = STATE_H - self.ship_height_clearance - 1
            self.y_velocity = 0

        # Prevent player or boss ship from getting too close to the other ship.
        if self.y_direction == 1 and self.y > (STATE_H - 1) / 2:
            self.y_actual = int((STATE_H - 1) / 2)
            self.y_velocity = 0
        if self.y_direction == -1 and self.y < (STATE_H - 1) / 2:
            self.y_actual = int((STATE_H - 1) / 2)
            self.y_velocity = 0

        self.x = round(self.x_actual)
        self.y = round(self.y_actual)

    def get_xy_positions(self):
        # Return all [x, y] pairs with ship pixel.
        return [[self.x, self.y]]

    def get_poly_render(self):
        l, b = math.floor(-self.ship_width / 2), math.floor(-self.ship_height / 2)
        r, t = l + self.ship_width, b + self.ship_height
        poly = rend.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        poly.set_color(.8, .6, .4)
        return poly


class PlayerShip(Ship):
    """
    This is the player ship.
             ^
           ^ ^ ^
         # #   # #
       # # # # # # #
         #   #   #
    """
    def __init__(self, x, y, y_direction=1):
        super().__init__(x, y, y_direction)
        self.weapon_charging = 0
        self.shield_charging = 0
        self.weapon_charged = 0
        self.shield_charged = 0
        self.shield_duration = 0
        self.max_hp = 1
        self.hp = 1
        self.ship_width = 7
        self.ship_height = 5
        self.ship_width_clearance = 5
        self.ship_height_clearance = 5

    def reset(self):
        super().reset()
        self.shield_charged = 0
        self.weapon_charged = 0
        self.shield_duration = 0

    def get_poly_render(self, scale=1):
        xy_line = [(-3.5, -1), (0, 2.5), (3.5, -1),
                   (2.5, -2.5), (1.5, -2.5), (1.5, -1.5),
                   (0.5, -1.5), (0.5, -2.5), (-0.5, -2.5), (-0.5, -1.5),
                   (-1.5, -1.5), (-1.5, -2.5), (-2.5, -2.5)]
        xy_top = [(-2, 0.5), (0, 2.5), (2, 0.5)]
        xy_left = [(-3.5, -1), (-0.5, 2), (-0.5, -1)]
        xy_right = [(3.5, -1), (0.5, 2), (0.5, -1)]
        xy_bottom = [(-3.5, -1), (-3, -0.5), (3, -0.5), (3.5, -1), (3, -1.5), (-3, -1.5)]
        xy_foot_1 = [(-3, -1.5), (-2.5, -2.5), (-1.5, -2.5), (-1.5, -1.5)]
        xy_foot_2 = [(0.5, -1.5), (0.5, -2.5), (-0.5, -2.5), (-0.5, -1.5)]
        xy_foot_3 = [(3, -1.5), (2.5, -2.5), (1.5, -2.5), (1.5, -1.5)]
        xy_line = np.array(xy_line, dtype=np.float) * [scale, scale]
        xy_top = np.array(xy_top, dtype=np.float) * [scale, scale]
        xy_left = np.array(xy_left, dtype=np.float) * [scale, scale]
        xy_right = np.array(xy_right, dtype=np.float) * [scale, scale]
        xy_bottom = np.array(xy_bottom, dtype=np.float) * [scale, scale]
        xy_foot_1 = np.array(xy_foot_1, dtype=np.float) * [scale, scale]
        xy_foot_2 = np.array(xy_foot_2, dtype=np.float) * [scale, scale]
        xy_foot_3 = np.array(xy_foot_3, dtype=np.float) * [scale, scale]
        poly = rend.Compound([
            rend.PolyLine(xy_line, close=True),
            rend.FilledPolygon(xy_top),
            rend.FilledPolygon(xy_left),
            rend.FilledPolygon(xy_right),
            rend.FilledPolygon(xy_bottom),
            rend.FilledPolygon(xy_foot_1),
            rend.FilledPolygon(xy_foot_2),
            rend.FilledPolygon(xy_foot_3)
        ])
        return poly

    def get_xy_positions(self):
        if self.shield_duration > 0:
            xy = [[-1 + x, 4] for x in range(3)] + \
                 [[-3 + x, 3] for x in range(7)] + \
                 [[-3 + x, 2] for x in range(7)] + \
                 [[-4 + x, 1] for x in range(9)] + \
                 [[-4 + x, 0] for x in range(9)] + \
                 [[-4 + x, -1] for x in range(9)] + \
                 [[-3 + x, -2] for x in range(7)] + \
                 [[-3 + x, -3] for x in range(7)] + \
                 [[-1 + x, -4] for x in range(3)]
        else:
            xy = [
                [0, 2],
                [-1, 1], [0, 1], [1, 1],
                [-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0],
                [-3, -1], [-2, -1], [-1, -1], [0, -1], [1, -1], [2, -1], [3, -1],
                [-2, -2], [0, -2], [2, -2]
            ]
        xy = np.array(xy, dtype=np.int8)
        # If direction is -1, ship is facing the other way in y direction.
        if self.y_direction != 1:
            xy = xy * [1, self.y_direction]
        # Add current ship [x, y] to all offset values in list.
        xy += np.array([self.x, self.y], dtype=np.int8)
        return xy

    def charge_and_shoot(self, charge_weapon, charge_shield, bullet_engine):
        """
        :param charge_weapon: Whether or not player has chosen to charge weapon.
        :param charge_shield: Whether or not player has chosen to charge shield.
        :param bullet_engine: Env instantiated bullet engine to keep track of all bullets flying around.
        :type bullet_engine: BulletEngine

        Notes:
            Normal weapon deals 1 damage and fires every N frames.
            Shield can be charged over 6+ normal weapon firings, to absorb all damage lasting for
                [0.25, 0.5, 1, 2, 3, 4] weapon firings.
            Weapon can be supercharged over 3+ normal firings, to deal
                [2, 4, 8] damage in a single firing.
                [2, 22, 2222]
        """
        self.weapon_cooldown -= 1
        self.shield_duration = max(0, self.shield_duration - 1)

        self.weapon_charging = 0
        self.shield_charging = 0
        if charge_weapon == 1:
            self.weapon_charging = 1
            self.weapon_charged += 1
            if self.shield_charged >= self.weapon_delay:
                self.activate_shield()
            self.shield_charged = 0
        elif charge_shield == 1:
            self.shield_charging = 1
            self.shield_charged += 1
            if self.weapon_charged >= self.weapon_delay:
                self.activate_mega_weapon(bullet_engine)
            self.weapon_charged = 0
        else:
            if self.shield_charged >= self.weapon_delay:
                self.activate_shield()
            elif self.weapon_charged >= self.weapon_delay:
                self.activate_mega_weapon(bullet_engine)
            else:
                self.activate_normal_weapon(bullet_engine)
            self.clear_charges()

    def activate_shield(self):
        shield_level = min(6, math.floor(self.shield_charged / self.weapon_delay))
        shield_effects = np.array([0, 0.25, 0.5, 1, 2, 3, 4]) * self.weapon_delay
        self.shield_duration = shield_effects[shield_level]

    def activate_mega_weapon(self, bullet_engine):
        """
        :param bullet_engine: Env instantiated bullet engine to keep track of all bullets flying around.
        :type bullet_engine: BulletEngine
        """
        yd = self.y_direction
        weapon_level = math.floor(self.weapon_charged / self.weapon_delay)
        if weapon_level == 0:
            return
        if weapon_level >= 3:
            bullet_engine.add_player_bullets([
                bullet_engine.create_bullet(x=self.x, y=self.y + 2 * yd, damage_ratio=2, speed_ratio=40),
                bullet_engine.create_bullet(x=self.x - 1, y=self.y + yd, damage_ratio=2, speed_ratio=40),
                bullet_engine.create_bullet(x=self.x, y=self.y + yd, damage_ratio=2, speed_ratio=40),
                bullet_engine.create_bullet(x=self.x + 1, y=self.y + yd, damage_ratio=2, speed_ratio=40)
            ])
        elif weapon_level == 2:
            bullet_engine.add_player_bullets([
                bullet_engine.create_bullet(x=self.x, y=self.y + 2 * yd, damage_ratio=2, speed_ratio=40),
                bullet_engine.create_bullet(x=self.x, y=self.y + yd, damage_ratio=2, speed_ratio=40),
            ])
        elif weapon_level == 1:
            bullet_engine.add_player_bullets([
                bullet_engine.create_bullet(x=self.x, y=self.y + 2 * yd, damage_ratio=2, speed_ratio=40)
            ])

        # If mega weapon is fired, also require cooldown for normal weapon.
        self.weapon_cooldown = self.weapon_delay

    def activate_normal_weapon(self, bullet_engine):
        """
        :param bullet_engine: Env instantiated bullet engine to keep track of all bullets flying around.
        :type bullet_engine: BulletEngine
        """
        if self.weapon_cooldown > 0:
            return
        self.weapon_cooldown = self.weapon_delay
        bullet_engine.add_player_bullets([
            bullet_engine.create_bullet(x=self.x, y=self.y + 2 * self.y_direction, damage_ratio=1, speed_ratio=20)
        ])

    def clear_charges(self):
        self.weapon_charged = 0
        self.shield_charged = 0


class BossShipSkullyTrident(Ship):
    """
            # # # # # # #
          # # # # # # # # #
          # #   # # #   # #
          # #   #   #   # #
            # #   #   # #
        # # # # # # # # # # #
        # # #   # # #   # # #
        # # #   # # #   # # #
        # # #   # # #   # # #
          v     # # #     v
                  v
    """
    def __init__(self, x, y, y_direction=1):
        super().__init__(x, y, y_direction)
        self.max_hp = 1000
        self.ship_width = 11
        self.ship_height = 11
        self.ship_width_clearance = 10
        self.weapon_delay = 30

    def reset(self):
        super().reset()

    def get_poly_render(self, scale=1):
        xy = [(0, 5), (-3, 5), (-4, 4), (-4, 2), (-3, 1), (-5, 0), (-5, -3), (-4, -4), (-3, -3), (-2, 0), (-1, -4),
              (0, -5), (1, -4), (2, 0), (3, -3), (4, -4), (5, -3), (5, 0), (3, 1), (4, 2), (4, 4), (3, 5)]
        xy = np.array(xy, dtype=np.float)
        xy = xy * [scale, scale]
        poly = rend.FilledPolygon(xy)
        return poly

    def get_xy_positions(self):
        xy = [[-3 + x, 5] for x in range(7)] + \
             [[-4 + x, 4] for x in range(9)] + \
             [[-4 + x, 3] for x in range(9)] + \
             [[-4 + x, 2] for x in range(9)] + \
             [[-3 + x, 1] for x in range(7)] + \
             [[-5 + x, 0] for x in range(11)] + \
             [[-5, -1], [-4, -1], [-3, -1], [-1, -1], [0, -1], [1, -1], [3, -1], [4, -1], [5, -1]] + \
             [[-5, -2], [-4, -2], [-3, -2], [-1, -2], [0, -2], [1, -2], [3, -2], [4, -2], [5, -2]] + \
             [[-5, -3], [-4, -3], [-3, -3], [-1, -3], [0, -3], [1, -3], [3, -3], [4, -3], [5, -3]] + \
             [[-4, -4], [-1, -4], [0, -4], [1, -4], [4, -4]] + \
             [[0, -5]]
        xy = np.array(xy, dtype=np.int8)
        # If direction is -1, ship is facing the other way in y direction.
        if self.y_direction != 1:
            xy = xy * [1, self.y_direction]
        xy += np.array([self.x, self.y], dtype=np.int8)
        return xy

    def charge_and_shoot(self, bullet_engine):
        self.weapon_cooldown -= 1

        if self.weapon_cooldown > 0:
            return

        self.weapon_cooldown = self.weapon_delay
        yd = self.y_direction
        bullet_engine.add_boss_bullets([
            bullet_engine.create_bullet(x=self.x, y=self.y + yd * 5, damage_ratio=5, speed_ratio=20),
            bullet_engine.create_bullet(x=self.x - 4, y=self.y + yd * 4, damage_ratio=5, speed_ratio=20),
            bullet_engine.create_bullet(x=self.x + 4, y=self.y + yd * 4, damage_ratio=5, speed_ratio=20)
        ])


class BossShipSkullyTridentLarge(Ship):
    """
                    # # # # # # #
                  # # # # # # # # #
                  # #   # # #   # #
                  # #   #   #   # #
                    # #   #   # #
      # # # # # # # # # # # # # # # # # # # # #
      # #       # # #   # # #   # # #       # #
      v         # # #   # # #   # # #         v
                # # #   # # #   # # #
                  v     # # #     v
                          v
    """
    def __init__(self, x, y, y_direction=1):
        super().__init__(x, y, y_direction)
        self.max_hp = 2000
        self.ship_width = 21
        self.ship_height = 11
        self.ship_width_clearance = 10
        self.weapon_delay = 30

    def reset(self):
        super().reset()

    def get_poly_render(self, scale=1):
        xy = [(0, 5), (-3, 5), (-4, 4), (-4, 2), (-3, 1), (-5, 0), (-5, -3), (-4, -4), (-3, -3), (-2, 0), (-1, -4),
              (0, -5), (1, -4), (2, 0), (3, -3), (4, -4), (5, -3), (5, 0), (3, 1), (4, 2), (4, 4), (3, 5)]
        xy = np.array(xy, dtype=np.int8)
        xy = xy * [scale, scale]
        poly = rend.FilledPolygon(xy)
        return poly

    def get_xy_positions(self):
        xy = [[-3 + x, 5] for x in range(7)] + \
             [[-4 + x, 4] for x in range(9)] + \
             [[-4 + x, 3] for x in range(9)] + \
             [[-4 + x, 2] for x in range(9)] + \
             [[-3 + x, 1] for x in range(7)] + \
             [[-5 + x, 0] for x in range(11)] + \
             [[-5, -1], [-4, -1], [-3, -1], [-1, -1], [0, -1], [1, -1], [3, -1], [4, -1], [5, -1]] + \
             [[-5, -2], [-4, -2], [-3, -2], [-1, -2], [0, -2], [1, -2], [3, -2], [4, -2], [5, -2]] + \
             [[-5, -3], [-4, -3], [-3, -3], [-1, -3], [0, -3], [1, -3], [3, -3], [4, -3], [5, -3]] + \
             [[-4, -4], [-1, -4], [0, -4], [1, -4], [4, -4]] + \
             [[0, -5]]
        xy = np.array(xy, dtype=np.int8)
        # If direction is -1, ship is facing the other way in y direction.
        if self.y_direction != 1:
            xy = xy * [1, self.y_direction]
        xy += np.array([self.x, self.y], dtype=np.int8)
        return xy

    def charge_and_shoot(self, bullet_engine):
        self.weapon_cooldown -= 1

        if self.weapon_cooldown > 0:
            return

        self.weapon_cooldown = self.weapon_delay
        yd = self.y_direction
        bullet_engine.add_boss_bullets([
            bullet_engine.create_bullet(x=self.x, y=self.y + yd * 5, damage_ratio=5, speed_ratio=20),
            bullet_engine.create_bullet(x=self.x - 4, y=self.y + yd * 4, damage_ratio=5, speed_ratio=20),
            bullet_engine.create_bullet(x=self.x + 4, y=self.y + yd * 4, damage_ratio=5, speed_ratio=20)
        ])


class BossShipSkullyRain:
    """
            # # # # # # #
          # # # # # # # # #
          # #   # # #   # #
          # #   #   #   # #
            # #   #   # #
            # # # # # # #
            # # # # # # #
          # #   #   #   # #
        #   #   #   #   #   #
        #   #   #   #   #   #
        v   v   v   v   v   v
    """


class BossShipSkullyWave:
    """
            # # # # # # #
          # # # # # # # # #
          # #   # # #   # #
          # #   #   #   # #
            # #   #   # #
            # # # # # # #
        # # # # # # # # # # #
        # #   # # # # #   # #
        # # v           v # #
          # v v v v v v v #
              v v v v v
    """


class BossShipSkullyYn:
    """
            # # # # # # #
          # # # # # # # # #
          # #   # # #   # #
          # #   #   #   # #
            # #   #   # #
            # # # # # # #
        # # # # # # # # # # #
        # # # # # # # # # # #
        # # #   # # #   # # #
        # Y #   # # #   # Y #
                # Y #
    """


class BossShipSkullyBubble:
    """
            # # # # # # #
          # # # # # # # # #
          # #   # # #   # #
          # #   #   #   # #
            # #   #   # #
            # # # # # # #
            # # # # # # #
          # # #       # # #
        # # #           # # #
        # # # # # # # # # # #
        # Y #   * * *   # Y #
    """


class BossShipQuindent:
    """
    And this... is the boss ship.

                     # # # # # # #
                   # # # # # # # # #
           # # # # #   # #   # #   # # # # #
         # # # # # #   # #   # #   # # # # # #
         # #     # # #   # # #   # # #     # #
         # #   # # # # # # # # # # # # #   # #
         # #   # # #     # # #     # # #   # #
         |     # # #     # # #     # # #     |
               # # #     # # #     # # #
                 |       # # #       |
                           |
    """
