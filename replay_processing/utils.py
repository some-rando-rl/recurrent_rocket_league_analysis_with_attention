import os
from os.path import join as p_join

import logging
from collections import Counter
from typing import Union

import carball as cb
import numpy as np
from carball.analysis.analysis_manager import AnalysisManager
from carball.controls.controls import ControlsCreator

from rlgym.utils import math
from rlgym.utils.common_values import ORANGE_TEAM, BLUE_TEAM, BOOST_LOCATIONS
from rlgym.utils.gamestates import GameState, PhysicsObject, PlayerData
import torch
import pickle

boost_locations = np.array(BOOST_LOCATIONS)  # Need ndarray for speed

_invert = np.array((-1, -1, 1))
_rot_correct = np.array((-1, 1, -1))


def convert_replay(replay: Union[str, AnalysisManager], include_frame=False):
    if isinstance(replay, str):
        replay = cb.analyze_replay_file(replay, logging_level=logging.CRITICAL)
    ControlsCreator().get_controls(replay.game)

    boost_timers = np.zeros(34)
    demo_timers = np.zeros(len(replay.game.players))

    blue_goals = 0
    orange_goals = 0
    goals = list(replay.game.goals)[::-1]
    touches = list(replay.protobuf_game.game_stats.hits)[::-1]
    demos = list(replay.game.demos)[::-1]

    match_goals = Counter()
    match_saves = Counter()
    match_shots = Counter()
    match_demos = Counter()
    match_boost_pickups = Counter()

    boost_amounts = {}
    last_locations = {}

    player_pos_pyr_vel_angvel_boost_controls = {  # Preload useful arrays so we can fetch by index later
        player.online_id: (
            player.data[["pos_x", "pos_y", "pos_z"]].values.astype(float),
            player.data[["rot_x", "rot_y", "rot_z"]].fillna(0).values.astype(float) * _rot_correct,
            player.data[["vel_x", "vel_y", "vel_z"]].fillna(0).values.astype(float) / 10,
            player.data[["ang_vel_x", "ang_vel_y", "ang_vel_z"]].fillna(0).values.astype(float) / 1000,
            player.data["boost"].fillna(0).astype(float) / 255,
            player.controls[["throttle", "steer", "pitch", "yaw", "roll",
                             "jump", "boost", "handbrake"]].fillna(0).values.astype(float),
        )
        for player in replay.game.players
    }

    ball_pos_pyr_vel_angvel = (
        replay.game.ball[["pos_x", "pos_y", "pos_z"]].values.astype(float),
        replay.game.ball[["rot_x", "rot_y", "rot_z"]].fillna(0).values.astype(float) * _rot_correct,
        replay.game.ball[["vel_x", "vel_y", "vel_z"]].fillna(0).values.astype(float) / 10,
        replay.game.ball[["ang_vel_x", "ang_vel_y", "ang_vel_z"]].fillna(0).values.astype(float) / 1000,
    )
    rallies = []
    for kf1, kf2 in zip(replay.game.kickoff_frames, replay.game.kickoff_frames[1:] + [replay.game.frames.index[-1]]):
        for goal in replay.game.goals:
            if kf1 < goal.frame_number < kf2:
                rallies.append((kf1, goal.frame_number))
                break
        else:  # No goal between kickoffs
            rallies.append((kf1, kf2))

    last_frame = 0
    for i, (frame, ball_row) in enumerate(replay.game.ball.iterrows()):
        for start, end in rallies:
            if start <= frame < end:
                # del rallies[0]
                break
        else:
            continue

        state = GameState()

        # game_type
        state.game_type = -1

        # blue_score/orange_score
        if len(goals) > 0 and goals[-1].frame_number <= frame:
            goal = goals.pop()
            match_goals[goal.player.online_id] += 1
            if goal.player_team == 0:
                blue_goals += 1
            else:
                orange_goals += 1
        state.blue_score = blue_goals
        state.orange_score = orange_goals

        # last_touch
        touched = set()
        while len(touches) > 0 and touches[-1].frame_number <= frame:
            touch = touches.pop()
            p_id = touch.player_id.id
            state.last_touch = p_id
            touched.add(p_id)
            if touch.save:
                match_saves[p_id] += 1
            if touch.shot:
                match_shots[p_id] += 1

        # demos for players
        demoed = set()
        while len(demos) > 0 and demos[-1]["frame_number"] <= frame:
            demo = demos.pop()
            attacker = demo["attacker"].online_id
            victim = demo["victim"].online_id
            match_demos[attacker] += 1
            demoed.add(victim)

        # players
        actions = []
        for n, player in enumerate(replay.game.players):
            player_data = PlayerData()
            if player.online_id in demoed:
                demo_timers[n] = 3

            player_data.car_id = player.online_id
            player_data.team_num = ORANGE_TEAM if player.team.is_orange else BLUE_TEAM
            player_data.match_goals = match_goals[player.online_id]
            player_data.match_saves = match_saves[player.online_id]
            player_data.match_shots = match_shots[player.online_id]
            player_data.match_demolishes = match_demos[player.online_id]
            player_data.boost_pickups = match_boost_pickups[player.online_id]
            player_data.is_demoed = demo_timers[n] > 0
            player_data.on_ground = None  # Undefined
            player_data.ball_touched = player.online_id in touched
            player_data.has_flip = None  # Undefined, TODO use jump_active, double_jump_active and dodge_active?
            pos, pyr, vel, ang_vel, boost, controls = (v[i] for v in
                                                       player_pos_pyr_vel_angvel_boost_controls[player.online_id])

            player_data.boost_amount = boost
            if np.isnan(pos).any():
                pos = last_locations[player.online_id]
            else:
                last_locations[player.online_id] = pos
            player_data.car_data = PhysicsObject(
                position=pos,
                quaternion=math.rotation_to_quaternion(math.euler_to_rotation(pyr)),
                linear_velocity=vel,
                angular_velocity=ang_vel
            )
            player_data.inverted_car_data = PhysicsObject(
                position=pos * _invert,
                quaternion=math.rotation_to_quaternion((math.euler_to_rotation(pyr).T * _invert).T),
                linear_velocity=vel * _invert,
                angular_velocity=ang_vel * _invert
            )

            old_boost = boost_amounts.get(player.online_id, float("inf"))
            boost_change = boost - old_boost
            boost_amounts[player.online_id] = player_data.boost_amount
            if boost_change > 0 and not (old_boost == 0 and boost == 85 / 255):  # Ignore boost gains on spawn
                closest_boost = np.linalg.norm(boost_locations - pos, axis=-1).argmin()
                if boost_locations[closest_boost][1] > 72:
                    boost_timers[closest_boost] = 10
                else:
                    boost_timers[closest_boost] = 4
                match_boost_pickups[player.online_id] += 1

            state.players.append(player_data)

            actions.append(controls)

        # ball
        pos, pyr, vel, ang_vel = (v[i] for v in ball_pos_pyr_vel_angvel)
        if np.isnan(pos).any():
            continue  # Goal scored, go next
        state.ball = PhysicsObject(
            position=pos,
            quaternion=math.rotation_to_quaternion(math.euler_to_rotation(pyr)),
            linear_velocity=vel,
            angular_velocity=ang_vel
        )

        # inverted_ball
        state.inverted_ball = PhysicsObject(
            position=pos * _invert,
            quaternion=math.rotation_to_quaternion((math.euler_to_rotation(pyr).T * _invert).T),
            linear_velocity=vel * _invert,
            angular_velocity=ang_vel * _invert
        )

        # boost_pads
        state.boost_pads = (boost_timers == 0) * 1

        # inverted_boost_pads
        state.inverted_boost_pads = state.boost_pads[::-1]

        d_time = (frame - last_frame) / 30  # Maybe use time delta from replay instead?
        boost_timers -= d_time  # Should this be before or after values are set?
        demo_timers -= d_time
        boost_timers[boost_timers < 0] = 0
        demo_timers[demo_timers < 0] = 0
        last_frame = frame

        state.players.sort(key=lambda p: (p.team_num, p.car_id))

        yield (state, actions, frame) if include_frame else (state, actions)


def binary_search(arr, x):
    low = 0
    high = len(arr) - 1

    while low <= high:

        mid = (high + low) // 2

        # If x is greater, ignore left half
        if arr[mid] < x:
            low = mid + 1

        # If x is smaller, ignore right half
        elif arr[mid] > x:
            high = mid - 1

        # means x is present at mid
        else:
            return mid

    # If we reach here, then the element was not present
    return low


def pair_jsons_with_replays():
    for game_mode in os.listdir("replays"):
        pairs = {}
        for file in os.listdir(p_join("replays", game_mode)):
            name, extension = file.split(".")
            if name not in pairs:
                pairs[name] = {}
            pairs[name][extension] = file
        os.mkdir(f"replays/new_location/{game_mode}")
        for pair in pairs:
            new_pair_path = p_join("replays/new_location", game_mode, pair)
            old_pair_path = p_join("replays", game_mode)
            os.mkdir(new_pair_path)
            os.replace(p_join(old_pair_path, pairs[pair]["json"]),
                       p_join(new_pair_path, pairs[pair]["json"]))
            os.replace(p_join(old_pair_path, pairs[pair]["replay"]),
                       p_join(new_pair_path, pairs[pair]["replay"]))


def get_times_to_goals(replay):
    goals = [goal.frame_number for goal in replay.game.goals]

    converted_replay = convert_replay(replay, include_frame=True)
    times = []
    for gs, actions, frame in converted_replay:
        if frame > goals[-1]:
            continue
        times.append(goals[binary_search(goals, frame)] - frame)
    return times


def convert_replays_to_inputs(replays_directory: str):
    for replay_dir in os.listdir(replays_directory):
        if replay_dir in os.listdir("bins"):  # saves some time
            continue
        replay_string = p_join(replays_directory, replay_dir, replay_dir + ".replay")
        try:
            replay = cb.analyze_replay_file(replay_string, logging_level=logging.CRITICAL)
        except:
            print(f"Failed to analyze replay: {replay_dir}")
            continue
        goals_teams = [goal.player_team for goal in replay.game.goals]
        goals_frames = [goal.frame_number for goal in replay.game.goals]
        print(goals_teams, goals_frames)
        if len(goals_frames) == 0:
            continue
        converted_replay = convert_replay(replay, include_frame=True)
        bins = [{"labels": list(), "inputs": list()} for _ in range(len(goals_teams))]
        try:
            for gs, actions, frame in converted_replay:
                if frame > goals_frames[-1]:
                    break  # we are after the last goal, frames should be in chronological order, no need to continue here
                goal_index = binary_search(goals_frames, frame)
                bins[goal_index]["labels"].append(goals_teams[goal_index])
                bins[goal_index]["inputs"].append(game_state_to_input(gs))
            for bin_i, bin in enumerate(bins):
                assert len(bin["labels"]) == len(bin[
                                                     "inputs"]), f"Inputs and labels in bin are not of the same size inputs length: {len(bin['inputs'])}, labels length: {len(bin['labels'])}."

                if not os.path.exists(f"bins/{replay_dir}"):
                    os.makedirs(f"bins/{replay_dir}")

                with open(f"bins/{replay_dir}/{str(bin_i)}", "wb") as f:
                    pickle.dump(bin, f)
            print(f"replay: {replay_dir} done")
        except KeyError:
            print(f"Key Error in replay: {replay_dir}")
            continue
        except ValueError:
            print(f"Value error in replay: {replay_dir}")
            continue




def game_state_to_input(game_state: GameState):
    b_pos = game_state.ball.position
    b_vel = game_state.ball.linear_velocity
    b_ang = game_state.ball.angular_velocity
    ball = np.concatenate([b_pos, b_vel, b_ang])
    players = []
    empty_player = np.zeros(15 * (3 - (len(game_state.players) // 2)))
    for player in game_state.players[:len(game_state.players) // 2]:
        player_info = np.asarray([player.is_demoed,
                                  player.ball_touched,
                                  player.boost_amount])
        p_pos = player.car_data.position
        p_vel = player.car_data.linear_velocity
        p_ang = player.car_data.angular_velocity
        p_forward = player.car_data.forward()
        players.append(np.concatenate([player_info, p_pos, p_vel, p_ang, p_forward]))
    players.append(empty_player)
    for player in game_state.players[:len(game_state.players) // 2]:
        player_info = np.asarray([player.is_demoed,
                                  player.ball_touched,
                                  player.boost_amount])
        p_pos = player.car_data.position
        p_vel = player.car_data.linear_velocity
        p_ang = player.car_data.angular_velocity
        p_forward = player.car_data.forward()
        players.append(np.concatenate([player_info, p_pos, p_vel, p_ang, p_forward]))
    players.append(empty_player)
    boosts = game_state.boost_pads
    result = np.concatenate([ball, np.concatenate(players), boosts])
    return torch.tensor(result, dtype=torch.float32)


def read():
    with open("bins/012938ce-d907-4710-a6e0-e613686b6727/0", "rb") as f:
        a = pickle.load(f)
    print(len(a["inputs"][0]), type(a["labels"][0]))

def get_all_bins(bins_path):
    all_bins = []
    for replay in os.listdir(bins_path):
        for bin in os.listdir(p_join(bins_path, replay)):
            all_bins.append(p_join(bins_path, replay, bin))
    arr = np.asarray(all_bins)
    np.random.shuffle(arr)
    arr = np.array_split(arr,900)
    return arr

def get_batch(files):
    inputs = []
    labels = []
    for file in files:
        with open(file, "rb") as f:
            a = pickle.load(f)
        try:
            inputs.append(torch.stack(a["inputs"]))
            labels.append(torch.tensor(a["labels"]))
        except:
            continue
    return torch.cat(inputs), torch.cat(labels)

if __name__ == '__main__':
    print(len(get_batch(get_all_bins("bins")[0])[1]))
    #convert_replays_to_inputs(p_join("replays/new_location/RankedDuels"))
