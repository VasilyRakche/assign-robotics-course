import sys
ABS_PATH = "/home/vasko/Documents/TUB3/AI_Robotics/robotics-course-prev"
sys.path.append(ABS_PATH+'/build')
import numpy as np
import libry as ry
import matplotlib.pyplot as plt
import time

from pyquaternion import Quaternion
import math

from lib.agent import Agent
from lib.helper import plot

from collections import deque
import lib.logger
import torch

def psi_to_quat(psi):
    return [math.cos(psi/2),0,0,math.sin(psi/2)]

def quat_to_psi(qd):
    return math.atan2( 2 * (qd.w * qd.z + qd.x * qd.y), 1 - 2 * (qd.y**2 + qd.z**2) )
    
def get_z_angle_diff(frame1, frame2):
    q1 = Quaternion(frame1.getQuaternion())
    q = Quaternion(frame2.getQuaternion())

    qd = q.conjugate * q1

    # Calculate Euler angles from this difference quaternion
    # phi   = math.atan2( 2 * (qd.w * qd.x + qd.y * qd.z), 1 - 2 * (qd.x**2 + qd.y**2) )
    # theta = math.asin ( 2 * (qd.w * qd.y - qd.z * qd.x) )
    psi   = quat_to_psi(qd)
    
    return psi

def get_xy_position_diff(frame1,frame2):
    res = frame1.getPosition() - frame2.getPosition()
    return res[0], res[1]

def rad_to_deg(angle):
    return angle/(2*math.pi)*360


def start_direction(frame, code):
    qd = Quaternion(frame.getQuaternion())
    psi   = math.atan2( 2 * (qd.w * qd.z + qd.x * qd.y), 1 - 2 * (qd.y**2 + qd.z**2) )
    
    R = np.array([[math.cos(psi), -math.sin(psi)],
                     [math.sin(psi), math.cos(psi)]])
    vel = .3
    dist = .075+.022+.005 
    offset = .07
    
    if code == 0:
        rel_start = np.array([-offset, -dist])
        direction = np.array([0, vel])
    elif code == 1:
        rel_start = np.array([0, -dist])
        direction = np.array([0, vel])
    elif code == 2:  
        rel_start = np.array([offset, -dist])
        direction = np.array([0, vel])
    elif code == 3:     
        rel_start = np.array([dist, -offset])
        direction = np.array([-vel, 0])
    elif code == 4:     
        rel_start = np.array([dist, 0])
        direction = np.array([-vel, 0])
    elif code == 5:     
        rel_start = np.array([dist, offset])
        direction = np.array([-vel, 0])
    elif code == 6:     
        rel_start = np.array([offset, dist])
        direction = np.array([0, -vel])
    elif code == 7:     
        rel_start = np.array([0, dist])
        direction = np.array([0, -vel])
    elif code == 8:     
        rel_start = np.array([-offset, dist])
        direction = np.array([0, -vel])
    elif code == 9:     
        rel_start = np.array([-dist, offset])
        direction = np.array([vel, 0])
    elif code == 10:   
        rel_start = np.array([-dist, 0])
        direction = np.array([vel, 0])
    elif code == 11: 
        rel_start = np.array([-dist, -offset])
        direction = np.array([vel, 0])
    
    pos = frame.getPosition()[:2] + R.dot(rel_start)
    direction = R.dot(direction)
    return list(pos) + [frame.getPosition()[2]] ,  list(direction) + [0.]  

class Game:

    def __init__(self, world_configuration = ABS_PATH+ "/scenarios/pushSimWorld.g"):
        self.C = ry.Config()
        self.C.addFile(world_configuration)
        self.S = self.C.simulation(ry.SimulatorEngine.bullet, False)
        self.S_verbose = self.C.simulation(ry.SimulatorEngine.bullet, True)
        
        self.tau = 0.02
        self.box = self.C.getFrame("box")
        self.box_t = self.C.getFrame("box_t")
        self.ball = self.C.getFrame("ball")
        
        self.disc_r = .01
        self.disc_r_ball = .1
        self.disc_angle = .3 # TODO: define meaningful angle
        self.state = 5*[0]
        self.start_r = 0
        self.start_angle = 0
        self.score = 0
        self.r_target_hits = 0

        self.max_dangle_dis = 4
        self.max_dr_dis = 16
        
        self.max_r = 1.
        self.max_r_ball_box = .6

        self.min_r_target_hits = 0
        self.initial_reward_for_good_guess = .3
        self.reset_box_pos = True

        # random initialization 
        self.reset()
        
    def calculate_state(self):
        x_diff_box, y_diff_box = get_xy_position_diff(self.box_t, self.box)
        x_diff_ball, y_diff_ball = get_xy_position_diff(self.box, self.ball)
        z_angle_diff = get_z_angle_diff(self.box_t, self.box)
        
        # self.state = [x_diff_box, y_diff_box, z_angle_diff, x_diff_ball, y_diff_ball]
        self.state = [x_diff_box, y_diff_box, z_angle_diff]


        
    def step(self, action, action_decoder, show_simulation = False):
        reward = 0
        game_over = False

        start, direction = start_direction(self.box, np.nonzero(action)[0][0])
        self.ball.setPosition(start)

        # direction = action_decoder(np.nonzero(action)[0][0])

        # Show output in simulation
        if show_simulation:
            self.S_verbose.setState(self.C.getFrameState())

            for t in range(3):
                time.sleep(self.tau)
                self.S_verbose.step(direction, self.tau,  ry.ControlMode.velocity)

        else:
            self.S.setState(self.C.getFrameState())

            for t in range(3):
                # time.sleep(self.tau)
                self.S.step(direction, self.tau,  ry.ControlMode.velocity)
        
        
        self.calculate_state()
        
        r = (self.state[0]**2 + self.state[1]**2)**.5
        
        dr = int(r / self.disc_r)
        dangle = int(self.state[2] / self.disc_angle)

        # r_ball_box = (self.state[3]**2 + self.state[4]**2)**.5
        r_ball_box = 0
        
        dr_ball_box = int(r_ball_box / self.disc_r_ball)

        if r_ball_box >= self.max_r_ball_box:
            ## Ball too far away FROM THE BOX
            # reward = -0.03
            print("Ball drifted: ", r_ball_box)
            game_over = True                
        
        if r >= self.max_r: 
            ## too big DISTANCE FROM TARGET
            # reward = -0.03
            self.reset_box_pos = True # allow change of box position only if some progress was achieved
            print("Distance too large: ", r)
            game_over = True        
         
        # reward += 10 - r*5 - r_ball_box*3

        # elif (dr - self.prev_dr) > self.max_dr_dis :
        #     ## too big DISTANCE FROM LAST ACHIEVEMENT
        #     game_over = True
        #     reward = -0.01
        #     print("Negative reward for position distance increase: ", reward)
        #     self.prev_dr = dr

        # elif (abs(dangle) - abs(self.prev_dangle)) > self.max_dangle_dis :
        #     ## too big ANGLE INCREASE FROM LAST ACHIEVEMENT
        #     game_over = True
        #     reward = -0.01
        #     print("Negative reward for angle increase: ", reward)
        #     self.prev_dangle = dangle
            
        if r < 0.1:
            ## Score
            self.r_target_hits += 1
            # Define desired more precise positioning 
            if r < 0.01 or self.r_target_hits > self.min_r_target_hits:
                print("Scored position: ", r)
                game_over = True
                self.reset_box_pos = True # allow change of box position only if some progress was achieved
                # Introduce better rewards for more precise positioning 
                reward = 10 - abs(dangle) - r*5

        
        # if dr_ball_box < self.prev_dr_ball:   
        #     reward += 0.5
        #     self.prev_dr_ball = dr_ball_box
        
        

        # Bias for movements towards the goal
        # update discrete states:
        # if abs(dangle) < abs(self.prev_dangle):
        #     reward += self.initial_reward_for_good_guess
        #     self.prev_dangle = dangle

        if dr < self.prev_dr:
            self.reset_box_pos = True # allow change of box position only if some progress was achieved
            reward += min(.2, .2*(self.prev_dr - dr))
            self.prev_dr = dr

        if dr > self.prev_dr:
            self.reset_box_pos = True # allow change of box position only if some progress was achieved
            reward += min(.01, .01*(dr - self.prev_dr))
            self.prev_dr = dr

        # reset variables
        if game_over:
            self.r_target_hits = 0

        self.score += reward

        return reward, game_over, self.score
        
    def get_state(self):
        return self.state
    
    def reset(self, random_box_pos = True, random_ball_pos = True):
        #define the new state of the box to be somewhere around the target:
        max_box_dist_target = .5
        if random_box_pos and self.reset_box_pos:
            # init in the circle around the target
            init_radius = .5
            alpha = 2*math.pi*np.random.rand()
            x = init_radius*math.cos(alpha)
            y = init_radius*math.sin(alpha)

            new_state = np.array(self.box_t.getPosition())+np.array([x, y, 0])

            self.box.setQuaternion(psi_to_quat(2*math.pi*np.random.rand()))
            self.box.setPosition(new_state)
            self.reset_box_pos = False

        # define ball state around the box
        if random_ball_pos:
            # random ball orientation
            init_radius_ball = .15
            alpha = 2*math.pi*np.random.rand()
            x = init_radius_ball*math.cos(alpha)
            y = init_radius_ball*math.sin(alpha)

            new_state_ball = np.array(self.box.getPosition())+np.array([x, y, 0])
            self.ball.setPosition(new_state_ball)

        self.S.setState(self.C.getFrameState())

        self.calculate_state()
        
        r = (self.state[0]**2 + self.state[1]**2)**.5
        # r_ball = (self.state[3]**2 + self.state[4]**2)**.5
        r_ball =0
        
        self.prev_dr = int(r / self.disc_r)
        self.init_dr = self.prev_dr
        # self.prev_dr = dr
        self.prev_dr_ball = int(r_ball / self.disc_r_ball)
        self.prev_dangle = int(self.state[2] / self.disc_angle)
        
        self.score = 0
        
        # Updating the environment
        # self.S.step([], 0,  ry.ControlMode.none)
        
#         for t in range(2):
#             time.sleep(self.tau)
#             self.S.step([], self.tau,  ry.ControlMode.none)            

        return 
    
class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(1.0, float(t) / self.schedule_timesteps)
        return self.initial_p + (self.final_p - self.initial_p) * fraction



class Worker:
    def __init__(self, 
                    # epsilon params
                    fraction_eps = 0.01, 
                    initial_eps = .5, 
                    final_eps = 0.05, 

                    # learning 
                    max_steps = 10_000_000, 
                    max_game_itr = 3_000,
                    gamma = 0.97, 
                    learning_rate = 1e-4, 
                    learning_start_itr = 100, 
                    target_q_update_freq = 500,
                    train_q_freq = 10,

                    # memory 
                    memory_len = 1_000, #10_000 in google code
                    batch_size = 64,

                    # network
                    layers_sizes = [3, 256, 64, 9],

                    # logging
                    log_freq = 100,
                    log_dir = "data/local/game_ball1",
                    
                    # simulation verbose
                    sim_verbose_freq_episodes = 20
                ):

        lib.logger.session(log_dir).__enter__()
        self.target_q_update_freq = target_q_update_freq
        self.learning_start_itr = learning_start_itr
        self.log_freq = log_freq
        self.train_q_freq = train_q_freq
        self.layers_sizes = layers_sizes
        self.act_dim = self.layers_sizes[-1]
        self.max_steps = max_steps
        self.max_game_itr = max_game_itr
        self.sim_verbose = True
        self.sim_verbose_freq_episodes = sim_verbose_freq_episodes

        # Learning Agent 
        self.agent = Agent(
            gamma = gamma, 
            learning_rate = learning_rate, 
            memory_len = memory_len, 
            layers_sizes = self.layers_sizes,
            batch_size = batch_size
            )

        # Environment
        self.game = Game()

        # Tactics for exploraion/exploitation
        self.exploration = LinearSchedule(
            schedule_timesteps=int(fraction_eps * max_steps),
            initial_p=initial_eps,
            final_p=final_eps)

    def dec_ball_action(self, encoded_action):
        vel = 1

        psi = (2*math.pi/self.act_dim)* encoded_action
        R = np.array([[math.cos(psi), -math.sin(psi)],
                     [math.sin(psi), math.cos(psi)]])

        direction = np.array([vel, 0])
        direction = list(R.dot(direction)) + [0.] 

        # reduce to 0. in case really small number
        return [0. if abs(d) < 1e-10 else d for d in direction] 

    def eps_greedy(self, state, epsilon):
        act = [0]*self.act_dim

        # Check Q function, do argmax.
        rnd = np.random.rand()
        if rnd > epsilon:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.agent.model(state0)
            move = torch.argmax(prediction).item()
            act[move] = 1
        else:
            act[np.random.randint(0, self.act_dim)] = 1
        
        return act

    def train(self):
        episode_rewards = []
        log_itr = 0
        episodes = 0
        last_game_itr = 0
        episode_return_mean_record = 0

        l_episode_return = deque([], maxlen=30)
        l_tq_squared_error = deque(maxlen=50)   
        
        for itr in range(self.max_steps):
            # get old state
            state_old = self.game.get_state()

            # get move
            act = self.eps_greedy(state_old, self.exploration.value(itr))

            # perform move and get new state
            reward, done, score = self.game.step(act, self.dec_ball_action, self.sim_verbose)
            state_new = self.game.get_state()

            episode_rewards.append(reward)

            # train short memory
            # self.agent.train_short_memory(state_old, act, reward, state_new, done)

            # remember
            self.agent.remember(state_old, act, reward, state_new, done)

            if done or (itr-last_game_itr)>self.max_game_itr:
                if reward == 0:
                    self.game.reset(random_box_pos=False)
                else:
                    self.game.reset()

                # turn off simulation output if it was on
                if self.sim_verbose:
                    self.sim_verbose = False               

                # Print basic game statistics
                print("~~~~~~~~~~ Game ~~~~~~~~~~")
                print('Running steps:           \t', itr - last_game_itr)

                # save to indicate how many iterations for start to finish of the game:
                last_game_itr = itr
                
                # Reward statistics
                episode_return = np.sum(episode_rewards)
                episodes +=1
                episode_rewards = []
                print('Reward:                  \t', episode_return)
                
                l_episode_return.append(episode_return)

                # Save the model if the preformance increased
                episode_return_mean = math.ceil(np.mean(l_episode_return)*10)/10

                if  episode_return_mean > episode_return_mean_record:
                    episode_return_mean_record = episode_return_mean
                    self.agent.model.save()
                    print('New mean reward record:  \t', episode_return_mean_record)

                print("~~~~~~~~~~~~~~~~~~~~~~~~~~")

                if episodes % self.sim_verbose_freq_episodes == 0:
                    self.sim_verbose = True

            if itr % self.target_q_update_freq == 0 and itr > self.learning_start_itr:
                self.agent.trainer.update_target_model()

            if itr % self.train_q_freq == 0 and itr > self.learning_start_itr:
                td_squared_error = self.agent.train_long_memory().data
                l_tq_squared_error.append(td_squared_error)
            # else: 
                # self.agent.train_short_memory(state_old, act, reward, state_new, done)

            if (itr + 1) % self.log_freq == 0 and len(l_episode_return) > 5:
                log_itr += 1
                lib.logger.logkv('Iteration', itr)
                lib.logger.logkv('Steps', log_itr)
                lib.logger.logkv('Epsilon', self.exploration.value(itr))
                lib.logger.logkv('Episodes', episodes)
                lib.logger.logkv('AverageReturn', np.mean(l_episode_return))
                lib.logger.logkv('TDError^2', np.mean(l_tq_squared_error))
                lib.logger.dumpkvs()

    def train_warm_start(self, model_filename = 'model.pth'):
        self.agent.trainer.model.restore_from_saved(model_filename)
        self.agent.trainer.update_target_model()
        self.train()

    def evaluate_model(self, model_filename = 'model.pth'):
        self.agent.trainer.model.restore_from_saved(model_filename)
        # Model should be switched to evaluation model and back when training is desired: mode.train()
        self.agent.trainer.model.eval() 

        episode_rewards = []
        episodes = 0
        last_game_itr = 0

        l_episode_return = deque([], maxlen=30)

        for itr in range(self.max_steps):
            state = self.game.get_state()
            act = [0]*self.act_dim

            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.agent.model(state0)
            move = torch.argmax(prediction).item()
            act[move] = 1

            # perform move
            reward, done, score = self.game.step(act, self.dec_ball_action, True)

            episode_rewards.append(reward)

            if done:
                self.game.reset()

                # Print basic game statistics
                print("~~~~~~~~~~ Game ~~~~~~~~~~")
                print('Running steps:           \t', itr - last_game_itr)

                # save to indicate how many iterations for start to finish of the game:
                last_game_itr = itr
                
                # Reward statistics
                episode_return = np.sum(episode_rewards)
                episodes +=1
                episode_rewards = []
                print('Reward:                  \t', episode_return)
                
                l_episode_return.append(episode_return)

                # Save the model if the preformance increased
                episode_return_mean = math.ceil(np.mean(l_episode_return)*10)/10.

                print('Mean reward:             \t', episode_return_mean)

                print("~~~~~~~~~~~~~~~~~~~~~~~~~~")


            
EVALUATION = False
WARM_START = False
if __name__ == "__main__":
    setup = Worker()
    if EVALUATION:
        setup.evaluate_model(model_filename = 'model_saved2.pth')
    elif WARM_START:
        setup.train_warm_start(model_filename = 'model.pth')
    else:
        setup.train()