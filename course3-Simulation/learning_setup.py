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
    vel = .6
    dist = .116 
    offset = .065   
    
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
        self.r_max = 2.4
        self.disc_r = .3
        self.disc_angle = .5 # TODO: define meaningful angle
        self.state = 3*[0]
        self.start_r = 0
        self.start_angle = 0
        self.score = 0
        self.r_target_hits = 0
        
        # random initialization 
        self.reset()
        
    def calculate_state(self):
        x_diff, y_diff = get_xy_position_diff(self.box_t, self.box)
        z_angle_diff = get_z_angle_diff(self.box_t, self.box)
        
        self.state = [x_diff, y_diff, z_angle_diff]

        
    def step(self, action, show_simulation = False):
        reward = 0
        game_over = False

        start, direction = start_direction(self.box,np.nonzero(action)[0][0])
        self.ball.setPosition(start)

        # Show output in simulation
        if show_simulation:
            self.S_verbose.setState(self.C.getFrameState())
            # for t in range(10):
            #     self.S_verbose.step([], 4*self.tau,  ry.ControlMode.none)

            for t in range(3):
                time.sleep(self.tau)
                # if t<3:
                self.S_verbose.step(direction, self.tau,  ry.ControlMode.velocity)
                # else:
                # self.S_verbose.step([], self.tau,  ry.ControlMode.none)

        else:
            self.S.setState(self.C.getFrameState())

            for t in range(6):
                # time.sleep(self.tau)
                # if t<3:
                self.S.step(direction, self.tau,  ry.ControlMode.velocity)
                # else:
                # self.S_verbose.step([], self.tau,  ry.ControlMode.none)
        
        
        self.calculate_state();
        
        r = (self.state[0]**2 + self.state[1]**2)**.5
        
        dr = int(r / self.disc_r)
        dangle = int(self.state[2] / self.disc_angle)
        
        
        if r >= self.r_max: 
            reward = -3.
            print("Distance too large: ", r)
            game_over = True        
            
        elif (dr - self.prev_dr) > 2 :
            game_over = True
            #TODO: dangle adding strategy
            reward = -1
            print("Negative reward for position distance increase: ", reward)
            self.prev_dr = dr
            
        elif r < 0.1:
            self.r_target_hits += 1
            # Define desired more precise positioning 
            if r < 0.01 or self.r_target_hits > 5:
                print("Scored position: ", r)
                game_over = True
                # Introduce better rewards for more precise positioning 
                reward = 10 - abs(dangle) - r*5
            
        # penalize angle difference increase 
        if (abs(dangle) - abs(self.prev_dangle)) > 2 :
            game_over = True
            reward = -1
            print("Negative reward for angle increase: ", reward)
            self.prev_dangle = dangle
            
        self.score += reward

        # Bias for movements towards the goal
        # update discrete states:
        if abs(dangle) < abs(self.prev_dangle):
            reward+=.02
            self.prev_dangle = dangle

        if dr <  self.prev_dr:
            reward+=.02
            self.prev_dr = dr

        # reset variables
        if game_over:
            self.r_target_hits = 0

        return reward, game_over, self.score
        
    def get_state(self):
        return self.state
    
    def reset(self):
        #define the new state of the box to be somewhere around the target:
        reset_max_dist = 1.5
        new_state = np.array(self.box_t.getPosition())+np.array([reset_max_dist*np.random.rand() - reset_max_dist/2, reset_max_dist*np.random.rand() - reset_max_dist/2, 0])

        self.box.setQuaternion(psi_to_quat(2*math.pi*np.random.rand()))
        self.box.setPosition(new_state)
        self.S.setState(self.C.getFrameState())
        
        self.calculate_state();
        
        r = (self.state[0]**2 + self.state[1]**2)**.5
        
        self.prev_dr = int(r / self.disc_r)
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
                    fraction_eps = 0.004, 
                    initial_eps = .3, 
                    final_eps = 0.05, 

                    # learning 
                    max_steps = 10_000_000, 
                    gamma = 0.97, 
                    learning_rate = 1e-3, 
                    learning_start_itr = 100, 
                    target_q_update_freq = 600,
                    train_q_freq = 2,

                    # memory 
                    memory_len = 1_000, #10_000 in google code
                    batch_size = 64,

                    # network
                    layers_sizes = [3, 256, 64, 12],

                    # logging
                    log_freq = 100,
                    log_dir = "data/local/game3",
                    
                    # simulation verbose
                    sim_verbose_freq_episodes = 100
                ):

        lib.logger.session(log_dir).__enter__()
        self.target_q_update_freq = target_q_update_freq
        self.learning_start_itr = learning_start_itr
        self.log_freq = log_freq
        self.train_q_freq = train_q_freq
        self.layers_sizes = layers_sizes
        self.act_dim = self.layers_sizes[-1]
        self.max_steps = max_steps
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
            reward, done, score = self.game.step(act, self.sim_verbose)
            state_new = self.game.get_state()

            episode_rewards.append(reward)

            # train short memory
            # self.agent.train_short_memory(state_old, act, reward, state_new, done)

            # remember
            self.agent.remember(state_old, act, reward, state_new, done)

            if done:
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

            if (itr + 1) % self.log_freq == 0 and len(l_episode_return) > 5:
                log_itr += 1
                lib.logger.logkv('Iteration', log_itr)
                lib.logger.logkv('Steps', itr)
                lib.logger.logkv('Epsilon', self.exploration.value(itr))
                lib.logger.logkv('Episodes', len(l_episode_return))
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
            reward, done, score = self.game.step(act, True)

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
WARM_START = True
if __name__ == "__main__":
    setup = Worker()
    if EVALUATION:
        setup.evaluate_model(model_filename = 'model_saved2.pth')
    elif WARM_START:
        setup.train_warm_start(model_filename = 'model.pth')
    else:
        setup.train()