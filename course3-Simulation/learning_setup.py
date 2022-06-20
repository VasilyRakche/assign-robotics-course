import sys
ABS_PATH = "/home/vasko/Documents/TUB3/AI_Robotics/robotics-course"
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
    vel = 1
    dist = .21 
    offset = .1
    
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
        self.S = self.C.simulation(ry.SimulatorEngine.bullet, True)
        
        self.tau = 0.02
        self.box = self.C.getFrame("box")
        self.box_t = self.C.getFrame("box_t")
        self.ball = self.C.getFrame("ball")
        self.r_max = 2.
        self.disc_r = .3
        self.disc_angle = .5 # TODO: define meaningful angle
        self.state = 3*[0]
        self.start_r = 0
        self.start_angle = 0
        self.score = 0
        
        # random initialization 
        self.reset()
        
    def calculate_state(self):
        x_diff, y_diff = get_xy_position_diff(self.box_t, self.box)
        z_angle_diff = get_z_angle_diff(self.box_t, self.box)
        
        self.state = [x_diff, y_diff, z_angle_diff]

        
    def step(self, final_move):
        reward = 0
        game_over = False

        start, direction = start_direction(self.box,np.nonzero(final_move)[0][0])
        self.ball.setPosition(start)
        self.S.setState(self.C.getFrameState())
        
        # Updating the environment
        # self.S.step([], 0,  ry.ControlMode.none)

        for t in range(3):
            time.sleep(self.tau)
            self.S.step(direction, self.tau,  ry.ControlMode.velocity)
        
        
        self.calculate_state();
        
        r = (self.state[0]**2 + self.state[1]**2)**.5
        
        dr = int(r / self.disc_r)
        dangle = int(self.state[2] / self.disc_angle)
        
        
        if r >= self.r_max: 
            reward = -3.
            game_over = True        
            
        elif dr < self.prev_dr:
            #TODO: dangle adding strategy
            reward = 0.1*(self.prev_dr - dr)
            print("Positive reward for position increase: ", reward)
            self.prev_dr = dr
            
        elif r < 0.1:
            print("Scored position: ", r)
            game_over = True
            reward = 10 - abs(dangle)
            
         
        # penalize angle difference increase 
        if abs(self.prev_dangle) < abs(dangle):
            reward += -.5
            print("Negative reward for angle increase: ", reward)
            self.prev_dangle = dangle
            
        self.score += reward

        if (self.score) < -1.5:
            print("Score unaceptably low: ",self.score )
            game_over = True

        return reward, game_over, self.score
        
    def get_state(self):
        return self.state
    
    def reset(self):
        #define the new state of the box to be somewhere around the target:
        new_state = np.array(self.box_t.getPosition())+np.array([3*np.random.rand() - 1.5, 3*np.random.rand() - 1.5, 0])
        
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
                    fraction_eps = 0.01, 
                    initial_eps = .3, 
                    final_eps = 0.05, 

                    # learning 
                    max_steps = 10_000_000, 
                    gamma = 0.97, 
                    learning_rate = 1e-3, 
                    learning_start_itr = 0, 
                    train_q_freq = 50,

                    # memory 
                    memory_len = 100_000,
                    batch_size = 1000,

                    #network
                    layers_sizes = [3, 256, 12],

                    #logging
                    log_freq = 100,
                    log_dir = "data/local/game",
                ):

        lib.logger.session(log_dir).__enter__()
        self.log_freq = log_freq
        self.train_q_freq = train_q_freq
        self.layers_sizes = layers_sizes
        self.act_dim = self.layers_sizes[-1]
        self.max_steps = max_steps

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
        plot_scores = []
        plot_mean_scores = []
        episode_rewards = []
        total_score = 0
        record = 0     
        log_itr = 0

        l_episode_return = deque([], maxlen=10)
        l_tq_squared_error = deque(maxlen=50)   
        
        for itr in range(self.max_steps):
            # get old state
            state_old = self.game.get_state()

            # get move
            act = self.eps_greedy(state_old, self.exploration.value(itr))

            # perform move and get new state
            reward, done, score = self.game.step(act)
            state_new = self.game.get_state()

            episode_rewards.append(reward)

            # train short memory
            self.agent.train_short_memory(state_old, act, reward, state_new, done)

            # remember
            self.agent.remember(state_old, act, reward, state_new, done)

            if done:
                # train long memory, plot result
    #             print("done routine")
                self.game.reset()
                episode_return = np.sum(episode_rewards)
                episode_rewards = []

                if score > record:
                    record = score
                    self.agent.model.save()

                l_episode_return.append(episode_return)

                td_squared_error = self.agent.train_long_memory().data

                l_tq_squared_error.append(td_squared_error)
                print("DONE!!!")
                print('Iteration: ', log_itr)
                print('Steps: ', itr)
                print('Epsilon: ', self.exploration.value(itr))
                print('Episodes: ', len(l_episode_return))
                print('Reward: ', episode_return)

                # print('Game', itr, 'Score', score, 'Record:', record)

                # plot_scores.append(score)
                # total_score += score
                # mean_score = total_score / itr
                # plot_mean_scores.append(mean_score)
                # plot(plot_scores, plot_mean_scores)

            # if itr % self.train_q_freq == 0 and itr > self.learning_start_itr:
            #     td_squared_error = self.agent.train_long_memory()
            #     l_tq_squared_error.append(td_squared_error)

            if (itr + 1) % self.log_freq == 0:
                log_itr += 1
                lib.logger.logkv('Iteration', log_itr)
                lib.logger.logkv('Steps', itr)
                lib.logger.logkv('Epsilon', self.exploration.value(itr))
                lib.logger.logkv('Episodes', len(l_episode_return))
                lib.logger.logkv('AverageReturn', np.mean(l_episode_return))
                lib.logger.logkv('TDError^2', np.mean(l_tq_squared_error))
                lib.logger.dumpkvs()

if __name__ == "__main__":
    setup = Worker()
    setup.train()