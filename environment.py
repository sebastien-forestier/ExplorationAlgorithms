import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.patches import Circle


from explauto.environment.dynamic_environment import DynamicEnvironment
from modular_environment import HierarchicalEnvironment

from explauto.utils import bounds_min_max
from explauto.environment.environment import Environment
from explauto.environment.simple_arm.simple_arm import joint_positions
from explauto.utils.utils import rand_bounds


class Arm(Environment):
    use_process = True
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 lengths, angle_shift, rest_state):
        
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.lengths = lengths
        self.angle_shift = angle_shift
        self.rest_state = rest_state
        self.reset()
        
    def reset(self):
        #print "reset gripper"
        self.logs = []
        self.lines = None
        
    def compute_motor_command(self, m):
        return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)
        #return m

    def compute_sensori_effect(self, m):
        a = self.angle_shift + np.cumsum(np.array(m))
        a_pi = np.pi * a 
        hand_pos = np.array([np.sum(np.cos(a_pi)*self.lengths), np.sum(np.sin(a_pi)*self.lengths)])
        angle = np.mod(a[-1] + 1, 2) - 1
        self.logs.append(m)
        return [hand_pos[0], hand_pos[1], angle]    
        
    def plot(self, ax, i, **kwargs_plot):
        m = self.logs[i]
        angles = np.array(m)
        angles[0] += self.angle_shift
        x, y = joint_positions(angles, self.lengths, 'std')
        x, y = [np.hstack((0., a)) for a in x, y]
        l = []
        l += ax.plot(x, y, 'grey', lw=4, **kwargs_plot)
        l += ax.plot(x[0], y[0], 'sk', ms=8, **kwargs_plot)
        for i in range(len(self.lengths)-1):
            l += ax.plot(x[i+1], y[i+1], 'ok', ms=8, **kwargs_plot)
        l += ax.plot(x[-1], y[-1], 'or', ms=8, **kwargs_plot)
        self.lines = l
        return l 
        
    def plot_update(self, ax, i, **kwargs_plot):
        if self.lines is None:
            self.plot(ax, 0, **kwargs_plot)
        m = self.logs[i]
        angles = np.array(m)
        angles[0] += self.angle_shift
        x, y = joint_positions(angles, self.lengths, 'std')
        x, y = [np.hstack((0., a)) for a in x, y]
        l = []
        l += [[x, y]]
        l += [[x[0], y[0]]]
        for i in range(len(self.lengths)-1):
            l += [[x[i+1], y[i+1]]]
        l += [[x[-1], y[-1]]]
        for (line, data) in zip(self.lines, l):
            line.set_data(data[0], data[1])
        return self.lines

        
class Ball(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 size, initial_position, color='y'):
        
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.size = size
        self.size_sq = size * size
        self.color = color
        self.initial_position = initial_position
        self.reset()
        
        
    def reset(self):
        self.move = False
        self.circle = None
        self.pos = np.array(self.initial_position)
        self.logs = []
        
    def compute_motor_command(self, m):
        return m

    def compute_sensori_effect(self, m):
        if self.move or ((m[0] - self.pos[0]) ** 2 + (m[1] - self.pos[1]) ** 2 < self.size_sq):
            self.pos = m[0:2]
            self.move = 1
        self.logs.append([self.pos,
                          self.move])
        return list(self.pos)
    
    def plot(self, ax, i, **kwargs_plot):
        self.logs = self.logs[-50:]
        pos = self.logs[i][0]     
        self.circle = Circle((pos[0], pos[1]), self.size, fc=self.color, **kwargs_plot)
        ax.add_patch(self.circle)  
        return [self.circle]
        
    def plot_update(self, ax, i, **kwargs_plot):
        if self.circle is None:
            self.plot(ax, 0, **kwargs_plot)
        self.logs = self.logs[-50:]
        pos = self.logs[i][0]    
        self.circle.center = tuple(pos)
        return [self.circle]
        
        
class ArmBall(DynamicEnvironment):
    def __init__(self):
        
        arm_config = dict(
            m_mins=[-1.] * 3,
            m_maxs=[1.] * 3, 
            s_mins=[-1.] * 3,
            s_maxs=[1.] * 3, 
            lengths=[0.5, 0.3, 0.2], 
            angle_shift=0.5,
            rest_state=[0.] * 3)
        
        ball_config = dict(
            m_mins=[-1.] * 2,
            m_maxs=[1.] * 2, 
            s_mins=[-1.] * 2,
            s_maxs=[1.] * 2,
            size=0.05,
            initial_position=[0.6, 0.6],
            color="y")
                
        arm_ball_cfg = dict(
            m_mins=[-1.] * 3,
            m_maxs=[1.] * 3,
            s_mins=[-1.] * 2,
            s_maxs=[1.] * 2,
            top_env_cls=Ball, 
            lower_env_cls=Arm, 
            top_env_cfg=ball_config, 
            lower_env_cfg=arm_config, 
            fun_m_lower= lambda m:m,
            fun_s_lower=lambda m,s:s[0:2],
            fun_s_top=lambda m,s_lower,s:s)
        
        dynamic_environment_config = dict(
            env_cfg=arm_ball_cfg,
            env_cls=HierarchicalEnvironment,
            m_mins=[-1.] * 3 * 3, 
            m_maxs=[1.] * 3 * 3, 
            s_mins=[-1] * 3 * 2,
            s_maxs=[1] * 3 * 2,
            n_bfs=3,
            move_steps=50, 
            n_dynamic_motor_dims=3,
            n_dynamic_sensori_dims=2, 
            max_params=1000)
        
        DynamicEnvironment.__init__(self, **dynamic_environment_config)
        
       
class Stick(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 length, type, handle_tol, handle_noise, rest_state):
        
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.length = length
        self.type = type
        self.handle_tol = handle_tol
        self.handle_tol_sq = handle_tol * handle_tol
        self.handle_noise = handle_noise
        self.rest_state = rest_state
        
        self.reset()


    def reset(self):
        #print "reset Stick"
        self.held = False
        self.handle_pos = np.array(self.rest_state[0:2])
        self.angle = self.rest_state[2]
        self.compute_end_pos()
        self.logs = []
        
    def compute_end_pos(self):
        a = np.pi * self.angle
        self.end_pos = [self.handle_pos[0] + np.cos(a) * self.length, 
                        self.handle_pos[1] + np.sin(a) * self.length]
                
    def compute_motor_command(self, m):
        #return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)
        return m

    def compute_sensori_effect(self, m):
        hand_pos = m[0:2]
        hand_angle = m[2]
        
        if not self.held:
            if (hand_pos[0] - self.handle_pos[0]) ** 2. + (hand_pos[1] - self.handle_pos[1]) ** 2. < self.handle_tol_sq:
                self.handle_pos = hand_pos
                self.angle = np.mod(hand_angle + self.handle_noise * np.random.randn() + 1, 2) - 1
                self.compute_end_pos()
                self.held = True
        else:
            self.handle_pos = hand_pos
            self.angle = np.mod(hand_angle + self.handle_noise * np.random.randn() + 1, 2) - 1
            self.compute_end_pos()
        
        #print "Stick log added"
        self.logs.append([self.handle_pos, 
                          self.angle, 
                          self.end_pos, 
                          self.held])
        #print "Tool hand_pos:", hand_pos, "hand_angle:", hand_angle, "gripper_change:", gripper_change, "self.handle_pos:", self.handle_pos, "self.angle:", self.angle, "self.held:", self.held 
        return list(self.end_pos) # Tool pos
    
    def plot(self, ax, i, **kwargs_plot):
        handle_pos = self.logs[i][0]
        end_pos = self.logs[i][2]
        
        
        ax.plot([handle_pos[0], end_pos[0]], [handle_pos[1], end_pos[1]], '-', color=colors_config['stick'], lw=6, **kwargs_plot)
        ax.plot(handle_pos[0], handle_pos[1], 'o', color = colors_config['gripper'], ms=12, **kwargs_plot)
        if self.type == "magnetic":
            ax.plot(end_pos[0], end_pos[1], 'o', color = colors_config['magnetic'], ms=12, **kwargs_plot)
        else:
            ax.plot(end_pos[0], end_pos[1], 'o', color = colors_config['scratch'], ms=12, **kwargs_plot)
                    
    


class Ball2(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 object_tol_hand, object_tol_tool, bounds):
        
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.object_tol_hand_sq = object_tol_hand * object_tol_hand
        self.object_tol_tool_sq = object_tol_tool * object_tol_tool
        self.bounds = bounds
        self.reset()
        
        
    def reset(self):
        #print "reset object"
        self.move = 0
        self.pos = rand_bounds(self.bounds)[0]
        self.logs = []
        
    def compute_motor_command(self, m):
        #return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)
        return m

    def compute_sensori_effect(self, m):
        if self.move == 1 or (abs(m[2] + 0.96213203) < 0.0001 and ((m[0] - self.pos[0]) ** 2 + (m[1] - self.pos[1]) ** 2 < self.object_tol_hand_sq)):
            self.pos = m[0:2]
            self.move = 1
        if self.move == 2 or (self.move == 0 and ((m[2] - self.pos[0]) ** 2 + (m[3] - self.pos[1]) ** 2 < self.object_tol_tool_sq)):
            self.pos = m[2:4]
            self.move = 2
#         if self.move == 3 or (self.move == 0 and (m[5] - self.pos[0]) ** 2 + (m[6] - self.pos[1]) ** 2 < self.object_tol_tool_sq):
#             self.pos = m[5:7]
#             self.move = 3
            #print "object moved by tool2"
        self.logs.append([self.pos,
                          self.move])
        return list(self.pos)
    
    def plot(self, ax, i, **kwargs_plot):
        self.logs = self.logs[-50:]
        pos = self.logs[i][0]        
        rectangle = plt.Rectangle((pos[0] - 0.05, pos[1] - 0.05), 0.1, 0.1, **kwargs_plot)
        ax.add_patch(rectangle) 



class ArmStickBall(DynamicEnvironment):
    def __init__(self, move_steps=50, max_params=None, noise=0, gui=False):

            
        arm_cfg = dict(m_mins=[-1, -1, -1],  # joints pos
                             m_maxs=[1, 1, 1], 
                             s_mins=[-1, -1, -1], # hand pos + hand angle 
                             s_maxs=[1, 1, 1], 
                             lengths=[0.5, 0.3, 0.2], 
                             angle_shift=0.5,
                             rest_state=[0., 0., 0.])
        
        
        stick1_cfg = dict(m_mins=[-1, -1, -1], 
                         m_maxs=[1, 1, 1], 
                         s_mins=[-2, -2],  # Tool pos
                         s_maxs=[2, 2],
                         length=0.3, 
                         type="1",
                         handle_tol=0.1, 
                         handle_noise=0.1 if noise == 1 else 0., 
                         rest_state=[-0.75, 0.25, 0.75])
        
        
        arm_stick_cfg = dict(m_mins=list([-1.] * 3), # 3DOF + gripper
                             m_maxs=list([1.] * 3),
                             s_mins=list([-2.] * 4),
                             s_maxs=list([2.] * 4),
                             top_env_cls=Stick, 
                             lower_env_cls=Arm, 
                             top_env_cfg=stick1_cfg, 
                             lower_env_cfg=arm_cfg, 
                             fun_m_lower= lambda m:m,
                             fun_s_lower=lambda m,s:s,  # (hand pos + hand angle) * 2 tools
                             fun_s_top=lambda m,s_lower,s:s_lower[0:2] + s) # from s: Tool1 end pos  from m: hand_pos
        
        
        
        object_cfg = dict(m_mins = list([-1.] * 4), 
                          m_maxs = list([1.] * 4), 
                          s_mins = [-2., -2.], # new pos
                          s_maxs = [2., 2.],
                          object_tol_hand = 0.2, 
                          object_tol_tool = 0.1,
                          bounds = np.array([[-0.5, -0.5],
                                                 [0.5, 0.5]]))
        
        
        def sensory_noise(s):
            return np.random.random(len(s)) * 0.1 + np.array(s)
        
        
        arm_sticks_object_cfg = dict(
                                   m_mins=arm_stick_cfg['m_mins'],
                                   m_maxs=arm_stick_cfg['m_maxs'],
                                   s_mins=list([-2.] * 6),
                                   s_maxs=list([2.] * 6), # (hand pos + tool1 end pos + last objects pos
                                   top_env_cls=Object, 
                                   lower_env_cls=HierarchicallyCombinedEnvironment, 
                                   top_env_cfg=object_cfg, 
                                   lower_env_cfg=arm_stick_cfg, 
                                   fun_m_lower= lambda m:m,
                                   fun_s_lower=lambda m,s:s,
                                   fun_s_top=lambda m,s_lower,s: sensory_noise(s_lower + s) if noise == 2 else s_lower + s)
        
        
        denv_cfg = dict(env_cfg=arm_sticks_object_cfg,
                        env_cls=HierarchicallyCombinedEnvironment,
                        m_mins=[-1.] * 3 * 3, 
                        m_maxs=[1.] * 3 * 3, 
                        s_mins=[-1.5] * 6 * 3,
                        s_maxs=[1.5] * 6 * 3,
                        n_bfs = 2,
                        n_motor_traj_points=3, 
                        n_sensori_traj_points=3, 
                        move_steps=move_steps, 
                        n_dynamic_motor_dims=3,
                        n_dynamic_sensori_dims=6, 
                        max_params=max_params,
                        motor_traj_type="DMP", 
                        sensori_traj_type="samples",
                        optim_initial_position=False, 
                        optim_end_position=True, 
                        default_motor_initial_position=[0.]*3, 
                        default_motor_end_position=[0.]*3,
                        default_sensori_initial_position=[0., 1., -0.85, 0.35, 0., 0.], 
                        default_sensori_end_position=[0., 1., -0.85, 0.35, 0., 0.],
                        gui=gui)
            
        
        DynamicEnvironment.__init__(self, **denv_cfg)
        
        
    @property
    def current_context(self):
        return self.env.top_env.pos
    
    # Change object sensory space to be 2D as relative change of position ds
    def compute_sensori_effect(self, m_traj):
        c = self.current_context
        s = DynamicEnvironment.compute_sensori_effect(self, m_traj)
        #print "s", s
        s_o_end = s[[-4,-1]]
        #print "s_o_end", s_o_end
        res = list(s[:-6]) + list(np.array(s_o_end) - np.array(c))
        #print res
        s_ = res
        #print "s_", s_
        obj_end_pos_y = s_o_end[1]
        tool1_moved = (abs(s_[-5] - s_[-3]) > 0.0001)
        #tool2_moved = (abs(ms[-5] - ms[-3]) > 0.0001)
        tool1_touched_obj = tool1_moved and (abs(s_[-3] - obj_end_pos_y) < 0.0001)
        #tool2_touched_obj = tool2_moved and (abs(ms[-3] - obj_end_pos_y) < 0.0001)
        obj_moved = abs(s_[-1]) > 0.0001
    
        obj_moved_with_hand = obj_moved and (not tool1_touched_obj)# and (not tool2_touched_obj)
        #print "obj_end_pos_y", obj_end_pos_y, "tool end y", s_[-3]
#         if tool1_moved:
#             print "tool moved"
#         if tool1_touched_obj:
#             print "object moved by tool"
#             #raise
#         elif obj_moved_with_hand:
#             print "object moved by hand"
             
        
        if tool1_touched_obj or (tool1_moved and not obj_moved_with_hand):
            tool_traj = [st[2:4] for st in self.s_traj]
            min_dist = min([np.linalg.norm(np.array(st) - np.array(s_o_end)) for st in tool_traj])
            #print min_dist
        else:
            hand_traj = [sh[:2] for sh in self.s_traj]
            min_dist = min([np.linalg.norm(np.array(sh) - np.array(s_o_end)) for sh in hand_traj])
        
        #print list(s[:-6]), [min_dist], list(np.array(s_o_end) - np.array(c))
        res = list(s[:-6]) + [min_dist] + list(np.array(s_o_end) - np.array(c))
        
        #print "s env", res
        self.env.lower_env.reset() # reset arm and tools but not object
        self.env.top_env.move = 0 # tools have been reset so object must not follow them
        return res
    
        