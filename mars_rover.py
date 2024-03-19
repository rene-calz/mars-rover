from simpleai.search import SearchProblem, breadth_first, depth_first, astar #Importamos la clase SearchProblem y la función del algoritmo de búsqueda
import math
import random
import time
import plotly.graph_objects as px
import numpy as np
import matplotlib.pyplot as plt
mars_map = np.load(f'C:/Users/areba/Downloads/mars_map.npy')
nr, nc = mars_map.shape

mars_map

mars_map.shape

def coordinates(y, x):
  scale = 10.0174
  r = nr - round(y/scale)
  c = round(x/scale)
  return (r,c)

costos = {'up': 1.0,
         'down': 1.0,
         'right': 1.0,
         'left': 1.0,
         'up, left': 1.4, #pitágoras
         'up, right': 1.4,
         'down, left': 1.4,
         'down, right': 1.4}

class MarsRover(SearchProblem):
  def __init__ (self, mars, initial_pos, goal_pos, barrier = 0.25):
    """
    maze: string
    initial_pos: set of 2 integer elements
    goal_pos: set of 2 integer elements
    """
    SearchProblem.__init__(self, initial_pos)
    self.barrier = barrier
    self.mars = mars
    self.initial = initial_pos
    self.goal = goal_pos

  def actions(self, state):
    act = []

    if self.mars[state[0]][state[1] + 1]  >= 0 and abs(self.mars[state[0]][state[1] + 1] - self.mars[state[0]][state[1]]) < self.barrier:
      act.append('right')

    if self.mars[state[0]][state[1] - 1] >= 0 and abs(self.mars[state[0]][state[1] - 1] - self.mars[state[0]][state[1]]) < self.barrier:
      act.append('left')

    if self.mars[state[0] + 1][state[1]] >= 0 and abs(self.mars[state[0] + 1][state[1]] - self.mars[state[0]][state[1]]) < self.barrier:
      act.append('down')

    if self.mars[state[0] - 1][state[1]] >= 0 and abs(self.mars[state[0] - 1][state[1]] - self.mars[state[0]][state[1]]) < self.barrier:
      act.append('up')

    if self.mars[state[0] + 1][state[1] + 1] >= 0 and abs(self.mars[state[0] + 1][state[1] + 1] - self.mars[state[0]][state[1]]) < self.barrier:
      act.append('down, right')

    if self.mars[state[0] - 1][state[1] + 1] >= 0 and abs(self.mars[state[0] - 1][state[1] + 1] - self.mars[state[0]][state[1]]) < self.barrier:
      act.append('up, right')

    if self.mars[state[0] + 1][state[1] - 1] >= 0 and abs(self.mars[state[0] + 1][state[1] - 1] - self.mars[state[0]][state[1]]) < self.barrier:
      act.append('down, left')

    if self.mars[state[0] - 1][state[1] - 1] >= 0 and abs(self.mars[state[0] - 1][state[1] - 1] - self.mars[state[0]][state[1]]) < self.barrier:
      act.append('up, left')


    return act

  def result (self, state, action):
    x = state[1]
    y = state[0]

    if action == 'right':
      x += 1
    if action == 'left':
      x -= 1
    if action == 'down':
      y += 1
    if action == 'up':
      y -= 1
    if action == 'down, right':
      x += 1
      y += 1
    if action == 'up, right':
      x += 1
      y -= 1
    if action == 'down, left':
      x -= 1
      y += 1
    if action == 'up, left':
      x -= 1
      y -= 1

    return (y, x)

  def is_goal(self, state):
    return state == self.goal

  def cost(self, state, action, state2):
    return costos[action]

  def heuristic(self, state):
    y,x = state
    gy, gx = self.goal
    distance = math.sqrt((gy - y)**2 + (gx - x)**2)
    return distance

r_i , c_i = coordinates(7603, 5709)
r_f, c_f = coordinates(8615, 6351)

initial_state = (r_i, c_i)
final_state = (r_f, c_f)

mars_result = astar(MarsRover(mars_map, initial_state, final_state), graph_search=True)
#states = []
# Print results

for i, (action, state) in enumerate(mars_result.path()):
    #states.append(state)
    print()
    if action == None:
        print('Initial configuration')
    elif i == len(mars_result.path()) - 1:
        print('After moving', action, 'Goal achieved!')
    else:
        print('After moving', action)


    print(state)

scale = 10.0174

row_ini = nr-round(r_i/scale)
col_ini = round(c_i/scale)

row_goal = nr-round(r_f/scale)
col_goal = round(c_f/scale)

if mars_result != None:
    path_x = []
    path_y = []
    path_z = []
    prev_state = []
    distance = 0
    for i, (action, state) in enumerate(mars_result.path()):    
        path_x.append( state[1] * scale  )            
        path_y.append(  (nr - state[0])*scale  )
        path_z.append(mars_map[state[0]][state[1]]+1)
        
        if len(prev_state) > 0:
            distance +=  math.sqrt(
            scale*scale*(state[0] - prev_state[0])**2 + scale*scale*(state[1] - prev_state[1])**2 + (
                mars_map[state[0], state[1]] - mars_map[prev_state[0], prev_state[1]])**2)

        prev_state = state

    print("Total distance", distance)

else:
    print("Unable to find a path between that connect the specified points")

## Plot results
if mars_result != None: 

    x = scale*np.arange(mars_map.shape[1])
    y = scale*np.arange(mars_map.shape[0])
    X, Y = np.meshgrid(x, y)

    fig = px.Figure(data = [px.Surface(x=X, y=Y, z=np.flipud(mars_map), colorscale='hot', cmin = 0, 
                                        lighting = dict(ambient = 0.0, diffuse = 0.8, fresnel = 0.02, roughness = 0.4, specular = 0.2),
                                        lightposition=dict(x=0, y=nr/2, z=2*mars_map.max())),
                        
                            px.Scatter3d(x = path_x, y = path_y, z = path_z, name='path', mode='markers',
                                            marker=dict(color=np.linspace(0, 1, len(path_x)), colorscale="Bluered", size=4))],
                
                    layout = px.Layout(scene_aspectmode='manual', 
                                        scene_aspectratio=dict(x=1, y=nr/nc, z=max(mars_map.max()/x.max(), 0.2)), 
                                        scene_zaxis_range = [0,mars_map.max()])
                    )
    fig.show() 