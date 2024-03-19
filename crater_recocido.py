import random
import numpy as np
import math 
import time 
import plotly.graph_objects as px 

class Point(object):
    def __init__(self, point, mars):
        self.point = point 
        self.map = mars

    def neighbor(self):
        while True:
            move = random.randint(0, 7)
            print(move)
            if move == 0:
                r = self.point[0] - 1 #y
                c = self.point[1] #x

                if self.map[r][c] >= 0 and abs(self.map[r][c] - self.map[self.point[0]][self.point[1]]) <= 2:
                    return Point((r,c), self.map)
                    break 

            if move == 1:
                r = self.point[0] - 1 
                c = self.point[1] + 1

                if self.map[r][c] >= 0 and abs(self.map[r][c] - self.map[self.point[0]][self.point[1]]) <= 2:
                    return Point((r,c), self.map)
                    break 
                
            
            if move == 2: 
                r = self.point[0] 
                c = self.point[1] + 1

                if self.map[r][c] >= 0 and abs(self.map[r][c] - self.map[self.point[0]][self.point[1]]) <= 2:
                    return Point((r,c), self.map)
                    break
                
            if move == 3: 
                r = self.point[0] + 1
                c = self.point[1] + 1

                if self.map[r][c] >= 0 and abs(self.map[r][c] - self.map[self.point[0]][self.point[1]]) <= 2:
                    return Point((r,c), self.map)
                    break 
                
            if move == 4: 
                r = self.point[0] + 1
                c = self.point[1] 

                if self.map[r][c] > 0 and abs(self.map[r][c] - self.map[self.point[0]][self.point[1]]) <= 2:
                    return Point((r,c), self.map)
                    break 
                
            if move == 5: 
                r = self.point[0] + 1
                c = self.point[1] - 1

                if self.map[r][c] > 0 and abs(self.map[r][c] - self.map[self.point[0]][self.point[1]]) <= 2:
                    return Point((r,c), self.map)
                    break

            if move == 6: 
                r = self.point[0] 
                c = self.point[1] - 1

                if self.map[r][c] > 0 and abs(self.map[r][c] - self.map[self.point[0]][self.point[1]]) <= 2:
                    return Point((r,c), self.map)
                    break 
                
            
                
            if move == 7: 
                r = self.point[0] - 1
                c = self.point[1] - 1

                if self.map[r][c] > 0 and abs(self.map[r][c] - self.map[self.point[0]][self.point[1]]) <= 2:
                    return Point((r,c), self.map)
                    break
        
    def costo(self):
        return self.map[self.point[0]][self.point[1]]
    

mars_map = np.load(f'C:/Users/Jose Carlos/Downloads/crater_map.npy')
nr, nc = mars_map.shape

scale = 10.045

def coordinates(y, x):
  r = nr - round(y/scale)
  c = round(x/scale)
  return (r,c)

r_i, c_i = coordinates(3766, 2993)

current_point = Point((r_i, c_i), mars_map)
  
current_cost = current_point.costo() # Initial cost  
print("Costo inicial: ", current_cost)  
step = 0                    # Step count

alpha = 0.999997              # Coefficient of the exponential temperature schedule        
t0 = 1                        # Initial temperature
t = t0    
moves = []
start_time = time.time()
while t > 0.0005 and current_cost > 1: # 

    # Calculate temperature
    t = t0 * math.pow(alpha, step)
    step += 1
        
    # Get random neighbor
    neighbor_point = current_point.neighbor()
    current_cost = current_point.costo()

    neighbor_cost = neighbor_point.costo()

    # Test neighbor
    
    if neighbor_cost < current_cost:
        current_point = neighbor_point

    else:
        # Calculate probability of accepting the neighbor
        p = math.exp((current_cost - neighbor_cost) / t)
        if p >= random.random():
            current_point = neighbor_point

    moves.append(list(current_point.point))
    print("Punto: ", current_point.point, "Iteration: ", step, "    Cost: ", current_cost, "    Temperature: ", t)

end_time = time.time()
elapsed_time = end_time - start_time  

path_x = []
path_y = []
path_z = []
prev_state = []
distance = 0
for r,c in moves:    
        path_x.append( c * scale  )            
        path_y.append((nr - r)*scale  )
        path_z.append(mars_map[r][c]+1)
        
        if len(prev_state) > 0:
            distance +=  math.sqrt(
            scale*scale*(r - prev_state[0])**2 + scale*scale*(c - prev_state[1])**2 + (
                mars_map[r, c] - mars_map[prev_state[0], prev_state[1]])**2)

        prev_state = (r,c)

print("Total distance", distance)

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