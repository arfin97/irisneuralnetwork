import numpy as np

class ParticleStructure():
    def __init__(self, nVar, seed):
        #np.random.seed(seed)
        self.position = np.random.randn(nVar, 1) * 0.1
        self.velocity = np.zeros((nVar, 1))
        self.cost = None
        self.best_position = self.position
        self.best_cost = self.cost

class Swarm():
    global_best_position = np.zeros((13002, 1))
    global_best_cost = np.Inf
    all_global_best = None
    X = None
    Y = None
    
    def __init__(self, nn, X, Y, maxIt, nPop):
        
        #problem statement
        self.nVar = np.size(nn.layers[1].weights_from_previous_layer) + \
                                 np.size(nn.layers[2].weights_from_previous_layer) + \
                                 np.size(nn.layers[3].weights_from_previous_layer) + \
                                 np.size(nn.layers[1].biases_for_this_layer) + \
                                 np.size(nn.layers[2].biases_for_this_layer) + \
                                 np.size(nn.layers[3].biases_for_this_layer)
        
        #constriction coefficients
        self.kappa = 1
        self.phi1 = 2.05
        self.phi2 = 2.05
        self.phi = self.phi1 + self.phi2
        self.chi = (2 * self.kappa) / abs(2-self.phi- np.sqrt((self.phi**2) - (4 * self.phi)))

        #parameters of PSO
        self.all_global_best = np.zeros((maxIt, 1))
        self.X = X
        self.Y = Y
        self.maxIt = maxIt                              #numer of iteration
        self.nPop = nPop                                #population size
        self.particle = []                              #all particles
        self.w = self.chi                               #intertia coefficient
        self.wdamp = 0.99                               #Damping Ratio of Inertia Coeffieicnt
        self.c1 = self.chi * self.phi1                  #personal acceleration coefficient
        self.c2 = self.chi * self.phi2                  #social acceleration coefficient
    
    def cost_function_pso(self, nn, particle):
        #run one forward prop of neural network
        nn.layers[1].weights_from_previous_layer = np.reshape(particle.position[0:12544],(16,784))
        nn.layers[2].weights_from_previous_layer = np.reshape(particle.position[12544:12800],(16,16))
        nn.layers[3].weights_from_previous_layer = np.reshape(particle.position[12800:12960],(10, 16))
        nn.layers[1].biases_for_this_layer = np.reshape(particle.position[12960:12976],(16,1))
        nn.layers[2].biases_for_this_layer = np.reshape(particle.position[12976:12992],(16,1))
        nn.layers[3].biases_for_this_layer = np.reshape(particle.position[12992:13002],(10, 1))
        nn.forward_propagation(self.X)
        return nn.calculate_network_loss(self.Y)
        
    def initialize_swarm(self, nn):
        for i in range (self.nPop):
            
            #initialize a Particle and add it to swarm list
            self.particle.append(ParticleStructure(self.nVar, i))
            
            #calculate the cost of a particle in current position
            self.particle[i].cost = self.cost_function_pso(nn, self.particle[i])
            print("Initialize Particle Cost: ",self.particle[i].cost)
            
            #update self best
            self.particle[i].best_position = self.particle[i].position
            self.particle[i].best_cost = self.particle[i].cost
            
            #update global best for whole swarm
            if self.particle[i].best_cost < self.global_best_cost:
                self.global_best_position = self.particle[i].best_position
                self.global_best_cost = self.particle[i].best_cost
            
    def pso_loop(self, nn):
        for i in range(self.maxIt):
            print("-------------------------------------------------")
            print("Round No: ", i)
            print("-------------------------------------------------")
            for j in range(self.nPop):
                print("*******************************")
                print("Particle No: ", j)
                print("*******************************")
                #update velocity
                self.particle[j].velocity = self.w * self.particle[j].velocity
                self.particle[j].velocity = self.particle[j].velocity\
                                            + self.c1 * np.multiply(np.random.randn(self.nVar,1) * 0.1,\
                                              (self.particle[j].best_position - self.particle[j].position))
                self.particle[j].velocity = self.particle[j].velocity\
                                            + self.c2 * np.multiply(np.random.randn(self.nVar,1) * 0.1,\
                                              (self.global_best_position - self.particle[j].position))
                #update particle position
                self.particle[j].position = self.particle[j].position + self.particle[j].velocity
                
                print("Old Cost: ", self.particle[j].cost)
                #calculate new particle cost
                self.particle[j].cost = self.cost_function_pso(nn, self.particle[j])
                print("New Cost: ", self.particle[j].cost)
                
                #update particle best 
                if self.particle[j].cost < self.particle[j].best_cost:
                    self.particle[j].best_position = self.particle[j].position
                    self.particle[j].best_cost = self.particle[j].cost
                    
                    #update global best
                    if self.particle[j].best_cost < self.global_best_cost:
                        self.global_best_position = self.particle[j].best_position
                        self.global_best_cost = self.particle[j].best_cost
             
            self.all_global_best[i] = self.global_best_cost
            self.w = self.w * self.wdamp
            print("Iteration Number: ", i+1, "Best Cost: ", self.all_global_best[i])