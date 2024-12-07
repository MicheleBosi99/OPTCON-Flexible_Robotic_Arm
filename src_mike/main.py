import numpy as np
from dynamics import natural_evolution as nat_evo 
from visualizer import animate_double_pendulum as anim

def main ():

    # Definiamo i punti di equilibrio desiderati:
    # Partenza
    T1_0 = 0
    T2_0 = np.pi

    # Arrivo
    T1_1 = np.pi
    T2_1 = np.pi

    # Definiamo uno stato di equilibrio iniziale
    x0 = np.zeros([1, 4])
    u0 = np.zeros([1, 4])

    x0[0,:] = [0,      0,   T1_0,  T2_0,]
    u0[0,:] = [0.2,    0,      0,     0,] 

    # Costruiamo una prima evoluzione naturale del sistema
    xe =  nat_evo(x0, u0)

    ###### Task 1
    # Compute two equilibria for your system and define a reference curve between the two.
    # Compute the optimal transition to move from one equilibrium to another exploiting the
    # Newton’s-like algorithm (in closed-loop version) for optimal control.
    # Hint: you can exploit any numerical root-finding routine to compute the equilibria.
    # Hint: define two long constant parts between the two equilibria with a transition in between.
    # Try to keep everything as symmetric as possible, 

    


    #Visualizziamo
    anim(xe, 10)

if __name__ == "__main__":
    main()