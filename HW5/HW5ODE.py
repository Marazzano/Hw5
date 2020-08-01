# hw 5: euler's method, revised example code

import numpy as np
import matplotlib.pyplot as plt


def calculate_pop(r, y0, t0, t1, h=.5, cond=True):
    """ if a = True then we use RK4 if false we use euler  """
    a = t0
    b = t1
    if cond==True:
        approx=RK4(pop_func,a,b,y0,h)
    if cond==False:
        approx=euler(pop_func, a, b, y0,h)
    return approx
    
    

def euler(f, a, b, y0, h=.5):
    """ Forward Euler with fixed step size using a while loop."""
    y = y0
    t = a
    yvals = [y]
    tvals = [t]
    #pop(r,y)
    if a < b: 
        while t < b - 1e-12:
            y += h*f(t, y)
            t += h
            yvals.append(y)
            tvals.append(t)  
    if a > b: 
         while t > b - 1e-12:
            y += h*f(t, y)
            t -= h
            yvals.append(y)
            tvals.append(t)
    
    return tvals, yvals

def RK4(f, a, b, y0, h=.5):
    u=y0
    t = a
    uvals = [u]
    tvals = [t]
    
    if a < b: 
        while t < b - 1e-12:
            t += h
            f1 = f(t,u)
            f2 = f(t+h/2, u+f1/2)
            f3 = f(t+h/2, u+f2/2)
            f4 = f(t+h, h*f3)
            u = u + (h/6)*(f1 + 2*f2 + 2*f3 + f4)
            uvals.append(u)
            tvals.append(t)
    if a > b: 
        while t > b - 1e-12:
            t -= h
            f1 = f(t,u)
            f2 = f(t+h/2, u+f1/2)
            f3 = f(t+h/2, u+f2/2)
            f4 = f(t+h, h*f3)
            u = u + (h/6)*(f1 + 2*f2 + 2*f3 + f4)
            uvals.append(u)
            tvals.append(t)
        
    return tvals, uvals
    
        
    
    
def exactsol(t):
    return ((1 + np.exp(-t))**(-1))

    
def pop_func(t,y,r=1):
    return r*y*(1-y)

def efunc(t, y):
    """ example ODE function """
    return 2*t*y


#-----------------------------------------------------------------
# example code [adapted from lecture] for euler's method for reference
def euler_plot():
    """ example: solve and plot solution with Euler's method """
    h = 0.5
    r =1 
    y0= .5
    a= 0
    b=5
    s = np.linspace(0, 5, 10)
    exact = exactsol(s)
   
    t, y = calculate_pop(r, y0, a, b, h, cond=True)
    p, z = calculate_pop(r, y0, a, b, h, cond=False)
   
    plt.figure(figsize=(3, 2.5))
    plt.plot(s, exact, '-k')  # plot exact
    plt.plot(t, y, '--.b', markersize=10, label='RK4')  # plot approx.
    plt.plot(p, z, '--.r', markersize=10, label='Euler')
    plt.legend(loc= "upper left")
    plt.xlabel('t')
    plt.ylabel('y')
    # plt.savefig('name.pdf', bbox_inches='tight')
    plt.show()

def euler_conv():
    """ calculate the error at t=b for various h's and plot """
    m = 9
    hvals = [(0.25)*2**(-k) for k in range(m)]
    y0 = .5
    b = 5

def error(r, y0, t0, t1, h=.5, cond=True):
    """Question 1.d"""
    t,y = calculate_pop(r, y0, t0, t1, h, cond=True)
    exact = [exactsol(i) for i in t]
    error = [abs(y[i] - exact[i]) for i in range(len(t))]
    return max(error)

if __name__ == "__main__":
    euler_plot()
    err_list= [error(1,.5,0,1,2**(-k)) for k in range(1,11)]
    hvals = [2**(-k) for k in range(1,11)]
    plt.loglog(hvals, err_list, '.--k')
    
    
    
    
    