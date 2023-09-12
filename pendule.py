from cmath import cos
import numpy as np
import matplotlib.pyplot as plt
import euler as e
import heun as he
import method as m

from method import disp_2D, meth_epsilon

'''
theta'' = -w0^2 sin(theta)
(theta, theta') -> (theta', theta'')
F(t; (a,b)) = (b, -w0^2 sin(theta))
'''
g = 9.8
L = 1
def f_1pendule(y, t):
    res = np.zeros(2)
    res[0] = y[1]
    res[1] = (-1)*(g/L)*y[0]
    return res



def disp_f_pendule(a, b, N):
    X = np.linspace(a, b, N)
    Y = np.zeros(N)
    for i in range(N):
        Y[i] = f_1pendule(X[i])
    plt.plot(X, Y)
    plt.show()


def get_freq(theta0, h, t0, tf):
    yN = [theta0, 0]
    tN = t0
    ynplus1 = e.step_runge_kutta4(yN, tN, h, f_1pendule)
    tnplus1 = tN + h
    t_start = 0
    periods = []
    N = int((tf - t0)/h)
    for i in range(N):
        if (yN[0] >= 0 and ynplus1[0] <= 0):
            if (t_start == 0):
                t_start = tN
            else:
                periods.append(tN - t_start)
                t_start = tN
        yN = ynplus1
        tN = tnplus1
        ynplus1 = e.step_runge_kutta4(ynplus1, tnplus1, h, f_1pendule)
        tnplus1 = tnplus1 + h

        
    print("nombre de valeurs de périodes trouvées:", len(periods))
    T = sum(periods)/len(periods)
    print("periode moyenne trouvée :", T)
    print("fréquence pour theta0 =", theta0, ":", (2*np.pi)/T)
    print("on a avec g=", g, "L=", L, "sqr(g/L) =", np.sqrt(g/L))
    plt.show()

get_freq(0.1, 0.01, 0, 10)

#pendule à 2 maillons
#θ1'' =  −g (2 m1 + m2) sin θ1 − m2 g sin(θ1 − 2 θ2) − 2 sin(θ1 − θ2) m2 (θ2'2 L2 + θ1'2 L1 cos(θ1 − θ2)) /
#L1 (2 m1 + m2 − m2 cos(2 θ1 − 2 θ2))


#θ2'' = 	2 sin(θ1 − θ2) (θ1'2 L1 (m1 + m2) + g(m1 + m2) cos θ1 + θ2'2 L2 m2 cos(θ1 − θ2))
# / L2 (2 m1 + m2 − m2 cos(2 θ1 − 2 θ2))

def f_2pendules(y, t): #y = [θ1, θ2, θ1', θ2'] 
    g = 9.8
    L = 1
    m = 1
    res = np.zeros(4) #[θ1', θ2', θ1'', θ2'']
    res[0] = y[2]
    res[1] = y[3]
    numerator1 = (-1)*g*3*m*np.sin(y[0]) - m*g*np.sin(y[0] - 2*y[1]) - 2*np.sin(y[0] - y[1])*m*(y[3]*y[3]*L + y[2]*y[2]*L*np.cos(y[0] - y[1]))
    denominator = L*(3*m - m*np.cos(2*y[0] - 2*y[1]))
    numerator2 = 2*np.sin(y[0] - y[1])*(y[2]*y[2]*L*2*m + g*2*m*np.cos(y[0]) + y[3]*y[3]*L*m*np.cos(y[0] - y[1]))
    res[2] = numerator1 / denominator
    res[3] = numerator2 / denominator
    return res

#[np.pi/2, np.pi/2, 0, 0]
# x1 = L1 sin θ1

#y1 = −L1 cos θ1

#x2 = x1 + L2 sin θ2

#y2 = y1 − L2 cos θ2
''' 
y2eps, t2eps = m.meth_epsilon(np.array([0.55, 0, 0, 0]), 0, 15, 0.01, f_2pendules, e.step_runge_kutta4)

print(y2eps)

N = len(y2eps)
print(N)
X1 = np.zeros(N)
Y1 = np.zeros(N)
theta1 = np.zeros(N)
theta2 = np.zeros(N)
X2 = np.zeros(N)
Y2 = np.zeros(N)
L=1
g=9.81
for i in range(N):
    th1 = y2eps[i][0]%(2*3.14)
    th2 = y2eps[i][1]%(2*3.14)
    if (th1 > 3.14):
        theta1[i] =  th1 - 2*3.14
    else:
        theta1[i] =  th1
    if (th2 > 3.14):
        theta2[i] =  th2 - 2*3.14
    else:
        theta2[i] =  th2
    X1[i] = L*np.sin(y2eps[i][0])
    Y1[i] = (-1)*L*np.cos(y2eps[i][0])
    X2[i] = X1[i] + L*np.sin(y2eps[i][1])
    Y2[i] = Y1[i] - L*np.cos(y2eps[i][1])


plt.plot(X1, Y1)
plt.plot(X2, Y2)
plt.show()
plt.scatter(theta1, theta2)
plt.show()

 '''### Temps de premier retournement ###

def get_angle(theta):
    angle = theta%(2*np.pi)
    if (angle > np.pi):
        return angle - 2*np.pi
    else:
        return angle

def get_time(theta1, theta2):
    print("Calculating time for (", theta1, theta2, ")")
    y, t = m.meth_epsilon([theta1, theta2, 0, 0], 0, 10, 0.1, f_2pendules, e.step_runge_kutta4)
    h = t[1] - t[0]
    t = 0
    for i in range(len(y)- 1):
        t+=h
        if (y[i][1]%(2*np.pi) <= np.pi and y[i+1][1]%(2*np.pi) >= np.pi) or (y[i][1]%(2*np.pi) >= np.pi and y[i+1][1]%(2*np.pi) <= np.pi):
            print("time found : ", t)
            return t
    print("time found : infinite",)
    return 10**10

def print_pendule(yN):
    N = len(yN)
    print("N = ", N)
    X2 = np.zeros(N)
    Y2 = np.zeros(N)
    L=1
    for i in range(N):
        X2[i] = L*np.sin(yN[i][0]) + L*np.sin(yN[i][1])
        Y2[i] = (-1)*L*np.cos(yN[i][0]) - L*np.cos(yN[i][1])
    plt.plot(X2, Y2)
    plt.show()

def get_time_fast(theta1, theta2, meth):
    print("Calculating time for (", theta1, theta2, ")")
    y0 = [theta1, theta2, 0, 0]
    interval = 10
    N = 10000
    h = interval/N
    yN = [y0]
    tN = [0]
    for i in range(N):
        ynplus1 = meth(yN[-1], tN[-1], h, f_2pendules)
        tnplus1 = tN[-1] + h
        if (yN[-1][1]%(2*np.pi) <= np.pi and ynplus1[1]%(2*np.pi) >= np.pi) or (yN[-1][1]%(2*np.pi) >= np.pi and ynplus1[1]%(2*np.pi) <= np.pi):
            yN.append(ynplus1)
            tN.append(tnplus1)
            print("time found : ", tnplus1)
            #print_pendule(yN)
            return tnplus1
        else:    
            yN.append(ynplus1)
            tN.append(tnplus1)
    print("time found : infinite",)
    return 10**10

def get_time_h(theta1, theta2, t0, tf, h):
    #print("Calculating time for (", theta1, theta2, ")")
    yN = [theta1, theta2, 0, 0]
    tN = t0
    ynplus1 = e.step_runge_kutta4(yN, tN, h, f_2pendules)
    tnplus1 = tN + h
    while not (get_angle(yN[1]) <= (np.pi/2) and get_angle(ynplus1[1]) >= (np.pi/2)) or (get_angle(yN[1]) >= (np.pi/2) and get_angle(ynplus1[1]) <= (np.pi/2)):
        if (tnplus1 >= tf):
     #       print("time found : infinite")
            return tf + 1
        yN = ynplus1
        tN = tnplus1
        ynplus1 = e.step_runge_kutta4(ynplus1, tnplus1, h, f_2pendules)
        tnplus1 = tnplus1 + h
    #print("time found : ", (tN + tnplus1)/2)       
    return (tN + tnplus1)/2


size_graph = 10
tf = 10
graph = np.zeros((size_graph, size_graph))
theta1 = np.linspace(-np.pi, np.pi, size_graph)
theta2 = np.linspace(-np.pi, np.pi, size_graph)

for i in range(size_graph):
    for j in range(size_graph):
        graph[i, j] = get_time_h(theta1[i], theta2[j], 0, tf, 0.01)

plt.imshow(graph, cmap="ocean")
plt.savefig("graph_retournement.pdf", format="pdf")
plt.show()