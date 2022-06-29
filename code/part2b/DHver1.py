#initialization
import matplotlib.pyplot as plt
import numpy as np
import random
import math

# importing Qiskit
from qiskit import IBMQ, Aer, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy

# import basic plot tools
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import MCXGate



w = [2, 2, 1, 8, 8, 8] #12 13 23

def QFT (qc, start, end):
    total = end-start+1

    for i in range(total):
       qc.h(start+i)
       for k in range(1, total-i):
           theta = 1/np.power(2, k)
           theta = 2 * np.pi /np.power(2, k+1)

           qc.crz(theta, start+i+k, start+i)




def IQFT (qc, start, end):
    total = end-start+1

    for iQ in range(total):
        i = total - iQ - 1
        
        for kQ in range(1, total-i):
            k = total-i -kQ
            theta = -1/np.power(2, k)
            theta = -2 * np.pi /np.power(2, k+1)        

            qc.crz(theta, start+i+k, start+i)
        qc.h(start+i)    




def qADD (qc, starta,enda,startb,endb):
    total = endb - startb + 1
    for i in range (total):
        for k in range(1,total-i+1):

            theta = 2*np.pi/np.power(2,k)
            qc.crz(theta, starta+k+i-1,startb+i)



def CZ(qc, b, bn, a, an,w):
    left = an-bn
    b = b-left
    for i in range(an):
        for j in range(1, an-i+1):
            theta = 2*np.pi*w/np.power(2,j)
            if (i+j-1 >= left):
                qc.crz(theta, b+i+j-1, a+i)
            


def T(n):
    x=0
    a=0
    b=0
    if n==0:
        a=0
        b=0
    elif n==1:
        a=0
        b=1
    elif n==2:
        a=0
        b=2        
    elif n==3:
        a=0
        b=3
    elif n==4:
        a=1
        b=0
    elif n==5:
        a=1
        b=1
    elif n==6:
        a=1
        b=2
    elif n==7:
        a=1
        b=3
    elif n==8:
        a=2
        b=0
    elif n==9:
        a=2
        b=1
    elif n==10:
        a=2
        b=2    
    elif n==11:
        a=2
        b=3    
    elif n==12:
        a=3
        b=0    
    elif n==13:
        a=3
        b=1 
    elif n==14:
        a=3
        b=2 
    elif n==15:
        a=3
        b=3 

    ## distance
    #distance 0a
    if a==1 or a==2:
        x = x+w[0]
    elif a==3:
        x = x+2*w[0]    
    else:
        x = x

    #distance 0b
    if b==1 or b==2:
        x = x+w[1]
    elif b==3:
        x = x+2*w[1]    
    else:
        x = x
        

    # distanace ab
    if n==0 or n==5 or n==10 or n==15:
        x=x
    elif n==3 or n==6 or n==9 or n==12:
        x = x+2*w[2]
    else:
        x = x+w[2]

    
    # constraint
    if a==0:
        x = x + w[3]
    if b==0:
        x = x + w[4]
    if a==b:
        x = x + w[5]


    return x

#################################################




total_n = 2 + 2 + 2 + 2 + 2 + 5 + 5
d12 = 4
d13 = 6
d23 = 8
s12 = 10
s13 = 11
s23 = 12
wxs = 13
wx = 14
wx_n = 5
q1 = 19
or0 = 20
orm = 21




total_n = orm+1
#total_n = 10

lamb = 6/5

N = 16


y = 6


    
y = random.randint(0,N-1)
y = 0
Thr = T(y)


dhIteration = 0
while dhIteration < 22.5*np.sqrt(N):
    print("Threshold:", Thr)
    biy = bin(Thr)
    b_count = np.zeros(wx_n+1)
    bit=1
    while True:
        if (biy[-bit]=='b'):
            break
        b_count[wx_n +1- bit] = biy[-bit]
        bit = bit+1    
    
    
    
    m=1
    
    still = True
    while (still):
        qc = QuantumCircuit(total_n)
        #qc = QuantumCircuit(total_n,2)
        
        
        #################################
        
        
        # Intengle every possibility
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.h(3)
        
        itj = random.randint(0,np.floor(m))
        
        for iteration in range(itj):
        ####################################################block
            
            # Calculate distance
            # d12
            qc.ccx(0,1,d12)
            qc.cx(0,d12+1)
            qc.cx(1,d12+1)
            
            #d 13
            qc.ccx(2,3,d13)
            qc.cx(2,d13+1)
            qc.cx(3,d13+1)
            
            #d23
            qc.cx(0,2)
            qc.cx(1,3)
            
            qc.ccx(2,3,d23)
            qc.cx(2,d23+1)
            qc.cx(3,d23+1)
            
            qc.cx(0,2)
            qc.cx(1,3)
            
            
            
            # calculate constarint
            # 1=2
            qc.x(0)
            qc.x(1)
            qc.ccx(0,1,s12)
            qc.x(0)
            qc.x(1)
            
            # 1=3
            qc.x(2)
            qc.x(3)
            qc.ccx(2,3,s13)
            qc.x(2)
            qc.x(3)
            
            
            # 2=3
            qc.cx(0,2)
            qc.cx(1,3)
            
            qc.x(2)
            qc.x(3)
            qc.ccx(2,3,s23)
            qc.x(2)
            qc.x(3)
            
            qc.cx(0,2)
            qc.cx(1,3)
            
            
            
            
            
            ## Calcualte Objective
            QFT(qc, wx, wx+wx_n-1)
            #d12
            CZ(qc, d12, 2, wx, wx_n, w[0])   
            #d13
            CZ(qc, d13, 2, wx, wx_n, w[1])   
            #d23
            CZ(qc, d23, 2, wx, wx_n, w[2])   
            #c12
            CZ(qc, s12, 1, wx, wx_n, w[3])   
            #c13
            CZ(qc, s13, 1, wx, wx_n, w[4])   
            #c23
            CZ(qc, s23, 1, wx, wx_n, w[5])   
               
            IQFT(qc, wx, wx+wx_n-1)
            
            
            
            #qc.x(wx+3)
            
            
            ## Oracle
            # Calcualte -w
            for i in range(wx_n+1):
                qc.x(wxs+i)
            
            
            qc.x(q1)
            QFT(qc, wxs, wx+wx_n-1)
            # + 1
            CZ(qc, q1, 1, wxs, wx_n+1, 1)   
            
            
            # a+
            for i in range(wx_n+1):
                for j in range(wx_n+1 - i):
                    if b_count[i+j] == 1:
                        theta = 2*np.pi/np.power(2,j+1)
                        qc.rz(theta, wxs+i)
            
            
            
            IQFT(qc, wxs, wx+wx_n-1)
            
            ####################################################block
            gate = qc.to_gate()
            inverse_g = gate.inverse()
            
            
            ###
             # fx
            qc.x(wxs)
            
            
            qc.x(orm)
            qc.h(orm)
            qc.cx(wxs, orm)
            qc.x(wxs)
            
            
            
            
            ## Uncomputation
            qc.append(inverse_g, qc.qubits)
            qc.h(0)
            qc.h(1)
            qc.h(2)
            qc.h(3)
            
            qc.h(orm)
            qc.x(orm)
            
            
            
            
            ## difuser
            for i in range(4):
                qc.h(i)
                qc.x(i)
                
            
            qc.h(3)
            gate = MCXGate(3)
            qc.append(gate, [0, 1, 2, 3])
            qc.h(3)
            
            for i in range(4):
                qc.x(i)
                qc.h(i)
                 
                
            dhIteration = dhIteration+1
    #########################################            
        
        qc.save_statevector() # Save state after CNOT (also a final state)
        backend=Aer.get_backend('aer_simulator') 
        qc = transpile(qc, backend)
        result = backend.run(qc).result()
        state = result.get_statevector()
        probib = np.power(abs(state[0:N]), 2)
        probib = probib / np.linalg.norm(probib, ord=1)
        output = np.random.choice(np.arange(0, N), p=probib)

        
    
        '''
        sim = Aer.get_backend('aer_simulator')  # Tell Qiskit how to simulate our circui
        qc.save_statevector() # Save statevector
        #qobj = assemble(qc)
        #state = sim.run(qobj).result().get_statevector() # Execute the circuit
        qobj = assemble(qc)
        result = sim.run(qobj).result()        
        state = result.get_statevector()
        probib = np.power(abs(state[0:N]), 2)
        output = np.random.choice(np.arange(0, N), p=probib)
        '''    
        
        '''
        qc.barrier()
        qc.measure([0, 1], [0,1])

        simulator = Aer.get_backend('qasm_simulator')
        result = simulator.run(qc, shots=1, memory=True).result()
        output = result.get_memory(qc)
        '''
        
        if (T(output) <= Thr):
            still = False
        elif (lamb *m> np.sqrt(N)):
            still = False
        else:
            m = lamb*m
    if T(output) <= Thr:
        Thr = T(output)
            
'''
# #####################################
qc.draw(output = 'mpl')

import qiskit.quantum_info as qi
stv1 = qi.Statevector.from_instruction(qc)
stv1 = stv1.to_dict(decimals =3)
print(stv1)

'''


print("solution :", format(6, 'b').zfill(4))



