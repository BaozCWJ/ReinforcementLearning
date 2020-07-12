from RandomWalk import *
import numpy as np
import matplotlib.pyplot as plt


def error(result):
    return np.sqrt(sum(abs(result[1:6]-[1/6.0,2/6.0,3/6.0,4/6.0,5/6.0])**2)/5)

A = RandomWalk(3)
A_result = A.TD(100,0.05,1)
print(A_result)
print(error(A_result))

B = RandomWalk(3)
B_result = B.MC(100,0.03,1)
print(B_result)
print(error(B_result))


episode_list = [0,10,20,30,40,50,60,70,80,90,100,150,200,250,300]
TDalpha = [0.15,0.1,0.05]
MCalpha = [0.03,0.02,0.01]
TDresult = np.zeros((3,len(episode_list)))
MCresult = np.zeros((3,len(episode_list)))

N=1000
for kk in range(N):
    for jj in range(3):
        for ii in range(len(episode_list)):
            episode = episode_list[ii]
            alphaA = TDalpha[jj]
            alphaB = MCalpha[jj]

            A = RandomWalk(3)
            Aresult = A.TD(episode,alphaA,1)
            Aerror = error(Aresult)
            TDresult[jj,ii]+=Aerror

            B = RandomWalk(3)
            Bresult = B.MC(episode,alphaB,1)
            Berror = error(Bresult)
            MCresult[jj,ii]+=Berror

TDresult=TDresult/N
MCresult=MCresult/N

td15,=plt.plot(episode_list,TDresult[0,:],color='b',linestyle='-')
mc03,=plt.plot(episode_list,MCresult[0,:],color='r',linestyle='-')
td1,=plt.plot(episode_list,TDresult[1,:],color='b',linestyle='dashed')
mc02,=plt.plot(episode_list,MCresult[1,:],color='r',linestyle='dashed')
td05,=plt.plot(episode_list,TDresult[2,:],color='b',linestyle='-.')
mc01,=plt.plot(episode_list,MCresult[2,:],color='r',linestyle='-.')
#plt.legend(handles = [td,mc], labels =['TD','MC'])
plt.legend(handles = [td15,td1,td05,mc03,mc02,mc01], labels =['TD 0.15','TD 0.10','TD 0.05','MC 0.03','MC 0.02','MC 0.01'])
plt.xlabel('Episode')
plt.ylabel('RMS error')
plt.title('Emprical RMS Error')
plt.show()
