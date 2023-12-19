import matplotlib.pyplot as plt

def visualize(X,Y,i,loss):
    plt.figure(figsize=(10, 6))
    plt.plot(X, label='Energy Predictions')
    plt.plot(Y, label='Energy Labels')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Estimated Energy vs Labels'+'\n'+'lambda: '+''+str((i+1)*0.05)+'MAE: '+str(loss))
    plt.legend()
    plt.savefig('/root/WorkSpace/project/spectrum_two_stage/results/test_{}.png'.format(i*400), format='png')