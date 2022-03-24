import numpy as np

def gradient_descent(x,y):
    m_curr = b_curr =0
    iterations = 1000
    n = len(x)
    lr = 0.0001
    for i in range(iterations):
        y_pred =  m_curr*x + b_curr
        md = -(2/n)*sum(x *(y-y_pred))
        bd = -(2/n)*sum(y-y_pred)

        m_curr = m_curr- lr * md 
        b_curr = b_curr - lr *bd
        print('m : ',m_curr,'b: ',b_curr, 'iteration: ',i)
    return y_pred

x = np.array([1,2,3,4,5])
y = np.array([5,6,7,9,13])

y_pred = gradient_descent(x,y)

acc = (1/len(x))*sum(y-y_pred)
print('Accuracy: ',acc*100)