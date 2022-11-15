import numpy as np


class HMM(object):
    def __init__(self, N ,M, pi, A ,B) -> None:
        self.N = N 
        self.M = M
        self.pi = pi
        self.A = A 
        self.B = B 

    def gen_data_from_distribution(self, dist):
        r = np.random.rand()

        for i, p in enumerate(dist):
            if r < p: return i
            r-=p
        
    def generate(self, T:int):
        z = self.gen_data_from_distribution(self.pi)
        o = self.gen_data_from_distribution(self.B[z])
        result = [o]
        for _ in range(T-1):
            z = self.gen_data_from_distribution(self.A[z])
            o = self.gen_data_from_distribution(self.B[z])
            result.append(o)
        return result

    def evaluate(self, X):
        alpha = self.pi * self.B[:, X[0]]
        print(alpha)

        for x in X[1:]:
            # alpha_next = np.empty(self.N)
            # for j in range(self.N):
            #     alpha_next[j] = np.sum(alpha * self.A[:, j] * self.B[j, x])
            # alpha = alpha_next

            alpha = np.sum(self.A * self.B[:, x].reshape(1, -1)* alpha.reshape(-1, 1), axis=0)
            print(alpha, alpha.shape)
        return alpha.sum()
            


if __name__ == '__main__':
    pi = np.array([0.2, 0.4, 0.4])
    A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])

    N = len(pi)
    M = B.shape[-1]

    hmm = HMM(N, M, pi, A, B)

    # print(hmm.generate(10))
    print(hmm.evaluate([0,1,0]))
