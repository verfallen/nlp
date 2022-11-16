import numpy as np


class HMM(object):
    def __init__(self, N, M, pi, A, B) -> None:
        self.N = N
        self.M = M
        self.pi = pi
        self.A = A
        self.B = B

    def gen_data_from_distribution(self, dist):
        r = np.random.rand()

        for i, p in enumerate(dist):
            if r < p:
                return i
            r -= p

    def generate(self, T: int):
        z = self.gen_data_from_distribution(self.pi)
        o = self.gen_data_from_distribution(self.B[z])
        result = [o]
        for _ in range(T - 1):
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

            alpha = np.sum(
                self.A * self.B[:, x].reshape(1, -1) * alpha.reshape(-1, 1), axis=0
            )
        return alpha.sum()

    def evaluate_backward(self, X):
        beta = np.ones(self.N)

        for x in X[:0:-1]:
            beta_next = np.empty(self.N)
            for i in range(self.N):
                beta_next[i] = np.sum(self.A[i, :] * self.B[:, x] * beta)

            beta = beta_next
        return np.sum(beta * pi * self.B[:, X[0]])

    def encode(self, X):
        T = len(X)
        x = X[0]
        delta = self.pi * self.B[:, x]
        varphi = np.zeros((T, self.N), dtype=int)
        path = [0] * T

        for i in range(1, T):
            delta = delta.reshape(-1, 1)
            tmp = delta * self.A
            varphi[i, :] = np.argmax(tmp, axis=0)
            delta = np.max(tmp, axis=0) * self.B[:, X[i]]

        path[-1] = np.argmax(delta)

        for i in range(T - 1, 0, -1):
            path[i - 1] = varphi[i, path[i]]

        return path


if __name__ == "__main__":
    pi = np.array([0.2, 0.4, 0.4])
    A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])

    N = len(pi)
    M = B.shape[-1]

    hmm = HMM(N, M, pi, A, B)

    X = [0, 1, 0]
    # print(hmm.generate(10))
    # print(hmm.evaluate([0,1,0]))
    # print(hmm.evaluate_backward([0, 1, 0]))
    print(hmm.encode(X))
