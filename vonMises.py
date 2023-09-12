from scipy.special import iv
import scipy.stats as stats
import numpy as np

class VonMisesFisher():
    def __init__(self,mu,kappa):
        self.eps = 1e-8
        self.p = len(mu)
        self.mu = np.array(mu)
        self.kappa = kappa
        if kappa == 0:
            self.normalization_const = 2/(np.pi**2)
        else:
            self.normalization_const = kappa**(self.p/2 - 1) / ((2 * np.pi)**(self.p/2) * iv(self.p/2 - 1, kappa))
        # if  kappa > eps:
        #     self.ln_normalization_const = (p/2 - 1) * math.log(kappa) - p/2*math.log(2*math.pi) - math.log(iv(p/2 - 1, kappa))
        # else:
        #     self.ln_normalization_const = float('-inf')

    def pdf(self,x):
        """
        Args:
            x : (..., p)
        Return:
            prob :(...)
        """
        assert x.shape[-1] == self.p
        return self.normalization_const * np.exp(self.kappa * (x @ self.mu[:,None]).squeeze(-1))

    def sample(self, size):
        x = np.empty((size,self.p))
        for i in range(len(x)):
            while True:
                # Step 1: Sample a point uniformly from the (p-1)-dimensional unit sphere
                z = np.random.normal(size=self.p)
                z /= np.linalg.norm(z)

                # Step 3: Accept the sample with probability proportional to the PDF
                if np.random.rand() < self.pdf(z) :
                    x[i] = z
                    break
        
        return x

    def update_parameter(self, x, weights):
        """
        Args:
            x : (N, p)
            weights : (N)
        """
        bar_x = x.T @ weights / np.sum(weights)
        bar_R = np.linalg.norm(bar_x)
        # bar_R = max(np.linalg.norm(bar_x),self.eps)
        self.mu = bar_x / bar_R
        self.kappa = bar_R*(self.p - bar_R**2 ) / (1 - bar_R**2 + self.eps)
        # self.kappa = max(bar_R*(self.p - bar_R**2 ) / (1 - bar_R**2 + self.eps),0)



if __name__ == "__main__":
    vmf = VonMisesFisher([1,0,0,0],0)
    x = vmf.sample(100)
    weights = vmf.pdf(x)
    vmf.update_parameter(x,weights)