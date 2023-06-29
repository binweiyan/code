class RBFLassoRegression:

    def __init__(self, rbf_dim=1, alpha=1.0, sigma=1.0, times = 1):
        """Kernel lasso regression using random Binning features.

        rbf_dim : Dimension of random feature.
        alpha :   Regularization strength. Should be a positive float.
        hash_ :   Hash function to use. Should be a function that takes a
                    vector of length D and returns a value in [0, rbf_dim).
        d_ :      p elements, each row of d_'s element represents a random vector of length D, generated from a gamma distribution
        U_ :      p elements, each row of U_'s element represents a random vector of length D, generated from a uniform distribution in [0, d] where d is the corresponding element of d_
        p :       number of times to repeat the random binning
        """
        self.fitted  = False
        self.rbf_dim = rbf_dim
        self.sigma   = sigma
        self.lm      = Lasso(alpha=alpha)
        self.d_      = None
        self.U_      = None
        self.p       = times

    def fit(self, X, y):
        """Fit model with training data X and target y.
        """
        Z, U, d = self._get_rbfs(X, return_vars=True)
        self.lm.fit(Z.T, y)
        self.U_ = U
        self.d_ = d
        self.fitted = True
        return self

    def predict(self, X):
        """Predict using fitted model and testing data X.
        """
        if not self.fitted:
            msg = "Call 'fit' with appropriate arguments first."
            raise NotFittedError(msg)
        Z = self._get_rbfs(X, return_vars=False)
        return self.lm.predict(Z.T)

    def _get_rbfs(self, X, return_vars):
        """Return random Fourier features based on data X, as well as random
        variables W and b.
        """

        N, D = X.shape
        if self.U_ is not None:
            U, d = self.U_, self.d_
        else:
            for i in range(self.p):
                dd = np.random.gamma(2, self.sigma, size=(self.rbf_dim, D))
                UU = np.random.uniform(0, d)
                U.append(UU)
                d.append(dd)

        for i in range(self.p):
            #generate the random binning features
            B  = np.repeat(U[i][:, np.newaxis], N, axis=1)
            m  = np.round((X.T - B)/d[i])
            #hash the random binning features from N * D to N * rbf_dim
            h = np.mod(np.sum(m, axis = 1), self.rbf_dim)
            #change h to one-hot encoding
            Z = np.zeros((self.rbf_dim, N))
            Z[h, np.arange(N)] = 1
            Z = Z.T
            #generate the random binning features
            if i == 0:
                Z_ = Z
            else:
                Z_ = np.hstack((Z_, Z))
        Z_ = Z_ / np.sqrt(self.p)
        if return_vars:
            return Z_, U, d