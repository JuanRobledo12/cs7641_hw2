import numpy as np
from tqdm import tqdm
from kmeans import KMeans


SIGMA_CONST = 1e-6
LOG_CONST = 1e-32

FULL_MATRIX = True # Set False if the covariance matrix is a diagonal matrix

class GMM(object):
    def __init__(self, X, K, max_iters=100):  # No need to change
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters

        self.N = self.points.shape[0]  # number of observations
        self.D = self.points.shape[1]  # number of features
        self.K = K  # number of components/clusters

    # Helper function for you to implement
    def softmax(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        Hint:
            Add keepdims=True in your np.sum() function to avoid broadcast error. 
        """
        #print(logit)

        #TRY TO SUM THE MAX ROW VALUE AFTER THE EXP CALCULATION AND BEFORE LOG

        max_row_val = np.amax(logit, axis=1, keepdims=True)
        #print(max_row_val)
        logit = logit - max_row_val
        #print('substact max_val to logit: ', logit)
        logit_exp = np.exp(logit)
        #print('exp of logit: ', logit_exp)
        logit_exp_sum = np.sum(logit_exp, axis=1, keepdims=True)
        #print('sum of logit exp across D: ', logit_exp_sum)
        prob = logit_exp / logit_exp_sum
        #print('softmax prob: ', prob)
        #print('shapes of logit and prob', logit.shape, prob.shape)

        return prob

    def logsumexp(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        Hint:
            The keepdims parameter could be handy
        """
        #print('logit: ', logit)
        max_row_val = np.amax(logit, axis=1, keepdims=True)
        #print('max_row_val', max_row_val)
        logit = logit - max_row_val
        #print('sum max val to logit: ', logit)
        logit_exp = np.exp(logit)
        #print('logit_exp: ', logit_exp)
        logit_exp_sum = np.sum(logit_exp, axis=1, keepdims=True)
        #print('logit_exp sum', logit_exp_sum)
        s = np.log(logit_exp_sum) + max_row_val
        #print('natural log: ', s)
        return s

    # for undergraduate student
    def normalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """

        raise NotImplementedError

    # for grad students
    def multinormalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            1. np.linalg.det() and np.linalg.inv() should be handy.
            2. The value in self.D may be outdated and not correspond to the current dataset,
            try using another method involving the current arguments to get the value of D
        """

        #print('points: ', points)
        #print('mu_i', mu_i)
        #print('sigma_i', sigma_i)
        
        try:
            sigma_inv = np.linalg.inv(sigma_i)
        except:
            sigma_inv = np.linalg.inv(sigma_i + SIGMA_CONST)
        
        exp_term_1 = (points - mu_i) @ sigma_inv
        exp_term_2 = np.transpose(points - mu_i)

        #print('exp_term_1: ', exp_term_1)
        #print('exp_term_2: ', exp_term_2)

        exp_term = np.transpose(exp_term_1) * exp_term_2

        #print('terms mult: ', exp_term)
        final_exp_term = np.exp(-0.5*np.sum(exp_term, axis=0))
        #print('final_exp_term: ', final_exp_term)

        division_term = 1 / ((2*np.pi)**(points.shape[1]/2))
        sigma_det_term = (1 / np.sqrt(np.linalg.det(sigma_i)))
        #print('division_term: ', division_term)
        #print('sigma_det_term: ', sigma_det_term)
        normal_pdf = division_term * sigma_det_term * final_exp_term
        return normal_pdf


    def _init_components(self, **kwargs):  # [5pts]

        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case

            Hint: np.random.seed(5) may be used at the start of this function to ensure consistent outputs.
        """
        np.random.seed(5) #Do Not Remove Seed

        N = self.points.shape[0]
        D = self.points.shape[1]
        #print('(N,D,K): ', N, D, self.K)
        pi = np.ones(self.K) * (1/self.K)
        mu = self.points[np.random.choice(N, size=self.K, replace=False)]
        sigma = np.zeros((self.K, D, D))

        for k in range(0, self.K):
            np.fill_diagonal(sigma[k], 1)
        
        #print(mu)
        #print(sigma)
        #print(pi)

        return pi, mu, sigma

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):  # [10 pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.

        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """

        # === graduate implementation
        #if full_matrix is True:
            #...
        #print('datapoints shape: ', self.points.shape)
        pdf_arr = np.ones((self.K, self.points.shape[0]))
        for k_i in range(self.K):
            mu_i = mu[k_i]
            #print('mu_i', mu_i)
            #print(mu_i.shape)
            sigma_i = sigma[k_i]
            #print('sigma_i', sigma_i)
            #print(sigma_i.shape)
            pdf_arr[k_i] = self.multinormalPDF(self.points, mu_i, sigma_i)
        #print('pdf_arr: ', pdf_arr)
        #print(pdf_arr.shape)
        log_pdf_arr = np.log(pdf_arr + LOG_CONST)
        log_pi_arr = np.log(pi + LOG_CONST)

        #Log Likelihood
        ll = log_pi_arr + np.transpose(log_pdf_arr)
        #print('log likelihood array: ', ll)
        #print(ll.shape)

        # === undergraduate implementation
        #if full_matrix is False:
            # ...
        return ll

    def _E_step(self, pi, mu, sigma, full_matrix = FULL_MATRIX , **kwargs):  # [5pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """

        #I have no idea how this works I guess it is because you are optimizing r?

        # === graduate implementation
        #if full_matrix is True:
            # ...
        
        log_likeli = self._ll_joint(pi, mu, sigma, True) 
        gamma = self.softmax(log_likeli)
        # === undergraduate implementation
        #if full_matrix is False:
            # ...

        return gamma

    def _M_step(self, gamma, full_matrix=FULL_MATRIX, **kwargs):  # [10pts]
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
            Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
        """
        # === graduate implementation
        #if full_matrix is True:
            # ...
        # print('gamma: ', gamma)
        N_arr = np.sum(gamma, axis=0)
        #print('N_arr', N_arr)
        #print('datapoints', self.points)
        #print('-------------------------')

        n_D = self.points.shape[1]  #Dimensions in the dataset
        new_mu = np.ones((self.K, n_D))
        new_sigma = np.ones((self.K, n_D, n_D))
        for k_i in range(self.K):
            #New Mean Calculation
            #print('K_i: ', k_i)
            gamma_k_i = gamma[:,k_i].reshape(gamma.shape[0], 1)
            #print('gamma_ki: ', gamma_k_i)
            num_new_mu = gamma_k_i * self.points
            new_mu[k_i] = np.sum(num_new_mu, axis=0) / N_arr[k_i]
            #print('new_mu_ku', new_mu[k_i])

            #New Covariance Matrix Calculation
            point_mean_subs = self.points - new_mu[k_i]
            #print('point_mean sub:', point_mean_subs)
            #print(point_mean_subs.shape)
            gamma_points_mean = np.transpose(gamma_k_i * point_mean_subs)
            #print('gamma_points_mean: ', gamma_points_mean)
            #print(gamma_points_mean.shape)
            sigma_mult = gamma_points_mean@point_mean_subs
            #print('sigma_mult: ', sigma_mult)
            new_sigma[k_i] = sigma_mult / N_arr[k_i]
            #print('new_sigma', new_sigma[k_i])
            #print('-------------------------')
        new_pi = N_arr / self.points.shape[0]
        #print('new_mu', new_mu)
        #print('------------------')
        #print('new_sigma', new_sigma)
        #print('new_pi', new_pi)

        #Either use a for loop or add a new axis to divide K.
        # === undergraduate implementation
        #if full_matrix is False:
            # ...

        return new_pi, new_mu, new_sigma

    def __call__(self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs):  # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma, full_matrix)

            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)

