# A collection of custom functions and classes used in the
# Hospital Staffing tutorial.
#
import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
from matplotlib.colors import CSS4_COLORS

######################
### Custom classes ###
class Gaussian:
    """Generates a Gaussian distribution with mean and 
    standard deviation 'mu' and 'sigma'.

    methods:
        pdf: Generates the probability density function (PDF)
        G(x_s) for the discrete set of points x_s : np.array.
            returns a numpy array.

        cdf: Generates the cumulative distribution function
        (CDF) for the discrete set of points x_s : np.array.
            returns a numpy array.

        sf: Generates the survival function (SF) for the
        discrete set of points x_s : np.array.
        sf(x_s) = 1 - cdf(x_s)
            returns a numpy array.

        auc: Generates the area under curve (AUC) in the interval
        x_left < x < x_right.
    """
    def __init__(self, mu=0.0, sigma=1.0):
        # Initiates the class
        self.mu = mu
        self.sigma = sigma
        self.variance = np.square(self.sigma)
        self.normalization = self.sigma*np.sqrt(2*np.pi)

    @property
    def mean(self):
        # Get the mean of the Gaussian, mu
        return self.mu

    @property
    def std(self):
        # Get the standard deviation of the Gaussian, sigma
        return self.sigma
        
    @mean.setter
    def set_mean(self, value):
        # Change the mean of the Gaussian
        self.mu = value

    @std.setter
    def set_std(self, value):
        # Change the standard deviation of the Gaussian
        if not value>0:
            raise ValueError('Standard deviation must be positive.')
        self.sigma = value
        self.variance = np.square(self.sigma)
        self.normalization = self.sigma*np.sqrt(2*np.pi)
    
        
    def pdf(self, x_s):
        # Compute the Probability Density Function [G(x), for x in x_s]
        return np.exp(-np.square(x_s - self.mu) / (2*self.variance)) / self.normalization
        
    def cdf(self, x_s):
        # Compute the Cumulitive Distribution Function 
        # [integrate(G(z), -inf, x) for x in x_s]
        # ingegrate(G(z), -inf, x) = erf(sqrt(2)*(x-mu)/sigma)
        return (1 + sp.special.erf((np.sqrt(2)/self.sigma)*(x_s - self.mu)))/2
        
    def sf(self, x_s):
        # Compute the Survival Function sf(x) = 1-cdf(x)
        return 1 - self.cdf(x_s)
        
    def auc(self, x_left=-4.0, x_right=4.0):
        # Compute the Area Under the Curve integrate(G(z), x_left, x_right)
        return self.cdf(x_right) - self.cdf(x_left)
        
    def expectation_value(self, v, x_s):
        # Evaluate the expectation value of v(x) over the discrete set of
        # points x_s:
        # <v>_(x_s) = Sum(v(x_i)*pdf(x_i)*(x_(i+1)-x_(i)), i in range(len(x_s)-1))
        if v.shape[-1] != x_s.shape[0]:
            raise ValueError('Invalid shape for input vector')
        delta_x_s = np.concatenate([x_s[1:] - x_s[:-1], [0.]])
        return np.sum((v*self.pdf(x_s))*delta_x_s, axis=-1)

    def sample(self, size):
        # Draw a random sample of size=size from the Gaussian distribution
        return np.random.normal(self.mu, self.sigma, size)


### Custom classes ###
######################


######################
## Custom functions ##

def df_mask(df: pd.DataFrame, cond: dict, return_index=False):
    '''Generates a pandas series mask for the df dataframe 
    using the conditions in the cond dictionary,

    Inputs:
        df: A pandas dataframe.
        cond: A dictionary of the form {column: value}
        return_index: Bool, if true, the function returns
        the index of rows satisfying cond.

    Returns:
        If return_index is False, returns a bool valued pandas 
        series sharing the index of df.
        If return_index is True, returns a pandas index object 
        with the indices of True values.
    '''
    mask = np.logical_and.reduce([df[key]==cond[key] for key in cond.keys()])
    mask = pd.Series(mask, index=df.index)
    if return_index:
        mask = df[mask].index
    return mask


def random_colors(keys: np.array, seed=67):
    '''Generates a random set of colors from CSS4_COLORS
    which can be used for plotting with matplotlib.

    Inputs:
        keys: A numpy array which will be used as the keys
        of the output dictionary.
        seed: An integer used as the seed for the random choice.

    Returns:
        A dictionary of key:color with randomly selected colors.
    '''
    np.random.seed(seed)
    colors = np.random.choice(list(CSS4_COLORS.keys()), keys.shape[0])
    return {k:c for k,c in zip(keys, colors)}


def bucketize(values: np.array, num_buckets: int) -> np.array:
    '''Bucketizes a 1D numpy array by splitting the data range,
    range(min(values), max(values)), into uniform buckets and 
    counting the numper of data points for each bucket.

    Inputs:
        values: A 1D numpy array of float values.

        num_buckets: An integet setting the number of buckets.

    Returns:
        A tuple of 1D numpy arrays (buckets, multiplicities) of shapes
        output_shapes = ((bucket_size), (bucket_size)).
    '''
    x = np.sort(values)
    num_samples = x.shape[0]
    bucket_size = (x[-1] - x[0])/num_buckets
    buckets = np.arange(x[0], x[-1]+bucket_size, bucket_size)[:num_buckets]
    
    multiplicities = []
    i = 0
    j = 1
    while j < num_buckets:
        m = 0
        while i < num_samples-1 and x[i] < buckets[j]:
            m += 1
            i += 1
        multiplicities.append(m)
        j += 1
    m = np.sum(x >= buckets[-1])
    multiplicities.append(m)
    return buckets, multiplicities


def r2_score(y_true, y_pred):
    '''Computes the R-squared score for predictions, y_pred, compared
    to the true values y_true.
    R2 = 1 - sum((y_true - y_pred)**2) / sum((y_true - <y_true>)**2)
    '''
    sum_squares_residuals = np.sum(np.square(y_true - y_pred))
    sum_squares = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - sum_squares_residuals / sum_squares


def fit_gaussian(x, best_fit=False, num_buckets=100, num_steps = 1000, learning_rate=0.1):
    ''' Fit a Gaussian distribution to the 1D data x. If best_fit is False, 
    the function returns a Gaussian class with mean and standard deviation 
    matching those of the data x.
    If best_fit is True, the function seeks to maximize the overlap of two 
    distributions using the cost_fn = Sum( (g - pdf(x))**2 ).

    Inputs:
        x: A numpy array, the dataset we wish to fit to gaussian.

        best_fit: A bool, decides whether to maximize the overlap of
        the gaussian with the data.

        num_buckets: An integer determining the number of buckets used when 
        evaluating the PDF corresponding to the data x.

        num_steps: An integer which in when best_fit is True determines
        the number of gradient descent steps when fitting to gaussian.

        learning_rate: A float setting the learning rate for the
        gradient descent when best_fit=True.
    '''
    # Evaluate the discretized PDF by bucketing the data.
    num_samples = x.shape[0]
    bucket_size = (np.max(x) - np.min(x))/num_buckets
    buckets, multiplicities = bucketize(x, num_buckets)
    
    pdf = np.array(multiplicities)/(bucket_size*num_samples)
    # Compute the mean and standard deviation of the data
    mean = np.mean(x)
    std = sp.sqrt(bucket_size * np.dot(np.square(buckets-mean),pdf))
    # Instantiate a gaussian with mean and std.
    g = Gaussian(mean, std)
    
    if best_fit:
        # Maximize loss = AUC((pdf-g)**2)/2
        for i in range(num_steps):
            # Compute derivatives d<pdf>/dmu, d<pdf>/dsigma
            gp_mu = g.pdf(buckets)*(buckets - g.mean)/g.std**2
            gp_sigma = g.pdf(buckets)*((buckets - g.mean)**2 - g.std**2)/g.std**3
            delta_mu = np.sum( (g.pdf(buckets) - pdf)*gp_mu )*bucket_size
            delta_sigma = np.sum( (g.pdf(buckets) - pdf)*gp_sigma)*bucket_size
            # Update mu and sigma
            g.set_mean = g.mean - learning_rate*delta_mu
            g.set_std = g.std - learning_rate*delta_sigma
            
    return g
