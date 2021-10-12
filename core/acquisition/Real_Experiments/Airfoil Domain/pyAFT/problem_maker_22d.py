import numpy as np
import matplotlib.pyplot as plt
import GPy
import itertools as it
import lhsmdu

import domain.problem_config_2d as mp

def problem_2in1out( xlower = 0, xupper = 10, nsamples = 10, ylower = 0, yupper = 20 ):
    '''n_dim_problem is a function to create a random, but well defined, problem
    landscape created by generating random points in a 3d landscape and 
    fitting a gp to the points. By constraining the hyperparameters we are able
    to produce a landscape with a controllable level of complexity.

    example :   my_problem_function = three_dim_problem( )

    INPUTS: (keywords, not required)
        xlower      -   Integer/Float - Defines the lower bound of all X's
        xupper      -   Integer/Float - Defines the upper bound of all X's
        nsamples    -   Integer       - number of points along each dimension
        ylower      -   Integer/Float - Defines the lower bound of Y
        yupper      -   Integer/Float - Defines the upper bound of Y  

    OUTPUT: 
        fun         -   Function      - A 3 dimensional function
            example :   yi = fun( [ xi1 , xi2 ] )
            example :   [yi,yj] = func( [ [ xi1 , xi2 ] , [ xj1 , xj2 ] )      
    '''
    X1   = np.linspace( xlower , xupper , nsamples )
    X2   = np.linspace( xlower , xupper , nsamples )
    Y    = np.ones( nsamples**2 )*np.random.uniform( ylower , yupper , nsamples**2 )
    kernel = GPy.kern.RBF( input_dim = 2 , variance = 0.001 , lengthscale = (xupper-xlower)/5 ) 
    X    = np.array (list( it.product( X1 , X2 ) ) ) 
    
    X    = np.array(lhsmdu.sample(nsamples**2,2))*10

    X = X.reshape( -1 , 2 )
    Y = Y.reshape( -1 , 1 )
    
    model = GPy.models.GPRegression( X, Y, noise_var = 1e-4 , kernel = kernel )
    def fun(x):
        assert( type(x) == list or type(x) == np.array , 'input must be a list or array')
        assert(len(x) >= 2  , 'Input is in incorrect format')  
        value = None
        try:
            value = [ float( model.predict( np.array( [ [ i ] ] ).reshape( -1 , 2 ) )[ 0 ] ) for i in x ]
            if value < ylower:
                value = ylower
            if value > yupper:
                value = yupper
            return( np.array(value) )
        except:
            pass
        try:
            value = float( model.predict( np.array( [ [ x ] ] ).reshape( -1 , 2 ) )[ 0 ] )
            if value < ylower:
                value = ylower
            if value > yupper:
                value = yupper
            return( np.array(value) )        
        except:
            print( '2in_1out problem ERROR - The x values do not match the required input size' )
        return( None )

    return( fun )


################################################################################
################## CODE TESTING ################################################
################################################################################

if __name__ == '__main__':
    xlower = 0 ; xupper = 10 ; nsamples = 10 ; ylower = 0 ; yupper = 20
    myfun = problem_2in1out( )
    X1   = np.linspace( xlower , xupper , nsamples*nsamples )
    X2   = np.linspace( xlower , xupper , nsamples*nsamples )
    X    = np.array(list(it.product(X1,X2)))
    
    import matplotlib as mpl
    from mpl_toolkits import mplot3d
    import numpy as np
    import matplotlib.pyplot as plt

    z = myfun(X)
    x = [i[0] for i in X]
    y = [i[1] for i in X]
    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    
    # Creating plot
    ax.scatter3D(x, y, z, color = "green")
    plt.title("simple 3D scatter plot")
    
    # show plot
    plt.show()

