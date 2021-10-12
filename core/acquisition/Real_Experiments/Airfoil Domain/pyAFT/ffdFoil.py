import numpy as np

# FFD
import scipy
from scipy.special import comb       # For bernstein polynomials
from matplotlib import pyplot as plt # plotting foils

# XFoil
import os         # creating xfoil directory
import subprocess # 'tail' to read xfoil output

class FfdFoil():
  def __init__(self, nPts, base=None, defRange=1):
    if isinstance(base, str):
      base = np.genfromtxt(base, delimiter=',')
    else:
      base = naca0012()

    self.base = base
    self.nPts = int(nPts)
    self.yOrigin = np.min(base[1,:])
    self.yHeight = np.max(base[1,:])-self.yOrigin
    self.defRange = defRange
    self.bern_x, self.bern_y = self.getBern() # Precalc for speed


  def randomInd(self, dim_x, params):
    mutSigma = 0.2
    x = 0.5 + np.random.randn(dim_x) * mutSigma
    randInd = np.clip(x, a_min=params["min"][0:len(x)], \
                         a_max=params["max"][0:len(x)])
    return x

  def getBern(self):
    # Split in dimensions
    nX = int( (self.nPts+4)/2 ) # All points along chord and extra for LE and TE
    nY = 2                      # Upper and lower

    # Normalize mesh to bounding box
    mesh = np.copy(self.base)
    mesh[1,:] = (mesh[1,:]-self.yOrigin)/self.yHeight

    # Compute Bernstein Polynomials (Bezier Matrices)
    nMeshPts = self.base.shape[1]
    bern_x = np.zeros((nX,nMeshPts))
    bern_y = np.zeros((nY,nMeshPts))

    for i in range(nX):
      aux1 = (1-mesh[0,:]) ** (nX-(i+1))
      aux2 = (  mesh[0,:]) ** i
      bern_x[i,:] = comb(nX-1,i) * aux1 * aux2

    for i in range(nY):
      aux1 = (1-mesh[1,:]) ** (nY-(i+1))
      aux2 = (  mesh[1,:]) ** i
      bern_y[i,:] = comb(nY-1,i) * aux1 * aux2

    return bern_x, bern_y


  def express(self, deform, getPts=False):
    # Parameterize Deformation    
    # - Make intuitive, first half is top pts, second half bottom pts
    # - Encoding between 0 and 1, deformations are between -0.5 and +0.5
    split = int(self.nPts/2)
    tmp = np.c_[deform[split:None],deform[0:split]]
    deform = tmp.flatten()   
    deform = np.r_[0,0,(deform-0.5)*2,0,0] # don't deform LE or TE
    deform *= self.defRange

    # Split in dimensions
    nX = int(deform.shape[0]/2) # All points along chord
    nY = 2                      # Upper and lower
    deformation = np.reshape(deform,(nX,nY))

    # Normalize mesh to bounding box
    mesh = np.copy(self.base)
    mesh[1,:] = (mesh[1,:]-self.yOrigin)/self.yHeight

    # Calculate Deformation in FFD coordinate space
    mesh_shift = np.zeros_like(self.base)
    for j in range(nY): 
      for i in range(nX):
        mesh_shift[1] += self.bern_x[i,:] * self.bern_y[j,:] * deformation[i,j]

    # Apply shifts and scale back to original space
    new_mesh = mesh + mesh_shift
    new_mesh[1,:] = new_mesh[1,:]*self.yHeight+self.yOrigin

    if getPts:
      # Control point locations (just for visualization)
      pts = deformation*self.yHeight
      pts[:,0] += self.yOrigin # Bottom Pts
      pts[:,1] += self.yOrigin + self.yHeight # Top Pts
      return new_mesh,pts
    else:
      return new_mesh


  def fPlot(self, deform, axis=False, scale='equal'):
    ''' Plots an airfoil with control points'''
    if axis is not False:
      ax = axis
      fig = ax.figure.canvas 
    else:
      fig, ax = plt.subplots()

    afCoords, cPts = self.express(deform, getPts=True)
    x = afCoords[0,:]
    y = afCoords[1,:]
    if np.any(cPts):
        ax.plot(np.linspace(0,1,cPts.shape[0]),cPts[:,0],'r--o')
        ax.plot(np.linspace(0,1,cPts.shape[0]),cPts[:,1],'b--o')
    ax.plot(x,y,'-')
    ax.axis(scale)
    ax.grid()
    return ax


  def evalGenome(self, deform):
    foil = self.express(deform)
    fit, beh = evalFoil(foil)

    # Normalize behavior vectors between 0 and 1
    behMin = np.array((0.00, 0.05))
    behMax = np.array((1.20, 0.20))
    beh = (beh-behMin)/(behMax-behMin)
    beh = np.clip(beh,0.0,1.0)

    return fit, beh
    
def naca0012():
  return \
  np.array([[9.9975e-01,  9.9901e-01,  9.9778e-01,  9.9606e-01,  9.9384e-01,
             9.9114e-01,  9.8796e-01,  9.8429e-01,  9.8015e-01,  9.7553e-01,
             9.7044e-01,  9.6489e-01,  9.5888e-01,  9.5241e-01,  9.4550e-01,
             9.3815e-01,  9.3037e-01,  9.2216e-01,  9.1354e-01,  9.0451e-01,
             8.9508e-01,  8.8526e-01,  8.7506e-01,  8.6448e-01,  8.5355e-01,
             8.4227e-01,  8.3066e-01,  8.1871e-01,  8.0645e-01,  7.9389e-01,
             7.8104e-01,  7.6791e-01,  7.5452e-01,  7.4088e-01,  7.2699e-01,
             7.1289e-01,  6.9857e-01,  6.8406e-01,  6.6937e-01,  6.5451e-01,
             6.3950e-01,  6.2435e-01,  6.0907e-01,  5.9369e-01,  5.7822e-01,
             5.6267e-01,  5.4705e-01,  5.3139e-01,  5.1570e-01,  5.0000e-01,
             4.8429e-01,  4.6860e-01,  4.5295e-01,  4.3733e-01,  4.2178e-01,
             4.0631e-01,  3.9093e-01,  3.7566e-01,  3.6050e-01,  3.4549e-01,
             3.3063e-01,  3.1594e-01,  3.0143e-01,  2.8711e-01,  2.7300e-01,
             2.5912e-01,  2.4548e-01,  2.3209e-01,  2.1896e-01,  2.0611e-01,
             1.9355e-01,  1.8129e-01,  1.6934e-01,  1.5773e-01,  1.4645e-01,
             1.3552e-01,  1.2494e-01,  1.1474e-01,  1.0492e-01,  9.5492e-02,
             8.6460e-02,  7.7836e-02,  6.9629e-02,  6.1847e-02,  5.4497e-02,
             4.7586e-02,  4.1123e-02,  3.5112e-02,  2.9560e-02,  2.4472e-02,
             1.9853e-02,  1.5708e-02,  1.2042e-02,  8.8560e-03,  6.1560e-03,
             3.9430e-03,  2.2190e-03,  9.8700e-04,  2.4700e-04,  0.0000e+00,
             2.4700e-04,  9.8700e-04,  2.2190e-03,  3.9430e-03,  6.1560e-03,
             8.8560e-03,  1.2042e-02,  1.5708e-02,  1.9853e-02,  2.4472e-02,
             2.9560e-02,  3.5112e-02,  4.1123e-02,  4.7586e-02,  5.4497e-02,
             6.1847e-02,  6.9629e-02,  7.7836e-02,  8.6460e-02,  9.5492e-02,
             1.0492e-01,  1.1474e-01,  1.2494e-01,  1.3552e-01,  1.4645e-01,
             1.5773e-01,  1.6934e-01,  1.8129e-01,  1.9355e-01,  2.0611e-01,
             2.1896e-01,  2.3209e-01,  2.4548e-01,  2.5912e-01,  2.7300e-01,
             2.8711e-01,  3.0143e-01,  3.1594e-01,  3.3063e-01,  3.4549e-01,
             3.6050e-01,  3.7566e-01,  3.9093e-01,  4.0631e-01,  4.2178e-01,
             4.3733e-01,  4.5295e-01,  4.6860e-01,  4.8429e-01,  5.0000e-01,
             5.1570e-01,  5.3139e-01,  5.4705e-01,  5.6267e-01,  5.7822e-01,
             5.9369e-01,  6.0907e-01,  6.2435e-01,  6.3950e-01,  6.5451e-01,
             6.6937e-01,  6.8406e-01,  6.9857e-01,  7.1289e-01,  7.2699e-01,
             7.4088e-01,  7.5452e-01,  7.6791e-01,  7.8104e-01,  7.9389e-01,
             8.0645e-01,  8.1871e-01,  8.3066e-01,  8.4227e-01,  8.5355e-01,
             8.6448e-01,  8.7506e-01,  8.8526e-01,  8.9508e-01,  9.0451e-01,
             9.1354e-01,  9.2216e-01,  9.3037e-01,  9.3815e-01,  9.4550e-01,
             9.5241e-01,  9.5888e-01,  9.6489e-01,  9.7044e-01,  9.7553e-01,
             9.8015e-01,  9.8429e-01,  9.8796e-01,  9.9114e-01,  9.9384e-01,
             9.9606e-01,  9.9778e-01,  9.9901e-01,  9.9975e-01,  1.0000e+00],
           [ 3.6000e-05,  1.4300e-04,  3.2200e-04,  5.7200e-04,  8.9100e-04,
             1.2800e-03,  1.7370e-03,  2.2600e-03,  2.8490e-03,  3.5010e-03,
             4.2160e-03,  4.9900e-03,  5.8220e-03,  6.7100e-03,  7.6510e-03,
             8.6430e-03,  9.6840e-03,  1.0770e-02,  1.1900e-02,  1.3071e-02,
             1.4280e-02,  1.5523e-02,  1.6800e-02,  1.8106e-02,  1.9438e-02,
             2.0795e-02,  2.2173e-02,  2.3569e-02,  2.4981e-02,  2.6405e-02,
             2.7838e-02,  2.9279e-02,  3.0723e-02,  3.2168e-02,  3.3610e-02,
             3.5048e-02,  3.6478e-02,  3.7896e-02,  3.9300e-02,  4.0686e-02,
             4.2052e-02,  4.3394e-02,  4.4708e-02,  4.5992e-02,  4.7242e-02,
             4.8455e-02,  4.9626e-02,  5.0754e-02,  5.1833e-02,  5.2862e-02,
             5.3835e-02,  5.4749e-02,  5.5602e-02,  5.6390e-02,  5.7108e-02,
             5.7755e-02,  5.8326e-02,  5.8819e-02,  5.9230e-02,  5.9557e-02,
             5.9797e-02,  5.9947e-02,  6.0006e-02,  5.9971e-02,  5.9841e-02,
             5.9614e-02,  5.9288e-02,  5.8863e-02,  5.8338e-02,  5.7712e-02,
             5.6986e-02,  5.6159e-02,  5.5232e-02,  5.4206e-02,  5.3083e-02,
             5.1862e-02,  5.0546e-02,  4.9138e-02,  4.7638e-02,  4.6049e-02,
             4.4374e-02,  4.2615e-02,  4.0776e-02,  3.8859e-02,  3.6867e-02,
             3.4803e-02,  3.2671e-02,  3.0473e-02,  2.8213e-02,  2.5893e-02,
             2.3517e-02,  2.1088e-02,  1.8607e-02,  1.6078e-02,  1.3503e-02,
             1.0884e-02,  8.2230e-03,  5.5210e-03,  2.7790e-03,  0.0000e+00,
            -2.7790e-03, -5.5210e-03, -8.2230e-03, -1.0884e-02, -1.3503e-02,
            -1.6078e-02, -1.8607e-02, -2.1088e-02, -2.3517e-02, -2.5893e-02,
            -2.8213e-02, -3.0473e-02, -3.2671e-02, -3.4803e-02, -3.6867e-02,
            -3.8859e-02, -4.0776e-02, -4.2615e-02, -4.4374e-02, -4.6049e-02,
            -4.7638e-02, -4.9138e-02, -5.0546e-02, -5.1862e-02, -5.3083e-02,
            -5.4206e-02, -5.5232e-02, -5.6159e-02, -5.6986e-02, -5.7712e-02,
            -5.8338e-02, -5.8863e-02, -5.9288e-02, -5.9614e-02, -5.9841e-02,
            -5.9971e-02, -6.0006e-02, -5.9947e-02, -5.9797e-02, -5.9557e-02,
            -5.9230e-02, -5.8819e-02, -5.8326e-02, -5.7755e-02, -5.7108e-02,
            -5.6390e-02, -5.5602e-02, -5.4749e-02, -5.3835e-02, -5.2862e-02,
            -5.1833e-02, -5.0754e-02, -4.9626e-02, -4.8455e-02, -4.7242e-02,
            -4.5992e-02, -4.4708e-02, -4.3394e-02, -4.2052e-02, -4.0686e-02,
            -3.9300e-02, -3.7896e-02, -3.6478e-02, -3.5048e-02, -3.3610e-02,
            -3.2168e-02, -3.0723e-02, -2.9279e-02, -2.7838e-02, -2.6405e-02,
            -2.4981e-02, -2.3569e-02, -2.2173e-02, -2.0795e-02, -1.9438e-02,
            -1.8106e-02, -1.6800e-02, -1.5523e-02, -1.4280e-02, -1.3071e-02,
            -1.1900e-02, -1.0770e-02, -9.6840e-03, -8.6430e-03, -7.6510e-03,
            -6.7100e-03, -5.8220e-03, -4.9900e-03, -4.2160e-03, -3.5010e-03,
            -2.8490e-03, -2.2600e-03, -1.7370e-03, -1.2800e-03, -8.9100e-04,
            -5.7200e-04, -3.2200e-04, -1.4300e-04, -3.6000e-05,  0.0000e+00]])