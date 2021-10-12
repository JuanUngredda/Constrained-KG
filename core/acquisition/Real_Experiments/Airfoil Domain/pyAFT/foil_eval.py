import numpy as np
import os         # creating xfoil directory
import subprocess # 'tail' to read xfoil output
import scipy
import ctypes
import gc

# ---------------------------------------------------------------------------- #
def polyArea(x,y): # Shoelace formula
  return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1))) 


def getThick(coord,pos,pt_per_size = 100):    
    assert np.shape(coord)[1] == 2*pt_per_size , "thickness constraint assumes 400pt foil: %i pts" %pt_per_size*2
    top = coord[1,:pt_per_size]
    bot = coord[1,pt_per_size:]
    tc = top[::-1] - bot[:] # thickness per chord
    return tc[pos-1]


def evalFoil(foil):
  # Constraints
  area = polyArea(foil[0,:], foil[1,:])
  # Additional constraints if you are using the FFD not parsec
  FFD = False
  if FFD:
    minArea = 0.05
    maxArea = 0.20
    midThick= getThick(foil,35*2) > 0.1
    endThick= getThick(foil,90*2) > 0.005
    valid = (area > minArea) & (area < maxArea) & midThick & endThick
  else:
    valid = True
    
  # Evaluate if valid
  if valid:
    cd, cl = xfoilEval(foil)
    fit = -np.log(cd)
    beh = np.array((cl,area))
    if cl < 0:
      valid = False

  if valid:
    return fit, beh
  else:
    return np.nan, np.array((np.nan,np.nan))
 

def xfoilEval(foil, wd='/tmp/xfoil/'):
  try:
      # Create target Directory
      os.mkdir(wd)
      print("Directory " , wd ,  " Created ") 
  except FileExistsError:
      #print("Directory " , wd ,  " already exists")
      pass

  if np.shape(foil)[1] != 2: # put in expected form
    foil = foil.T
  #scipy.random.seed()
  id = np.random.randint(1e7)
  fname = 'xfoil' + str(id)
  f_foil = wd + fname + '.foil'
  f_comm = wd + fname + '.inp'
  f_data = wd + fname + '.dat'
  
  # Write coord file
  if np.shape(foil)[1] != 2: # put in expected form
    foil = foil.T
  np.savetxt(f_foil,foil,header=str(id),fmt='%2f')
  
  # Make command file
  with open(f_comm, 'a') as comm_file:
      comm_file.write('\n\nPLOP\nG\n\n') # Disable graphics
      comm_file.write('load ' + f_foil + '\n\n');   # Load Foil
      comm_file.write('pane\n');
      comm_file.write('oper\n');    
      comm_file.write('iter\n');
      comm_file.write('100\n');
      comm_file.write('re 1e+06\n');
      comm_file.write('mach 0.5\n');
      comm_file.write('visc\n');
      comm_file.write('pacc\n\n\n');    
      comm_file.write('alfa 2.7\n');
      comm_file.write('pwrt\n');
      comm_file.write(f_data + '\n');
      comm_file.write('plis\n\n');
      comm_file.write('quit\n');

  # Run Xfoil
  # ** Testing on mac requires 'gtimeout' from: brew install coreutils **
  xFoilPath = os.getcwd()+'/domain/pyAFT/xfoil/bin/xfoil'
  #xFoilPath = '/home/pkent/bin/xfoil'
  command = (xFoilPath + ' -g < ' + f_comm + ' > ' + wd + 'xfoil' + str(id) + '.out')
  err = os.system('timeout 1s ' + command + ' 2>/dev/null' + ' -k') # handle errors and hangs
  gc.collect()
  libc = ctypes.CDLL("libc.so.6")
  libc.malloc_trim(0)
  # Get drag and lift values
  try:
    line = subprocess.check_output(['tail', '-1', f_data])
    raw = line.split(None,3)
    cL = eval(raw[1])
    cD = eval(raw[2])
    #print(raw)
  except:
    cL = np.nan
    cD = np.nan
    #print('*| Xfoil did not converge')

  os.system('rm ' + wd + fname + '.*') # remove log files
  return cD, cL

   
    
