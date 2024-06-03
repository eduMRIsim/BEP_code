import numpy as np
from scipy.io import loadmat
from scipy import ndimage
#from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from model import Model
#from PIL import Image
import registration as reg
import registration_util as util


def load_data(path_to_data):
    '''Load the data from the .mat file and return a dictionary of the fields in the .mat file. The .mat file contains a 1x1 struct called VObj (stands for Virtual Object). VObj contains the 16 fields described in https://mrilab.sourceforge.net/manual/MRiLab_User_Guide_v1_3/MRiLab_User_Guidech3.html#x8-120003.1
    
    Args:
    path_to_data (str): The path to the .mat file containing the data
    
    Returns:
    data_dict (dict): A dictionary containing the 16 fields in the VObj struct'''

    # Load the .mat file. 
    mat_contents = loadmat("resources/BrainHighResolution.mat")

    # Access the VObj struct
    VObj = mat_contents['VObj']
    
    # Create a dictionary of the fields
    data_dict = {}
    for field_name in VObj.dtype.names:
        data_dict[field_name] = VObj[0, 0][field_name]

    return data_dict

def scan(scan_parameters, model):
    '''Simulate a scan of the model using the scan parameters. The scan parameters include the TR (repetition time), TE (echo time), flip_angle. The function should return the simulated MRI image.
    
    Args:
    model (Model): The model to scan
    scan_parameters (dict): A dictionary containing the scan parameters
    
    Returns:
    simulated_image (np.ndarray): The simulated MRI image'''

    # Simulate the MRI image
    signal_array = calculate_signal(scan_parameters, model)

    # Add image artifacts
    simulated_image = add_artifacts(signal_array, scan_parameters)

    return simulated_image

def calculate_signal(scan_parameters, model):
    '''Calculate the signal intensity given the scan parameters for each voxel of the model. Note that the signal intensity is calculated using the signal equation for a spin echo sequence. 
    
    Args: 
    scan_parameters (dict): A dictionary containing the scan parameters
    model (Model): The model to scan'''

    TE = scan_parameters['TE']
    TR = scan_parameters['TR']
    TI = scan_parameters['TI']

    PD = model.PDmap[:, :] 
    T1 = model.T1map[:, :]
    T2 = model.T2map[:, :]

    signal_array = np.abs(PD * np.exp(np.divide(-TE,T2)) * (1 - 2 * np.exp(np.divide(-TI, T1)) + np.exp(np.divide(-TR, T1)))) # calculate the signal intensity using the signal equation for a spin echo sequence.  
    signal_array = np.nan_to_num(signal_array, nan=0) # replace all nan values with 0. This is necessary because the signal_array can contain nan values, for example if both TI and T1 are 0. 

    return signal_array

def add_artifacts(signal_array, scan_parameters):
    # This will be the main focus of your project. For now this function does not do anything
    return signal_array

def calculate_2dft(input_image):
    ft = np.fft.ifftshift(input_image)
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)
    return ft

def calculate_2dift(input):
    ift = np.fft.ifftshift(input)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    return ift.real

def main():
    data_dict = load_data("resources/BrainHighResolution.mat") #This functions loads the anatomical model from the .mat file and returns a dictionary of the data contained in the anatomical model. The 16 data fields contained in data_dict are described in https://mrilab.sourceforge.net/manual/MRiLab_User_Guide_v1_3/MRiLab_User_Guidech3.html#x8-120003.1

    print(data_dict.keys()) 

    model = Model("BrainHighResolution", data_dict["T1"], data_dict["T2"], data_dict["T2Star"], data_dict["Rho"], 
                  data_dict['XDim'], data_dict['YDim'], data_dict['ZDim'],
                  data_dict['XDimRes'], data_dict['YDimRes'], data_dict['ZDimRes'])

    # You can view a transverse slice of the T1 map by using the following code:
   
    # plt.figure()
    # plt.imshow(model.T1map[:, :, 90], cmap='gray')
    # plt.title("Model")
    # plt.show() # TYou will need to close the image before the rest of the code executes. 

    # Play around with the code to view the rest of the data. Think about the following: which anatomical direction (e.g. from left to right/right to left, top to bottom/bottom to top, front to back/back to front) do the x, y and z axes correspond to?
    

    # Simulate a scan
    scan_parameters = {
        # PD-weighted
        # "TE": 0.13, 
        # "TR": 0.864,
        # "TI": 0,
        
        # T1-weighted
        "TE": 0.264,
        "TR": 0.864,
        "TI": 0,
        
        # T2-weighted
        # "TE": 0.864,
        # "TR": 0.864,
        # "TI": 0,
    }

    simulated_image = scan(scan_parameters, model)
    return model, simulated_image, scan_parameters
          
    # You can view a transverse slice of the simulated image by using the following code:
   
    # plt.figure()
    # plt.imshow(simulated_image[:, :, 90], cmap='gray')
    # plt.title("Simulated Image")
    # plt.show()
    

    # Play around with the code to view the rest of the simulated image. What if you want to view a coronal or sagittal slice of the simulated image? You could also try simulating the image with different scan parameters. Try to generate a T1-weighted image, a T2-weighted image, a proton density-weighted image, etc. 

    # Also think about...what if you want to see the k-space data? How would you do that?


    # Read and process image
    
def axes_image(slice_nr):
       
    model, simulated_image = main()
    
    fig, ax = plt.subplots(1,3,figsize=(10,30))
    ax[0].imshow(model.T1map[:, :, slice_nr], cmap='gray')
    ax[0].set_title("Transversal")
        
    ax[1].imshow(model.T1map[:, slice_nr, :].swapaxes(-2,-1)[...,::-1], cmap='gray')
    ax[1].set_title("Sagittal")
        
    ax[2].imshow(np.rot90(model.T1map[slice_nr, :, :],3), cmap='gray')
    ax[2].set_title("Coronal")
    plt.show()
        
    
def fourier_image(slice_nr):
    
    model, simulated_image = main()
    
    image = simulated_image[:, :, slice_nr]
    ft = calculate_2dft(image)
        
    fig, ax = plt.subplots(1,3,figsize=(10,30))
    ax[0].imshow(model.T1map[:, :, slice_nr], cmap='gray')
    ax[0].set_title("Model")
        
    ax[1].imshow(image, cmap='gray')
    ax[1].set_title("Simulated Image")
        
    ax[2].imshow(np.log(abs(ft)), cmap='gray')
    ax[2].set_title("Fourier Transform")
    plt.show()

def translated_image(slice_nr, tx = 0, ty = 0):
    
    model, simulated_image, pars = main()
    
    image = simulated_image[:, :, slice_nr]
    N, M = image.shape
    image_translated = np.zeros_like(image)

    image_translated[max(ty,0):N+min(ty,0), max(tx,0):M+min(tx,0)] = image[-min(ty,0):N-max(ty,0),-min(tx,0):M-max(tx,0)]  
    ft = np.log(abs(calculate_2dft(image)))
    ft_trans = np.log(abs(calculate_2dft(image_translated)))        
                
    fig, ax = plt.subplots(1,4,figsize=(10,30))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Simulated Image")
        
    ax[1].imshow(image_translated, cmap='gray')
    ax[1].set_title("Translation")

    ax[2].imshow(ft, cmap='gray')
    ax[2].set_title("FT Original")
    
    ax[3].imshow(ft_trans, cmap='gray',vmin=ft.min(), vmax=ft.max())
    ax[3].set_title("FT Translation")
    
    plt.show()

def rotated_image(slice_nr, angle = 0):
     
    model, simulated_image = main()
    
    image = simulated_image[:, :, slice_nr]

    image_rotated = ndimage.rotate(image, angle, reshape=False)
    
    fig, ax = plt.subplots(1,2,figsize=(10,20))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Simulated Image")
        
    ax[1].imshow(image_rotated, cmap='gray', vmin=image.min(), vmax=image.max())
    ax[1].set_title("Rotation")
    
    plt.show()
    
def translation_loop(slice_nr, n):
    
    model, simulated_image = main()
    
    image = simulated_image[:, :, slice_nr]
    N, M = image.shape
    
    ty = 0
    for tx in range(n):
        image_translated = np.zeros_like(image)
        image_translated[max(ty,0):N+min(ty,0), max(tx,0):M+min(tx,0)] = image[-min(ty,0):N-max(ty,0),-min(tx,0):M-max(tx,0)]  


        plt.figure()
        plt.imshow(image_translated, cmap='gray')
        plt.title("Simulated Image")
        plt.show()
        
def rotation_loop(slice_nr, n):
    
    model, simulated_image = main()
    
    image = simulated_image[:, :, slice_nr]

    for angle in range(n):
        image_rotated = ndimage.rotate(image, angle, reshape=False)

        plt.figure()
        plt.imshow(image_rotated, cmap='gray', vmin=image.min(), vmax=image.max())
        plt.title("Simulated Image")
        plt.show()

def fr_plot(moved_image,fr_image, inv_fr_image):
    
    fig, ax = plt.subplots(1,3,figsize=(10,30))
    ax[0].imshow(moved_image, cmap='gray')
    ax[0].set_title("Simulated image")
            
    ax[1].imshow(np.log(abs(fr_image)), cmap='gray')
    ax[1].set_title("Fourier Transform")
        
    ax[2].imshow(inv_fr_image, cmap='gray')
    ax[2].set_title("Inverse Fourier Transform")
    plt.show()

def abrupt_translation(slice_nr, slice_type, speed, angle):
    
    model, simulated_image, pars = main()
    
    if slice_type == 'trans':  
        image = simulated_image[:, :, slice_nr]
    elif slice_type == 'sag':
        image = simulated_image[:, slice_nr, :].swapaxes(-2,-1)[...,::-1]
    else:
        raise Exception("No valid slice direction was chosen.")
        
    N, M = image.shape
    
    i = 0
    
    t = N * pars["TR"]
    y = speed * np.sin(np.pi * angle/180) * t
    x = speed * np.cos(np.pi * angle/180) * t
    
    if angle % 180 == 0:
        s = abs(round(x))+1
    else:
        s = abs(round(y))+1
    step = int(N/s)
    
    if (N % s) != 0: 
        step += 1  
        
    p = y / (s-1)
    q = x / (s-1)
        
    ft_org = calculate_2dft(image)
    comp_fr_image = np.zeros_like(ft_org)
    
    for a in range(s):
        
        mat_identity = np.array([[1, 0, a*p], [0, 1, a*q], [0, 0, 1]])
        image_translated = ndimage.affine_transform(image, np.linalg.inv(mat_identity))
        image_translated[image_translated < image.min()] = image.min()
        image_translated[image_translated > image.max()] = image.max()
        
        ft = calculate_2dft(image_translated)
        comp_fr_image[i:i+step,:] = ft[i:i+step,:]
        i += step 
        
        fr_plot(image_translated,comp_fr_image)        
        
        #image_translated = np.zeros_like(image)
        #image_translated[max(ty,0):N+min(ty,0), max(tx,0):M+min(tx,0)] = image[-min(ty,0):N-max(ty,0),-min(tx,0):M-max(tx,0)]  

    
    XDimRes = model.X_dim_res[0][0]
    YDimRes = model.Y_dim_res[0][0]
    
    dx = x * XDimRes * 100
    dy = y * YDimRes * 100
    distance = np.sqrt(dx**2 + dy**2)  
    
    print('Translation: {0:.2f} cm in x-direction, {1:.2f} cm in y-direction'.format(dx, dy))
    print('Total translation: {0:.2f} cm'.format(distance))
    #print('Speed: {0:.4f} cm/s'.format(distance/t))
    
        
        
def abrupt_rotation(slice_nr, slice_type, speed):
    
    model, simulated_image, pars = main()
    
    if slice_type == 'trans':  
        image = simulated_image[:, :, slice_nr]
    elif slice_type == 'sag':
        image = simulated_image[:, slice_nr, :].swapaxes(-2,-1)[...,::-1]
    else:
        raise Exception("No valid slice direction was chosen.")
    
    N, M = image.shape
    
    t = N * pars["TR"] 
    deg = speed * t
    
    s = abs(round(deg))+1
    step = int(N/s)
    
    if (N % s) != 0: 
        step += 1
        
    i = 0
    ft_org = calculate_2dft(image)
    comp_fr_image = np.zeros_like(ft_org)
    
    for a in range(s):
        if deg < 0:
            a *= -1
        
        angle = a * deg / (s-1)
        image_rotated = ndimage.rotate(image, angle, reshape=False)
        image_rotated[image_rotated < image.min()] = image.min()
        image_rotated[image_rotated > image.max()] = image.max()

        ft = calculate_2dft(image_rotated)
        comp_fr_image[i:i+step,:] = ft[i:i+step,:]
        
        i += step
        
        fr_plot(image_rotated,comp_fr_image)

    print('Rotation: {0:.2f} degrees'.format(angle))
    #rint('Speed: {0:.4f} degrees/s'.format(angle/t))

def mask(slice_nr, n):
    model, simulated_image = main()
    
    image = simulated_image[:, :, slice_nr]
    #image = simulated_image[:, slice_nr, :].swapaxes(-2,-1)[...,::-1]
    
    N, M = image.shape
    i = 0
    step = int(N/n)
    crop = np.zeros_like(image)
    for k in range(n):
        mask = np.zeros_like(image)
        mask[i:i+step,:] = 1
        i += step
        imgslice = mask*image
        crop += imgslice
        
        fig, ax = plt.subplots(1,3,figsize=(10,30))
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title("Simulated image")
            
        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title("Mask")
        
        ax[2].imshow(crop, cmap='gray')
        ax[2].set_title("Masked image")
        plt.show()

def fr_trans_rot_loop(slice_nr, slice_type, direction, n):
    
    m = abs(n)+1
    model, simulated_image = main()
    
    if slice_type == 'trans':  
        image = simulated_image[:, :, slice_nr]
    elif slice_type == 'sag':
        image = simulated_image[:, slice_nr, :].swapaxes(-2,-1)[...,::-1]
    else:
        raise Exception("No valid slice direction was chosen.")
        
    N, M = image.shape
    
    i = 0
    step = int(N/m)
        
    ft_org = calculate_2dft(image)
    comp_fr_image = np.zeros_like(ft_org)
    image_translated = np.zeros_like(image)
    
    for p in range(m):
        if n < 0:
            p *= -1
            
        if direction == 'x':
            tx = p
            ty = 0
        elif direction == 'y':
            tx = 0
            ty = p
        else: 
            raise Exception("No valid translation direction was chosen.")
            
        image_rotated = ndimage.rotate(image, p, order = 0, reshape=False)
        image_translated[max(ty,0):N+min(ty,0), max(tx,0):M+min(tx,0)] = image_rotated[-min(ty,0):N-max(ty,0),-min(tx,0):M-max(tx,0)]  
        
        ft = calculate_2dft(image_translated)
        comp_fr_image[i:i+step,:] = ft[i:i+step,:]
        i += step
        
        fig, ax = plt.subplots(1,3,figsize=(10,30))
        ax[0].imshow(image_translated, cmap='gray')
        ax[0].set_title("Simulated image")
            
        ax[1].imshow(np.log(abs(comp_fr_image)), cmap='gray')
        ax[1].set_title("Fourier Transform")
        
        ax[2].imshow(calculate_2dift(comp_fr_image), cmap='gray')
        ax[2].set_title("Inverse Fourier Transform")
        plt.show()

def gradual_translation(slice_nr, slice_type, speed, angle):
      
    model, simulated_image, pars = main()
    
    if slice_type == 'trans':  
        image = simulated_image[:, :, slice_nr]
    elif slice_type == 'sag':
        image = simulated_image[:, slice_nr, :].swapaxes(-2,-1)[...,::-1]
    else:
        raise Exception("No valid slice direction was chosen.")
        
    N, M = image.shape
        
    ft_org = calculate_2dft(image)
    comp_fr_image = np.zeros_like(ft_org)
    
    t = N * pars["TR"]
    y = speed * np.sin(np.pi * angle/180) * t
    x = speed * np.cos(np.pi * angle/180) * t
    p = y / (N-1)
    q = x / (N-1)
    
    for i in range(N):
        
        mat_identity = np.array([[1, 0, i*p], [0, 1, i*q], [0, 0, 1]])
        image_translated = ndimage.affine_transform(image, np.linalg.inv(mat_identity))
        image_translated[image_translated < image.min()] = image.min()
        image_translated[image_translated > image.max()] = image.max()
        
        ft = calculate_2dft(image_translated)
        comp_fr_image[i:i+1,:] = ft[i:i+1,:]
        inv_image = calculate_2dift(comp_fr_image)
        
        #fr_plot(image_translated,comp_fr_image, inv_image)  
        
        fig, ax = plt.subplots(2,2,figsize=(10,10))
        ax[0,0].imshow(image_translated, cmap='gray')
        ax[0,0].set_title("Simulated image")

        ax[0,1].imshow(np.log(abs(ft)), cmap='gray')
        ax[0,1].set_title("Fourier Transform")
              
        ax[1,0].imshow(np.log(abs(comp_fr_image)), cmap='gray')
        ax[1,0].set_title("Fourier Transform")
            
        ax[1,1].imshow(inv_image, cmap='gray')
        ax[1,1].set_title("Inverse Fourier Transform")
        plt.show()
        
        #image_translated = np.zeros_like(image)
        #image_translated[max(ty,0):N+min(ty,0), max(tx,0):M+min(tx,0)] = image[-min(ty,0):N-max(ty,0),-min(tx,0):M-max(tx,0)]  

    
    XDimRes = model.X_dim_res[0][0]
    YDimRes = model.Y_dim_res[0][0]
    
    dx = x * XDimRes * 100
    dy = y * YDimRes * 100
    distance = np.sqrt(dx**2 + dy**2)  
    
    print('Translation: {0:.2f} cm in x-direction, {1:.2f} cm in y-direction'.format(dx, dy))
    print('Total translation: {0:.2f} cm'.format(distance))
    #print('Speed: {0:.4f} cm/s'.format(distance/t))

def gradual_rotation(slice_nr, slice_type, speed):

    model, simulated_image, pars = main()
    
    if slice_type == 'trans':  
        image = simulated_image[:, :, slice_nr]
    elif slice_type == 'sag':
        image = simulated_image[:, slice_nr, :].swapaxes(-2,-1)[...,::-1]
    else:
        raise Exception("No valid slice direction was chosen.")
    
    N, M = image.shape
    t = N * pars["TR"]
    deg = speed * t
    #i = 0
    ft_org = calculate_2dft(image)
    comp_fr_image = np.zeros_like(ft_org)
    
    #angle = deg/(N-1)
    
    for i in range(N):
        angle = i * deg / (N-1) 
        image_rotated = ndimage.rotate(image, angle, reshape=False)
        image_rotated[image_rotated < image.min()] = image.min()
        image_rotated[image_rotated > image.max()] = image.max()

        ft = calculate_2dft(image_rotated)
        comp_fr_image[i:i+1,:] = ft[i:i+1,:]
        inv_image = calculate_2dift(comp_fr_image)
        #inv_image[inv_image < image.min()] = image.min()
        #inv_image[inv_image > image.max()] = image.max()
        #i += 1
        
        fr_plot(image_rotated,comp_fr_image,inv_image) 

    print('Rotation: {0:.2f} degrees'.format(angle))
    #print('Speed: {0:.4f} degrees/s'.format(angle/t))

def altered_k_border_rotation(slice_nr, slice_type, deg, begin, end):
    model, simulated_image, pars = main()
    
    if slice_type == 'trans':  
        image = simulated_image[:, :, slice_nr]
    elif slice_type == 'sag':
        image = simulated_image[:, slice_nr, :].swapaxes(-2,-1)[...,::-1]
    else:
        raise Exception("No valid slice direction was chosen.")
    
    N, M = image.shape

    
    ft_org = calculate_2dft(image)
    comp_fr_image = np.zeros_like(ft_org)
    
    i = 0
    for a in range(N-(end-begin)):
        angle = a * deg / (N-1-(end-begin)) 
        image_rotated = ndimage.rotate(image, angle, reshape=False)
        image_rotated[image_rotated < image.min()] = image.min()
        image_rotated[image_rotated > image.max()] = image.max()

        ft = calculate_2dft(image_rotated)
        comp_fr_image[i:i+1,:] = ft[i:i+1,:]
        
        if i+1 == begin:
            i += end-begin
            comp_fr_image[begin:end,:] = ft[begin:end,:]
        i += 1
        
        fr_plot(image_rotated,comp_fr_image)
    
    t = (N - (end - begin)) * pars["TR"] 
    
    print('Rotation: {0:.2f} degrees'.format(deg))
    print('Speed: {0:.4f} degrees/s'.format(deg/t))
    
def altered_k_centre_rotation(slice_nr, slice_type, deg, begin, end):
    model, simulated_image, pars = main()
    
    if slice_type == 'trans':  
        image = simulated_image[:, :, slice_nr]
    elif slice_type == 'sag':
        image = simulated_image[:, slice_nr, :].swapaxes(-2,-1)[...,::-1]
    else:
        raise Exception("No valid slice direction was chosen.")
    
    N, M = image.shape

    ft_org = calculate_2dft(image)
    comp_fr_image = np.zeros_like(ft_org)
    comp_fr_image[:begin,:] = ft_org[:begin,:]
    
    i = begin
    for a in range(end-begin):
        
        angle = a * deg / (end-begin-1) 
        image_rotated = ndimage.rotate(image, angle, reshape=False)
        image_rotated[image_rotated < image.min()] = image.min()
        image_rotated[image_rotated > image.max()] = image.max()

        ft = calculate_2dft(image_rotated)
        comp_fr_image[i:i+1,:] = ft[i:i+1,:]
        
        i += 1
        
        fr_plot(image_rotated,comp_fr_image)
    
    comp_fr_image[end:,:] = ft_org[end:,:]
    fr_plot(image_rotated,comp_fr_image)
    
    t = (end - begin) * pars["TR"] 

    print('Rotation: {0:.2f} degrees'.format(deg))
    print('Speed: {0:.4f} degrees/s'.format(deg/t))
    
def periodic_motion(slice_nr, slice_type, direction, A, f):
    model, simulated_image, pars = main()
    
    if slice_type == 'trans':  
        image = simulated_image[:, :, slice_nr]
    elif slice_type == 'sag':
        image = simulated_image[:, slice_nr, :].swapaxes(-2,-1)[...,::-1]
    else:
        raise Exception("No valid slice direction was chosen.")
        
    N, M = image.shape
          
    ft_org = calculate_2dft(image)
    comp_fr_image = np.zeros_like(ft_org)
    
    for i in range(N):
                
        harmonic = A * np.sin(2 * np.pi * f * pars["TR"] * i)
        
        if direction == 'y':
            p = harmonic
            q = 0
        elif direction == 'x':
            p = 0
            q = harmonic

        mat_identity = np.array([[1, 0, p], [0, 1, q], [0, 0, 1]])
        
        image_translated = ndimage.affine_transform(image, np.linalg.inv(mat_identity))
        #image_translated = np.zeros_like(image)
        #image_translated[max(p,0):N+min(p,0), max(q,0):M+min(q,0)] = image[-min(p,0):N-max(p,0),-min(q,0):M-max(q,0)]  
        image_translated[image_translated < image.min()] = image.min()
        image_translated[image_translated > image.max()] = image.max()

        ft = calculate_2dft(image_translated)
        comp_fr_image[i:i+1,:] = ft[i:i+1,:]
                
        fr_plot(image_translated,comp_fr_image) 
        
    XDimRes = model.X_dim_res[0][0]
    YDimRes = model.Y_dim_res[0][0]
    
    print('Amplitude: {0:.2f} cm'.format(A*YDimRes*100))
    print('Repetitition every {0:.2f} lines'.format(1/(f*pars["TR"])))

   
#periodic_motion(100,'trans','y',5,0.4)
gradual_translation(100, 'trans', 0.0536*4, 0)
#altered_k_centre_rotation(100,'trans',15,72,144)



# if __name__ == "__main__":
#     main()
