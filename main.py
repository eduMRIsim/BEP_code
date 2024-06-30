import numpy as np
from scipy.io import loadmat
from scipy import ndimage
import matplotlib.pyplot as plt
from model import Model


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
    signal_array (np.ndarray): The simulated MRI image'''

    # Simulate the MRI image
    signal_array = calculate_signal(scan_parameters, model)

    # Add image artifacts
    # simulated_image = add_artifacts(signal_array, scan_parameters)

    return signal_array

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

def calculate_2dft(input_image):
    """
    Calculates the 2D FFT of an input image
    
    Input: 
    input_image (array)
    
    Output:
        - ft (array): 2D FFT of input_image (its k-space visualisation
                      can be acquired by taking the log of its magnitude)
    """
    
    ft = np.fft.ifftshift(input_image)
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)
    return ft

def calculate_2dift(input_array):
    """
    Calculates the 2D inverse FFT of an input array and returns its real part
    
    Input: 
    input_array (array): this array should be in the Fourier domain 
    
    Output:
        - ft (array): 2D IFFT of input_image (should contain the motion-affected image)
    """
    ift = np.fft.ifftshift(input_array)
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
    
def axes_image(slice_nr):
    """
    Plots an T1-weighted image of the 3D MRiLab model in three different slice planes

    Input:
        - slice_nr (int): slice number corresponding to the desired slice depiction
    """
    model, simulated_image, pars = main()
    
    fig, ax = plt.subplots(1,3,figsize=(10,30))
    ax[0].imshow(model.T1map[:, :, slice_nr], cmap='gray')
    ax[0].set_title("Transversal")
        
    ax[1].imshow(model.T1map[:, slice_nr, :].swapaxes(-2,-1)[...,::-1], cmap='gray')
    ax[1].set_title("Sagittal")
        
    ax[2].imshow(np.rot90(model.T1map[slice_nr, :, :],3), cmap='gray')
    ax[2].set_title("Coronal")
    plt.show()
    
def fourier_image(slice_nr):
    """
    Selects three images:
        1. A transversal T1-weighted slice from the MRiLab model
        2. The same slice as simulated by the function calculate_signal()
        3. The FFT of the simulated slice image
        
    A plot consisting of these three images is returned
    
    Input:
        - slice_nr (int): slice number corresponding to the desired slice depiction  
    """
    model, simulated_image, pars = main()
    
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

def slice_selection(slice_nr, slice_plane):
    """
    Selects a 2D slice from the MRilab model
    
    Input:
        - slice_nr (int): slice number corresponding to the desired slice
        - slice_plane (str): the desired slice plane, either transversal (trans),
                             sagittal (sag) or coronal (cor)   
                             
    """
    model, simulated_image, pars = main()
    
    if slice_plane == 'trans':  
        slice_image = simulated_image[:, :, slice_nr]
    elif slice_plane == 'sag':
        slice_image = simulated_image[:, slice_nr, :].swapaxes(-2,-1)[...,::-1]
    elif slice_plane == 'cor':   
        slice_image = np.rot90(simulated_image[slice_nr, :, :],3)
    else:
        raise Exception("No valid slice plane was chosen.")

    return slice_image

def fr_plot(moved_image, comp_fr_image):
    """
    Returns a plot consisting of three images:
        1. The moved 2D reference image
        2. The log-scaled magnitude of the composite k-space
        3. The IFFT of the composite k-space
    
    Input:
        - moved_image (array): the moved reference image in its current position
        - comp_fr_image (array): the composite k-space

    """
    fig, ax = plt.subplots(1,3,figsize=(10,30))
    ax[0].imshow(moved_image, cmap='gray')
    ax[0].set_title("Simulated image")
            
    ax[1].imshow(np.log10(abs(comp_fr_image)), cmap='gray')
    ax[1].set_title("Fourier Transform")
        
    ax[2].imshow(calculate_2dift(comp_fr_image), cmap='gray')
    ax[2].set_title("Inverse Fourier Transform")
    plt.show()

def abrupt_translation(slice_nr, slice_plane, speed, angle, sample_amount):
    """
    Simulates the artefacts corresponding with the occurrence of subject translation 
    in an abrupt way. 
    
    Returns:
        - an updated plot with the reference image in its
          current position, the composite k-space and the motion-affected image corresponding
          with the composite k-space.
    Prints:
        - the total translation in x and y-direction and its total magnitude
    

    Input:
        - slice_nr (int): slice number corresponding to the desired slice
        - slice_plane (str): the desired slice plane, either transversal (trans),
                             sagittal (sag) or coronal (cor)
        - speed (float): the desired translation speed in pixel/s
        - angle (float): the desired translation angle in degrees
        - sample_amount (int): the amount of steps in which the translation is desired to take place
    """
    model, simulated_image, pars = main() # Initiate image simulation 
    
    image = slice_selection(slice_nr, slice_plane) # Select reference image slice from simulation
        
    N, M = image.shape # Determine the dimensions of the image
       
    t = N * pars["TR"] # Calculate scan duration
    y = speed * np.sin(np.pi * angle/180) * t # Calculate total displacement in y-direction
    x = speed * np.cos(np.pi * angle/180) * t # Calculate total displacement in x-direction
    
    dy = y / (sample_amount-1) # Determine the y-translation of one step
    dx = x / (sample_amount-1) # Determine the x-translation of one step
    
    step = int(N/sample_amount) # Calculate the number of lines needed for each k-space sample
    
    if (N % sample_amount) != 0: # When the amount of steps is not an integer, the the sample size is increased by one line,
        step += 1                # or else the comp. k-space will not get fully filled
               
    # Create empty composite k-space
    ft_org = calculate_2dft(image)
    comp_fr_image = np.zeros_like(ft_org)
    
    i = 0 # Initiate step count
    
    for a in range(sample_amount):
        
        # Determine transformation matrix
        mat_identity = np.array([[1, 0, a*dy], [0, 1, a*dx], [0, 0, 1]]) 
        
        # Apply transformation matrix to reference image
        image_translated = ndimage.affine_transform(image, np.linalg.inv(mat_identity), order = 5)
        
        # Apply original contrast window
        image_translated[image_translated < image.min()] = image.min()
        image_translated[image_translated > image.max()] = image.max()
        
        ft = calculate_2dft(image_translated) # Apply FFT to moved image
        comp_fr_image[i:i+step,:] = ft[i:i+step,:] # Add sample to comp. k-space
        
        i += step  # Update step count
        
        fr_plot(image_translated,comp_fr_image) # Plot results     
    
    # Determine pixel dimensions 
    XDimRes = model.X_dim_res[0][0] 
    YDimRes = model.Y_dim_res[0][0]
    
    # Calculate the total translation in x and y-direction in cm
    x_cm = x * XDimRes * 100
    y_cm = y * YDimRes * 100
    output_distance = np.sqrt(x_cm**2 + y_cm**2) # Calculation total translation 
    
    print('Translation: {0:.2f} cm in x-direction, {1:.2f} cm in y-direction'.format(x_cm,y_cm))
    print('Total translation: {0:.2f} cm'.format(output_distance))
    
def abrupt_rotation(slice_nr, slice_plane, speed, sample_amount):
    """
    Simulates the artefacts corresponding with the occurrence of subject rotation 
    in an abrupt way. 
    
    Returns:
        - an updated plot with the reference image in its
          current position, the composite k-space and the motion-affected image corresponding
          with the composite k-space.
    Prints:
        - the total rotation angle
    
    
    Input:
        - slice_nr (int): slice number corresponding to the desired slice
        - slice_plane (str): the desired slice plane, either transversal (trans),
                             sagittal (sag) or coronal (cor)
        - speed (float): the desired rotation speed in degree/s
        - sample_amount (int): the amount of steps in which the rotation is desired to take place
    """
    model, simulated_image, pars = main() # Initiate image simulation
    
    image = slice_selection(slice_nr, slice_plane) # Select reference image slice from simulation
        
    N, M = image.shape # Determine the dimensions of the image
       
    t = N * pars["TR"] # Calculate scan duration
    deg = speed * t # Calculate total rotation angle
    
    step = int(N/sample_amount) # Calculate the number of lines needed for each k-space sample
    
    if (N % sample_amount) != 0: # When the amount of steps is not an integer, the the sample size is increased by one line,
        step += 1                # or else the comp. k-space will not get fully filled
        
    # Create empty composite k-space
    ft_org = calculate_2dft(image)
    comp_fr_image = np.zeros_like(ft_org)
    
    i = 0 # Initialise step count
    
    for a in range(sample_amount):
        
        angle = a * deg / (sample_amount-1) # Determine the rotation angle for each step
        
        # Apply rotation to the reference image
        image_rotated = ndimage.rotate(image, angle, reshape=False, order = 5)
        
        # Apply original contrast window
        image_rotated[image_rotated < image.min()] = image.min()
        image_rotated[image_rotated > image.max()] = image.max()

        ft = calculate_2dft(image_rotated) # Apply FFT to moved image
        comp_fr_image[i:i+step,:] = ft[i:i+step,:] # Add sample to comp. k-space
        
        i += step  # Update step count
        
        fr_plot(image_rotated,comp_fr_image) # Plot results  
        
    print('Rotation: {0:.2f} degrees'.format(angle))

def gradual_translation(slice_nr, slice_plane, speed, angle):
    """
    Simulates the artefacts corresponding with the occurrence of gradual subject translation.
    The composite k-space gets updated line-by-line.
     
    Returns:
        - an updated plot with the reference image in its
          current position, the composite k-space and the motion-affected image corresponding
          with the composite k-space.
    Prints:
        - the total translation in x and y-direction and its total magnitude
    

    Input:
        - slice_nr (int): slice number corresponding to the desired slice
        - slice_plane (str): the desired slice plane, either transversal (trans),
                             sagittal (sag) or coronal (cor)
        - speed (float): the desired translation speed in pixel/s
        - angle (float): the desired translation angle in degrees
        
    """
    model, simulated_image, pars = main() # Initiate image simulation
    
    image = slice_selection(slice_nr, slice_plane) # Select reference image slice from simulation
        
    N, M = image.shape # Determine the dimensions of the image
       
    t = N * pars["TR"] # Calculate scan duration
     
    # Create empty composite k-space
    ft_org = calculate_2dft(image)
    comp_fr_image = np.zeros_like(ft_org)
    
    y = speed * np.sin(np.pi * angle/180) * t # Calculate total displacement in y-direction
    x = speed * np.cos(np.pi * angle/180) * t # Calculate total displacement in x-direction
    dy = y / (N-1) # Determine the y-translation of one step
    dx = x / (N-1) # Determine the x-translation of one step
    
    
    for i in range(N):
        
        # Determine transformation matrix
        mat_identity = np.array([[1, 0, i*dy], [0, 1, i*dx], [0, 0, 1]])
        
        # Apply transformation matrix to reference image
        image_translated = ndimage.affine_transform(image, np.linalg.inv(mat_identity), order = 5)
        
        # Apply original contrast window
        image_translated[image_translated < image.min()] = image.min()
        image_translated[image_translated > image.max()] = image.max()
        
        ft = calculate_2dft(image_translated) # Apply FFT to moved image
        comp_fr_image[i,:] = ft[i,:] # Add sample to comp. k-space
        
        fr_plot(image_translated,comp_fr_image) # Plot results  

    # Determine pixel dimensions 
    XDimRes = model.X_dim_res[0][0] 
    YDimRes = model.Y_dim_res[0][0]
    
    # Calculate the total translation in x and y-direction in cm
    x_cm = x * XDimRes * 100
    y_cm = y * YDimRes * 100
    output_distance = np.sqrt(x_cm**2 + y_cm**2) # Calculation total translation
    
    print('Translation: {0:.2f} cm in x-direction, {1:.2f} cm in y-direction'.format(x_cm, y_cm))
    print('Total translation: {0:.2f} cm'.format(output_distance))

def gradual_rotation(slice_nr, slice_plane, speed):
    """
    Simulates the artefacts corresponding with the occurrence of gradual subject rotation.
    The composite k-space gets updated line-by-line.
    
    Returns:
        - an updated plot with the reference image in its
          current position, the composite k-space and the motion-affected image corresponding
          with the composite k-space.
    Prints:
        - the total rotation angle
    
    
    Input:
        - slice_nr (int): slice number corresponding to the desired slice
        - slice_plane (str): the desired slice plane, either transversal (trans),
                             sagittal (sag) or coronal (cor)
        - speed (float): the desired rotation speed in degree/s
    """
    model, simulated_image, pars = main() # Initiate image simulation
    
    image = slice_selection(slice_nr, slice_plane) # Select reference image slice from simulation
        
    N, M = image.shape # Determine the dimensions of the image
       
    t = N * pars["TR"] # Calculate scan duration
    deg = speed * t # Calculate total rotation angle
    
    # Create empty composite k-space
    ft_org = calculate_2dft(image)
    comp_fr_image = np.zeros_like(ft_org)
    
    for i in range(N):
        angle = i * deg / (N-1) # Determine the rotation angle for each step
        
        # Apply rotation to the reference image
        image_rotated = ndimage.rotate(image, angle, order = 5, reshape=False)
        
        # Apply original contrast window
        image_rotated[image_rotated < image.min()] = image.min()
        image_rotated[image_rotated > image.max()] = image.max()

        ft = calculate_2dft(image_rotated) # Apply FFT to moved image
        comp_fr_image[i,:] = ft[i,:] # Add sample to comp. k-space
        
        fr_plot(image_rotated,comp_fr_image) # Plot results 
        
    print('Rotation: {0:.2f} degrees'.format(angle))

def altered_k_periphery_rotation(slice_nr, slice_plane, speed, begin = 72, end = 144):
    """
    Simulates the ringing artefacts corresponding with the occurrence of gradual subject rotation during the sampling of the k-space periphery.
    The composite k-space gets updated line-by-line during the sampling of the 
    top and bottom periphery, the centre is directly sampled from the k-space of the image in that position at that time
    
    Returns:
        - an updated plot with the reference image in its
          current position, the composite k-space and the motion-affected image corresponding
          with the composite k-space.
    Prints:
        - the total rotation angle
    
    
    Input:
        - slice_nr (int): slice number corresponding to the desired slice
        - slice_plane (str): the desired slice plane, either transversal (trans),
                             sagittal (sag) or coronal (cor)
        - speed (float): the desired rotation speed in degree/s
        - begin (int): determines the first line of the assumed k-space centre,
                       default is set to 72 (at one third of the total k-space)
        - end (int): determines the final line of the assumed k-space centre,
                       default is set to 144 (at two thirds of the total k-space)
    """    
    model, simulated_image, pars = main() # Initiate image simulation
    
    image = slice_selection(slice_nr, slice_plane) # Select reference image slice from simulation
        
    N, M = image.shape # Determine the dimensions of the image
    
    # Create empty composite k-space
    ft_org = calculate_2dft(image)
    comp_fr_image = np.zeros_like(ft_org)
    
    t = (N - (end - begin)) * pars["TR"] # Calculate amount of time in which motion takes place
    deg = speed * t # Calculate total rotation angle
    
    i = 0 # Initialise step count
    
    for a in range(N-(end-begin)): # Loop over the periphery of the image
        angle = a * deg / (N-1-(end-begin)) # Determine the rotation angle for each step
        
        # Apply rotation to the reference image
        image_rotated = ndimage.rotate(image, angle, reshape=False, order = 5)
        
        # Apply original contrast window
        image_rotated[image_rotated < image.min()] = image.min()
        image_rotated[image_rotated > image.max()] = image.max()

        ft = calculate_2dft(image_rotated) # Apply FFT to moved image
        comp_fr_image[i:i+1,:] = ft[i:i+1,:] # Add sample to comp. k-space
        
        if i+1 == begin: # Check whether centre is reached
            i += end-begin # Update step count
            comp_fr_image[begin:end,:] = ft[begin:end,:] # Update comp. k-space with the samples belonging to the current fixed position
        
        i += 1 # Update step count
        
        fr_plot(image_rotated,comp_fr_image) # Plot results 
    
    print('Rotation: {0:.2f} degrees'.format(deg))
    
def altered_k_centre_rotation(slice_nr, slice_plane, speed, begin = 72, end = 144):
    """
    Simulates the blurring artefacts corresponding with the occurrence of gradual subject rotation during the sampling of the k-space centre.
    The composite k-space gets updated line-by-line during the sampling of the centre, 
    the top periphery is sampled from the k-space of the reference image, the bottom periphery is directly sampled from the k-space of the image in that position at that time
    
    Returns:
        - an updated plot with the reference image in its
          current position, the composite k-space and the motion-affected image corresponding
          with the composite k-space.
    Prints:
        - the total rotation angle
    
    
    Input:
        - slice_nr (int): slice number corresponding to the desired slice
        - slice_plane (str): the desired slice plane, either transversal (trans),
                             sagittal (sag) or coronal (cor)
        - speed (float): the desired rotation speed in degree/s
        - begin (int): determines the first line of the assumed k-space centre,
                       default is set to 72 (at one third of the total k-space)
        - end (int): determines the final line of the assumed k-space centre,
                       default is set to 144 (at two thirds of the total k-space)
    """       
    model, simulated_image, pars = main() # Initiate image simulation
    
    image = slice_selection(slice_nr, slice_plane) # Select reference image slice from simulation
        
    N, M = image.shape # Determine the dimensions of the image

    # Create empty composite k-space
    ft_org = calculate_2dft(image)
    comp_fr_image = np.zeros_like(ft_org)
    
    comp_fr_image[:begin,:] = ft_org[:begin,:] # Update comp. k-space with the samples belonging to the original position
    
    t = (end - begin) * pars["TR"] # Calculate amount of time in which motion takes place
    deg = speed * t # Calculate total rotation angle
    
    i = begin # Initialise step count
    
    for a in range(end-begin): # Loop over the centre of k-space
        
        angle = a * deg / (end-begin-1) # Determine the rotation angle for each step
        
        # Apply rotation to the reference image
        image_rotated = ndimage.rotate(image, angle, reshape=False, order = 5)
        
        # Apply original contrast window
        image_rotated[image_rotated < image.min()] = image.min()
        image_rotated[image_rotated > image.max()] = image.max()

        ft = calculate_2dft(image_rotated) # Apply FFT to moved image
        comp_fr_image[i:i+1,:] = ft[i:i+1,:] # Add sample to comp. k-space
        
        i += 1 # Update step count
        
        fr_plot(image_rotated,comp_fr_image) # Plot results 
    
    comp_fr_image[end:,:] = ft[end:,:] # Update comp. k-space with the samples belonging to the final fixed position

    fr_plot(image_rotated,comp_fr_image) # Plot results 

    print('Rotation: {0:.2f} degrees'.format(deg))
    
def periodic_motion(slice_nr, slice_plane, direction, A, f):
    """
    Simulates the artefacts corresponding with the occurrence of periodic motion, specifically discrete ghosts.
    The applied motion is gradual, so the composite k-space gets updated line-by-line.
     
    Returns:
        - an updated plot with the reference image in its
          current position, the composite k-space and the motion-affected image corresponding
          with the composite k-space.
    Prints:
        - the amplitude in cm
        - the amount of sampled lines at which one motion period takes place
        - the expected ghost separation distance (SEP)

    Input:
        - slice_nr (int): slice number corresponding to the desired slice
        - slice_plane (str): the desired slice plane, either transversal (trans),
                             sagittal (sag) or coronal (cor)
        - direction (str): the desired motion direction (either x or y)
        - A (float): the desired amplitude of the motion in pixels
        - f (float): the desired motion frequency in Hz
        
    """    
    model, simulated_image, pars = main() # Initiate image simulation
    
    image = slice_selection(slice_nr, slice_plane) # Select reference image slice from simulation
        
    N, M = image.shape # Determine the dimensions of the image
    
    # Create empty composite k-space      
    ft_org = calculate_2dft(image)
    comp_fr_image = np.zeros_like(ft_org)
    
    for i in range(N):
                
        harmonic = A * np.sin(2 * np.pi * f * pars["TR"] * i) # Determine position within periodic movement
        
        # Determine translation direction 
        if direction == 'y':
            y = harmonic
            x = 0
        elif direction == 'x':
            y = 0
            x = harmonic
        
        # Determine transformation matrix
        mat_identity = np.array([[1, 0, y], [0, 1, x], [0, 0, 1]])
        
        # Apply transformation matrix to reference image
        image_translated = ndimage.affine_transform(image, np.linalg.inv(mat_identity), order = 5)
        
        # Apply original contrast window
        image_translated[image_translated < image.min()] = image.min()
        image_translated[image_translated > image.max()] = image.max()

        ft = calculate_2dft(image_translated) # Apply FFT to moved image
        comp_fr_image[i,:] = ft[i,:] # Add sample to comp. k-space
                
        fr_plot(image_translated,comp_fr_image) # Plot results 
    
    # Determine the pixel size in the chosen motion direction
    if direction == 'y':
        DimRes = model.Y_dim_res[0][0]
    elif direction == 'x':
        DimRes = model.X_dim_res[0][0]
        
    # Calculate expected ghost separation distance
    SEP = f * N * pars["TR"] * model.Y_dim_res[0][0]
 
    print('Amplitude: {0:.2f} cm'.format(A*DimRes*100))
    print('Repetitition every {0:.2f} lines'.format(1/(f*pars["TR"])))
    print('Expected SEP: {0:.1f} mm'.format(SEP*1000))

def sudden_rotation(slice_nr, slice_plane, speed, motion_line):
    """
    Simulates the artefacts corresponding with the occurrence of sudden subject rotation 
    at one user-defined time point during k-space sampling. 
    
    Returns:
        - an updated plot with the reference image in its
          current position, the composite k-space and the motion-affected image corresponding
          with the composite k-space.
    Prints:
        - the total rotation angle
    

    Input:
        - slice_nr (int): slice number corresponding to the desired slice
        - slice_plane (str): the desired slice plane, either transversal (trans),
                             sagittal (sag) or coronal (cor)
        - speed (float): the desired rotation speed in degree/s
        - morion_line: the line in k-space at which the rotation is desired to take place
    """
    model, simulated_image, pars = main() # Initiate image simulation
    
    image = slice_selection(slice_nr, slice_plane) # Select reference image slice from simulation
        
    N, M = image.shape # Determine the dimensions of the image
    
    # Create empty composite k-space
    ft_org = calculate_2dft(image)
    comp_fr_image = np.zeros_like(ft_org)
    
    angle = speed * pars["TR"] # Calculate rotation angle
    
    # Apply rotation to the reference image
    image_rotated = ndimage.rotate(image, angle, reshape=False)
    
    # Apply original contrast window
    image_rotated[image_rotated < image.min()] = image.min()
    image_rotated[image_rotated > image.max()] = image.max()
    
    ft = calculate_2dft(image_rotated) # Apply FFT to moved image
    
    comp_fr_image[:motion_line,:] = ft_org[:motion_line,:] # Update comp. k-space with the samples belonging to the original position
    fr_plot(image,comp_fr_image) # Plot results 
    
    comp_fr_image[motion_line:,:] = ft[motion_line:,:] # Update comp. k-space with the samples belonging to the final position
    fr_plot(image_rotated,comp_fr_image) # Plot results 
                
    print('Rotation: {0:.2f} degrees'.format(angle))
    

def sudden_translation(slice_nr, slice_plane, speed, angle, motion_line):
    """
    Simulates the artefacts corresponding with the occurrence of sudden subject translation 
    at one user-defined time point during k-space sampling. 
    
    Returns:
        - an updated plot with the reference image in its
          current position, the composite k-space and the motion-affected image corresponding
          with the composite k-space.
    Prints:
        - the total translation in x and y-direction and its total magnitude
    

    Input:
        - slice_nr (int): slice number corresponding to the desired slice
        - slice_plane (str): the desired slice plane, either transversal (trans),
                             sagittal (sag) or coronal (cor)
        - speed (float): the desired translation speed in pixel/s
        - angle (float): the desired translation angle in degrees
        - morion_line: the line in k-space at which the translation is desired to take place
    """
    model, simulated_image, pars = main() # Initiate image simulation
    
    image = slice_selection(slice_nr, slice_plane) # Select reference image slice from simulation
        
    N, M = image.shape # Determine the dimensions of the image
    
    # Create empty composite k-space
    ft_org = calculate_2dft(image)
    comp_fr_image = np.zeros_like(ft_org)
    
    y = speed * np.sin(np.pi * angle/180) * pars["TR"] # Calculate total displacement in y-direction
    x = speed * np.cos(np.pi * angle/180) * pars["TR"] # Calculate total displacement in x-direction
    
    # Determine transformation matrix
    mat_identity = np.array([[1, 0, y], [0, 1, x], [0, 0, 1]])
    
    # Apply transformation matrix to reference image
    image_translated = ndimage.affine_transform(image, np.linalg.inv(mat_identity))
    
    # Apply original contrast window
    image_translated[image_translated < image.min()] = image.min()
    image_translated[image_translated > image.max()] = image.max()
    
    ft = calculate_2dft(image_translated) # Apply FFT to moved image
    
    comp_fr_image[:motion_line,:] = ft_org[:motion_line,:] # Update comp. k-space with the samples belonging to the original position
    fr_plot(image,comp_fr_image) # Plot results 
    
    comp_fr_image[motion_line:,:] = ft[motion_line:,:] # Update comp. k-space with the samples belonging to the final position
    fr_plot(image_translated,comp_fr_image) # Plot results 
        
    # Determine pixel dimensions 
    XDimRes = model.X_dim_res[0][0] 
    YDimRes = model.Y_dim_res[0][0]
    
    # Calculate the total translation in x and y-direction in cm
    x_cm = x * XDimRes * 100
    y_cm = y * YDimRes * 100
    output_distance = np.sqrt(x_cm**2 + y_cm**2) # Calculation total translation 
    
    print('Translation: {0:.2f} cm in x-direction, {1:.2f} cm in y-direction'.format(x_cm, y_cm))
    print('Total translation: {0:.2f} cm'.format(output_distance))


# Some examples of simulations are shown below
# Uncomment one of the functions and run the code to see its outcome
   
#periodic_motion(100,'trans','y',5,0.25)
#gradual_translation(100, 'trans', 0.134,-45)
#abrupt_rotation(100, 'trans',0.1072,6)
#sudden_translation(100, 'trans', 16.368, 45, 10)
#altered_k_centre_rotation(100,'trans',0.241)




