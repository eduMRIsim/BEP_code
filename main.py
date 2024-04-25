import numpy as np
from scipy.io import loadmat
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

def main():
    data_dict = load_data("resources/BrainHighResolution.mat") #This functions loads the anatomical model from the .mat file and returns a dictionary of the data contained in the anatomical model. The 16 data fields contained in data_dict are described in https://mrilab.sourceforge.net/manual/MRiLab_User_Guide_v1_3/MRiLab_User_Guidech3.html#x8-120003.1

    print(data_dict.keys()) 

    model = Model("BrainHighResolution", data_dict["T1"], data_dict["T2"], data_dict["T2Star"], data_dict["Rho"])

    # You can view a transverse slice of the T1 map by using the following code:
    plt.figure()
    plt.imshow(model.T1map[:, :, 90], cmap='gray')
    plt.title("Model")
    plt.show() # TYou will need to close the image before the rest of the code executes. 

    # Play around with the code to view the rest of the data. Think about the following: which anatomical direction (e.g. from left to right/right to left, top to bottom/bottom to top, front to back/back to front) do the x, y and z axes correspond to?
    

    # Simulate a scan
    scan_parameters = {
        "TE": 0.014,
        "TR": 0.864,
        "TI": 0,
    }

    simulated_image = scan(scan_parameters, model)
    # You can view a transverse slice of the simulated image by using the following code:
    plt.figure()
    plt.imshow(simulated_image[:, :, 90], cmap='gray')
    plt.title("Simulated Image")
    plt.show()

    # Play around with the code to view the rest of the simulated image. What if you want to view a coronal or sagittal slice of the simulated image? You could also try simulating the image with different scan parameters. Try to generate a T1-weighted image, a T2-weighted image, a proton density-weighted image, etc. 

    # Also think about...what if you want to see the k-space data? How would you do that?

if __name__ == "__main__":
    main()