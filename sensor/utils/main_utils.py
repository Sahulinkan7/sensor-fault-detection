from sensor.exception import SensorException
import os,sys
import yaml
import numpy as np
import dill

def read_yaml_file(file_path:str) -> dict:
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise SensorException(e,sys) from e
    

def write_yaml_file(file_path: str,content: object, replace: bool =False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'w') as file:
            yaml.dump(content,file)
            
    except Exception as e:
        raise SensorException(e,sys) from e
    
    
def save_numpy_array_data(file_path: str,array: np.array):
    '''
    saves numpy array data into file
    
    file_path : str location of file
    array : np.array data to save in the file
    '''
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise SensorException(e,sys) from e
    
    
def load_numpy_array_data(file_path: str) ->np.array:
    '''
    loads numpy array data from file
    file_path : str location of file to be
    return : np.array data 
    '''
    try:
        with open(file_path,"rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise SensorException(e,sys) from e
    
def save_object(file_path: str, obj: object) -> None:
    '''
    save object as file
    '''
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise SensorException(e,sys) from e
    
def load_object(file_path: str) ->object:
    '''
    load object from given file path
    '''
    try:
        if not os.path.exists(file_path):
            raise Exception(f" file path {file_path} doesn't exists")
        with open(file_path,'rb') as file_obj:
            dill.load(file_obj)
    except Exception as e:
        raise SensorException(e,sys) from e
