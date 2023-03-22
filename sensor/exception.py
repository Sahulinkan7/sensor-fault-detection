import os,sys

class SensorException(Exception):
    def __init__(self,error_message:Exception,error_details:sys):
        '''
        error_message : an Exception object
        error_details : from sys module
        '''
        super().__init__(error_message)
        self.error_message=SensorException.get_error_details(error_message=error_message,error_details=error_details)
        
        
    @staticmethod
    def get_error_details(error_message,error_details):
        e_type,e_val,exec_tb=error_details.exc_info()
        file_name=exec_tb.tb_frame.f_code.co_filename
        try_block_line_number=exec_tb.tb_lineno
        exception_block_line_number=exec_tb.tb_frame.f_lineno
        
        error_message=f"""
        Error occurred in script : [{file_name}]
        at try block line number : [{try_block_line_number}] and exception block line number : [{exception_block_line_number}]
        error message : [{error_message}]
        """
        return error_message
    
    def __str__(self):
        return self.error_message