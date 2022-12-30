import sys,os
from sensor.logger import logging

def error_message_details(error,error_details:sys):
    _,_,exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"error occured at python script [{file_name}] line number [{exc_tb.tb_lineno}] message [{error}]"
    return error_message


class SensorException(Exception):
    def __init__(self,error_message,error_details:sys):
        self.error_message = error_message_details(error_message, error_details)
        self.error_details = error_details


    def __str__(self):
        return self.error_message
