import os 
import logging 

class logger(object):
	"""
	set logger
	"""
	def __init__(self, logger_path):
		self.logger = logging.getLogger()
		self.logger.setLevel(logging.INFO)
		self.logfile = logging.FileHandler(logger_path)
		self.logdisplay = logging.StreamHandler()
		self.logger.addHandler(self.logfile)
		self.logger.addHandler(self.logdisplay)

	def get_logger(self):
		return self.logger
