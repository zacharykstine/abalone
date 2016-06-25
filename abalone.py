"""****************************************************************************************
# Created by Zachary Stine 2016-06-25
#
# Description: machine learning program for predicting the age of an abalone using
# the linear regression to adjust weights based on physical measurements.
#
# Data: archive.ics.uci.edu/ml/datasets/Abalone
# [sex, length, diameter, height, whole weight, shucked weight, viscera weight, shell weight, rings]
# 
# For sex, value can be F, M, or I (infant). Heaton uses one-of-n encoding so that
# [1, 0, 0, ...] = F, 
# [0, 1, 0, ...] = M, and 
# [0, 0, 1, ...] = I. 
#
# This attempt will use least squares fitting from the numpy module. 
****************************************************************************************"""
