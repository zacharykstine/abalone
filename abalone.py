"""****************************************************************************************
# Created by Zachary Stine on  2016-06-25
#
# Description: This program uses a type of linear regression called least squares fitting
# to generate a predictive model of an abalone's age based on different physical
# measurements (inspired by a section of _Artificial Intelligence for Humans Vol 1_ by
# Jeff Heaton). This solution relies on the least squares method provided by numpy
# (numpy.linalg.lstsq()). Interestingly, my results differ slightly from the results
# shown on Mr. Heaton's GitHub page, despite both using the same numpy method.
# 
# Data: archive.ics.uci.edu/ml/datasets/Abalone
# [sex, length, diameter, height, whole weight, shucked weight, viscera weight, shell weight, rings]
# 
# For sex, value can be F, M, or I (infant). Heaton uses one-of-n encoding so that
# [1, 0, 0, ...] = F, 
# [0, 1, 0, ...] = M, and 
# [0, 0, 1, ...] = I. 
#
# Resources:
# github.com/jeffheaton/aifh/blob/master/vol1/python-examples/examples/example_linear_regression.py
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html
****************************************************************************************"""

import numpy as np

def main():
    # Import data
    with open('abalone.data', 'r') as in_file:
        # Convert the file into a list of normalized lists
        data = get_normalized_data(in_file)

    # Convert the lists into numpy arrays
    matrix = np.array(data)

    # Create matrix A by selecting the first ten columns of our matrix and append a column of 1s
    observation_matrix = matrix[:, :-1]
    ones_column = np.ones((len(observation_matrix), 1))
    x = np.concatenate((observation_matrix, ones_column), axis=1)
    y = matrix[:, 10:]

    # Perform least squares on coefficient matrix and vector of ideal values
    coefficients = np.linalg.lstsq(x, y)[0]
    print(coefficients)
    print('\n')

    # Evaluate
    for d in data:
        actual = estimate_age(coefficients, d[:-1])
        ideal = d[-1:]
        print('Actual: ' + str(actual) + ' -> Ideal: ' + str(ideal))

def estimate_age(coeffs, d):
    result = 0
    
    for i in range(0, len(d)):
        result += coeffs[i] * d[i]
    result += coeffs[len(coeffs) - 1, 0]
    return result

def get_normalized_data(f):
    rows = []
    for line in f:
        # Get raw data as list of strings
        raw_row = line.strip('\n').split(',')

        # Use one-of-n encoding to represent sex as 3-element list
        s_list = get_one_of_n_list(raw_row[0])

        # Convert all other attributes from strings to floats
        numeric_list = [float(x) for x in raw_row[1:]]

        # Concatenate the encoded sex list with the converted numeric list
        normalized_row = s_list + numeric_list
        
        # Append the normalized row to the collection of rows (a list of lists)
        rows.append(normalized_row)
    return rows

def get_one_of_n_list(sex):
    # Check if observation is F, M, or I
    if sex == 'F':
        return [1.0, 0.0, 0.0]
    elif sex == 'M':
        return [0.0, 1.0, 0.0]
    else:
        return [0.0, 0.0, 1.0]

if __name__ == '__main__':
   main()
