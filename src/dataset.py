# adapted from scikit-multilearn
# https://github.com/scikit-multilearn/

import arff
import numpy as np


class Dataset(object):
    @classmethod
    def load_arff(cls, filename, n_labels, endian = "big", input_feature_type = 'float', encode_nominal = True):
        """Method for loading ARFF files as numpy array
        Parameters
        ----------
        filename : string
            Path to ARFF file
        n_labels: integer
            Number of labels in the ARFF file
        endian: string{"big", "little"}
            Whether the ARFF file contains labels at the beginning of the attributes list ("big" endianness, MEKA format) 
            or at the end ("little" endianness, MULAN format)
        input_feature_type: numpy.type as string
            The desire type of the contents of the return 'X' array-likes, default 'i8', 
            should be a numpy type, see http://docs.scipy.org/doc/numpy/user/basics.types.html
        encode_nominal: boolean
            Whether convert categorical data into numeric factors - required for some scikit classifiers that can't handle non-numeric input featuers.
        Returns
        -------
        
        data: dictionary {'X': numpy matrix with input_feature_type elements, 'y': numpy matrix of binary (int8) label vectors }
            The dictionary containing the data frame, with 'X' key storing the input space array-like of input feature vectors
            and 'y' storing labels assigned to each input vector, as a binary indicator vector (i.e. if 5th position has value 1
            then the input vector has label no. 5)
        """
        matrix = None
        arff_frame = arff.load(open(filename,'r'), encode_nominal = encode_nominal, return_type=arff.DENSE)
        matrix = np.array(arff_frame['data']).astype(input_feature_type)

        X, Y = None, None
        data = {}
        
        if endian == "big":
            X, Y = matrix[:,n_labels:], matrix[:,:n_labels].astype(int)
        elif endian == "little":
            X, Y = matrix[:,:-n_labels], matrix[:,-n_labels:].astype(int)

        data['X'] = X
        data['Y'] = Y

        return data

    @classmethod
    def dump_data_arff(cls, original_filename, destination_filename, X, Y):
        # dump always in big endian
        new_data = np.concatenate((Y,X), axis=1)
        arff_frame = arff.load(open(original_filename,'r'), encode_nominal = True, return_type=arff.DENSE)
        arff_frame['data'] = new_data.astype(int).tolist()
        f = open(destination_filename,"w")
        arff.dump(arff_frame, f)
        f.close()

    @classmethod
    def arff_to_big_endian(cls, filename, dataset, n_labels):

        data = Dataset.load_arff(filename, n_labels, endian = "little", input_feature_type = 'float', encode_nominal = True)
        new_data = np.concatenate((data['Y'],data['X']), axis=1)

        arff_frame = arff.load(open(filename,'r'), encode_nominal = True, return_type=arff.DENSE)

        arff_frame['data'] = new_data.tolist()
        # make the labels nominal
        for i in range(data['Y'].shape[0]):
            for j in range(data['Y'].shape[1]):
                arff_frame['data'][i][j] = int(arff_frame['data'][i][j])

        arff_frame['attributes'] = arff_frame['attributes'][-n_labels:] + arff_frame['attributes'][:-n_labels]

        # nominal attributes to int format
        attributes = arff_frame['attributes']
        for j in range(data['Y'].shape[1], data['X'].shape[1] + data['Y'].shape[1]):
            if isinstance(attributes[j][1], list):
                for i in range(data['Y'].shape[0]):
                    arff_frame['data'][i][j] = int(arff_frame['data'][i][j])

                    

        arff_frame['relation'] = dataset + "_mlcsn: -C " + str(n_labels)
        f = open(filename,"w")
        arff.dump(arff_frame, f)
        f.close()

