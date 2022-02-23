label_inputs.txt, label_strels.txt, and label_results.txt are test
vectors generated using ndimage.label from scipy version 0.10.0, and
are used to verify that the cython version behaves as expected.  The
script to generate them is in ../../utils/generate_label_testvectors.py 
