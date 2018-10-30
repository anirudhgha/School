This project will involve programming an pyraMiD-LSTM in order to traige 3d volumetric data. The LSTM parameters
(learning rate, activation function) can be modified to see which fits the data best.

Such ordering networks exist, but this one will attempt to increase accuracy by varying parameters of the traiging
network. For each 3D volumetric data set, the network will run with multiple activation functions. The average severity
index assigned based on multiple parameter tests will be the final value that we use to queue the data. The final
result should be a queue that contains data sets of the highest severity index at the front, and lowest at the back.

This data will be compared against data queued using a single activation function. The sample sizes will be small
(~15-20), so the accuracy of the order can be ascertained visually.




Contributions from each paper
1. Improved image segmentation via Cost Minimization of Multiple Hypotheses
    - Will modify the pyraMiD-LSTM parameters to find which gives the most complete segmentation
2. Parallel Multi-Dimensional LSTM with Application to Fast Biomedical Volumetric Image Segmentation
    - The core of this project will be a pyraMiD-LSTM, following the structure described in this paper
3. Automated deep-neural-network Surveillance of cranial images for acute neurological events
    - The LSTM network will be used to triage non-medical information. "Degrees of severity" will be assigned
      by the researcher based on some criteria to the image set. The LSTM must correctly order the images based
      on that severity index.




Which parameters can we modify in an LSTM?
    1. the tanh can be swapped for a hard-tanh, ReLU
        source: https://stackoverflow.com/questions/40761185/what-is-the-intuition-of-using-tanh-in-lstm
    2.


Bottom line: - I'm looking to increase the accuracy at which 3D medical data can be triaged, an application
               that could cut time where it matters most, and save lives