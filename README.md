# Deep Learning for Distributed Channel Feedback and Multiuser Precoding in FDD Massive MIMO

The codes provided here are corresponding to the numerical simulations in the paper entitled 
"Deep Learning for Distributed Channel Feedback and Multiuser Precoding in FDD Massive MIMO," which can be found on the following link:
https://arxiv.org/abs/2007.06512

The codes are partitioned into 5 folders as follows:

1- Generic_Network: This folder provides the codes for the generic deep learning framework for multi-user precoding in FDD massive MIMO systems
presented in Section II of the manuscript. For brevity, only one instance of the trained generic network for parameters
K=2, M=64, L=8, B=30, and L<sub>p</sub> = 2 is inlcuded.

2- Generalizable_Network_wrt_B: This folder provides the codes for the proposed training approach for enhancing the generalizability of
the generic DNN with respect to the feedback rate limit B, presented in Section IV.B of the manuscript. Note that only one instance of the
trained network with the modified-B training scheme for parameters K=2, M=64, L=8, B=30, and L<sub>p</sub> = 2 is included.

3- Generalizable_Nerwork_wrt_K: This folder provides the codes for the proposed training approach in Section IV.C for enhancing
the generalizability of the generic DNN with respect to the number of users K. Note that only one instance of the
trained network with the modified-K training scheme for parameters K=3, M=64, L=8, B=30, and L<sub>p</sub> = 2 is included.

4- Baselines: This folder provides the codes for the baselines described in Section VI of the paper. Note that for the DNN-CE approach, 
only one instance of the trained network with parameters: K=1, M=64, L=8, B=30, and L<sub>p</sub> =2 is included.

5- Plot_Figures: This folder provides the codes and data for replotting the numerical results in th manuscript, presented in Figures 4-10.

If you have any questions, feel free to reach me at fsohrabi@ece.utoronto.ca




