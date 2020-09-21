
"""lola.py - Implementation of the cola layer using pytorch
Author: Kevin Greif
Last updated 3/26/19
Language: python3
"""

import numpy as np

import torch
import torch.nn as nn


########

### Lola ###

########

class Lola(nn.Module):

    def __init__(self,
                 input_shape,
                 es=0,
                 xs=0,
                 ys=0,
                 zs=0,
                 ms=1,
                 pts=1,
                 dls=0,
                 n_train_es=1,
                 n_train_ms=0,
                 n_train_pts=0,
                 n_train_sum_dijs=2,
                 n_train_min_dijs=2,
                 device=None,
                 **kwargs):
        """
        Arguments:
        input_shape (tuple): (n_features, n_particles)
            es,xs,ys,zs (int): energy, x, y, z
            ms,pts,dls (int): mass, pt, and dl minkowski distance
            n_train_es (int): Number of trainable parameters for energy
            n_train_ms (int): Number of trainable parameters for mass
            n_train_pts (int): Number of trainable parameters for pt
            n_train_sum_dijs (int): Number of trainable parameters for sum_dij
            n_train_min_dijs (int): Number of trainable parameters for min_dij
        """

        # Set all arguments to member variables
        self.es = es
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.ms = ms
        self.pts = pts
        self.dls = dls
        self.n_train_es = n_train_es
        self.n_train_ms = n_train_ms
        self.n_train_pts = n_train_pts
        self.n_train_sum_dijs = n_train_sum_dijs
        self.n_train_min_dijs = n_train_min_dijs

        # Find total trains
        self.total_trains = (n_train_es + n_train_ms + n_train_pts +
                             n_train_sum_dijs + n_train_min_dijs)

        # Extract dimensions from input_shape
        self.n_particles = input_shape[2]
        self.n_features = input_shape[1]

        # Find nout
        self.nout = (int(self.xs)  +
                     int(self.ys)  +
                     int(self.zs)  +
                     int(self.ms)  +
                     int(self.es)  +
                     int(self.pts) +
                     int(self.dls) +
                     self.total_trains)

        # Call inheritance
        super(Lola, self).__init__(**kwargs)

        # Print out layer input information
        print("We have n_features={0} / n_particles={1}".format(
            self.n_features, self.n_particles))

        ################# Layer Initialization ##############

        if self.total_trains > 0:
            temp_w = torch.empty(self.total_trains,
                                 self.n_particles,
                                 self.n_particles,
                                 dtype=torch.double,
                                 device=device).uniform_(-0.05, 0.05)

            self.w = nn.Parameter(temp_w, requires_grad=True)


    def forward(self, x, device=torch.device('cuda'), **kwargs):
        """
        Arguments:
            x (tensor) - layer input in form (b,f,p)
                (batch_size, features, particles)
        """

        # Initialize constants
        weight_index = 0
        out_features = []

        # Build metric (ignore learn metric case)
        metric_vector = [-1., 1., 1., 1.]
        # Ignore extra features if present
        if self.n_features > 4:
            metric_vector.extend([0.] * (self.n_features - 4))
        # Send metric to pytorch tensor
        self.metric = torch.as_tensor(metric_vector,
                                      dtype=torch.double,
                                      device=device)

        # First parse input into helpful matrices, pull vectors of n_features
        # Es, Xs, Ys, and Zs
        Es = x[:, 0, :]
        Xs = x[:, 1, :]
        Ys = x[:, 2, :]
        Zs = x[:, 3, :]

        # Element wise square of inputs
        x2 = torch.pow(x, 2)

        # Mass^2 and transverse momentum
        Ms = torch.tensordot(x2, self.metric, dims=([1], [0]))

        Pts = torch.abs(torch.sqrt(x2[:, 1, :] + x2[:, 2, :]))

        # Append certain features to out_features, (4 vectors are getting
        # written to a 1d array. Is this a problem?)
        if self.es:
            out_features.append(Es)
        if self.xs:
            out_features.append(Xs)
        if self.ys:
            out_features.append(Ys)
        if self.zs:
            out_features.append(Zs)

        if self.ms:
            out_features.append(Ms)
        if self.pts:
            out_features.append(Pts)

        # Find difference to leading particle
        if self.dls:
            ex_dims = torch.unsqueeze(x[:, :, 0], -1) # Adds an extra dimension
            repeated = ex_dims.repeat(1, 1, self.n_particles) # Repeats 4
            # vectors into new dimension
            dl = torch.pow(x - repeated, 2) # Find difference between x and
            # first 4 vector, and square
            # Multiply by metric
            dl = torch.tensordot(dl, self.metric, dims=([1], [0]))
            # Finally append to out_features
            out_features.append(dl)

        # Loop through w matrix multiplying slice of 3d tensor by Es and
        # append to out_features
        for i in range(self.n_train_es):
            out_features.append(torch.tensordot(Es, self.w[weight_index, :, :],
                                                dims=([1], [0])))
            weight_index += 1

        # Same loop as above, but this time for Ms
        for i in range(self.n_train_ms):
            out_features.append(torch.tensordot(Ms, self.w[weight_index, :, :],
                                                dims=([1], [0])))
            weight_index += 1

        # Same loop as above, but this time for Pts
        for i in range(self.n_train_pts):
            out_features.append(torch.tensordot(Pts, self.w[weight_index, :, :],
                                                dims=([1], [0])))
            weight_index += 1


        # Create magic matrices for taking sums/differences
        eye1_np = np.identity(self.n_particles)
        magic1_np = np.repeat(eye1_np, self.n_particles, axis=1)
        magic1 = torch.as_tensor(magic1_np, dtype=torch.double, device=device)

        eye2_np = np.identity(self.n_particles)
        magic2_np = np.tile(eye2_np, self.n_particles)
        magic2 = torch.as_tensor(magic2_np, dtype=torch.double, device=device)

        magic_diff = magic1 - magic2

        # x * magic  gives b f p^2, reshape to b f p p'
        x_mag = torch.matmul(x, magic_diff)
        x_mag_dim = torch.unsqueeze(x_mag, -1)

        d2_ij = torch.reshape(x_mag_dim, (x.shape[0], x.shape[1],
                                          x.shape[2], x.shape[2]))
        # square elements
        d2_ij = torch.pow(d2_ij, 2)

        # fold with the metric
        # b f p p' * f  = b p p'
        for i in range(self.n_train_sum_dijs):

            m_d2_ij = torch.tensordot(d2_ij, self.metric, dims=([1], [0]))

            m_d2_ij_x = torch.tensordot(m_d2_ij, self.w[weight_index, :, :],
                                        dims=([1], [0]))
            max_features = torch.sum(m_d2_ij_x, 2)
            out_features.append(max_features)
            weight_index += 1


        # And do the same for minima

        # b f p p'
        # x * magic  gives b f p^2, reshape to b f p p'
        x_mag = torch.matmul(x, magic_diff)
        x_mag_dim = torch.unsqueeze(x_mag, -1)

        m2_ij = torch.reshape(x_mag_dim, (x.shape[0], x.shape[1],
                                          x.shape[2], x.shape[2]))
        # square elements
        m2_ij = torch.pow(m2_ij, 2)

        # fold with the metric
        # b f p p' * f  = b p p'

        for i in range(self.n_train_min_dijs):

            m_m2_ij = torch.tensordot(m2_ij, self.metric, dims=([1], [0]))

            m_m2_ij_x = torch.tensordot(m_m2_ij, self.w[weight_index, :, :],
                                        dims=([1], [0]))
            min_features = torch.min(m_m2_ij_x, 2)[0]
            out_features.append(min_features)
            weight_index += 1


        # Create results stack and return
        results = torch.stack(out_features, dim=1)

        return results

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nout, input_shape[2])
