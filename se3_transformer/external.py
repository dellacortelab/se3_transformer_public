import numpy as np
import torch
import lie_learn
import math

# From SE3CNN
# https://github.com/mariogeiger/se3cnn

def get_spherical_from_cartesian_torch(cartesian, divide_radius_by=1.0):

    ###################################################################################################################
    # ON ANGLE CONVENTION
    #
    # sh has following convention for angles:
    # :param theta: the colatitude / polar angle, ranging from 0(North Pole, (X, Y, Z) = (0, 0, 1)) to pi(South Pole, (X, Y, Z) = (0, 0, -1)).
    # :param phi: the longitude / azimuthal angle, ranging from 0 to 2 pi.
    #
    # the 3D steerable CNN code therefore (probably) has the following convention for alpha and beta:
    # beta = pi - theta; ranging from 0(South Pole, (X, Y, Z) = (0, 0, -1)) to pi(North Pole, (X, Y, Z) = (0, 0, 1)).
    # alpha = phi
    #
    ###################################################################################################################

    # initialise return array
    # ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    spherical = torch.zeros_like(cartesian)

    # indices for return array
    ind_radius = 0
    ind_alpha = 1
    ind_beta = 2

    cartesian_x = 2
    cartesian_y = 0
    cartesian_z = 1

    # get projected radius in xy plane
    # xy = xyz[:,0]**2 + xyz[:,1]**2
    r_xy = cartesian[..., cartesian_x] ** 2 + cartesian[..., cartesian_y] ** 2

    # get second angle
    # version 'elevation angle defined from Z-axis down'
    spherical[..., ind_beta] = torch.atan2(torch.sqrt(r_xy), cartesian[..., cartesian_z])
    # ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2])
    # version 'elevation angle defined from XY-plane up'
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy))
    # spherical[:, ind_beta] = np.arctan2(cartesian[:, 2], np.sqrt(r_xy))

    # get angle in x-y plane
    spherical[...,ind_alpha] = torch.atan2(cartesian[...,cartesian_y], cartesian[...,cartesian_x])

    # get overall radius
    # ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    if divide_radius_by == 1.0:
        spherical[..., ind_radius] = torch.sqrt(r_xy + cartesian[...,cartesian_z]**2)
    else:
        spherical[..., ind_radius] = torch.sqrt(r_xy + cartesian[...,cartesian_z]**2)/divide_radius_by

    return spherical

def pochhammer(x, k):
    """Compute the pochhammer symbol (x)_k.
    (x)_k = x * (x+1) * (x+2) *...* (x+k-1)
    Args:
        x: positive int
    Returns:
        float for (x)_k
    """
    xf = float(x)
    for n in range(x+1, x+k):
        xf *= n
    return xf
    
def semifactorial(x):
    """Compute the semifactorial function x!!.
    x!! = x * (x-2) * (x-4) *...
    Args:
        x: positive int
    Returns:
        float for x!!
    """
    y = 1.
    for n in range(x, 1, -2):
        y *= n
    return y

class SphericalHarmonics(object):
    def __init__(self):
        self.leg = {}

    def clear(self):
        self.leg = {}

    def negative_lpmv(self, l, m, y):
        """Compute negative order coefficients"""
        if m < 0:
            y *= ((-1)**m / pochhammer(l+m+1, -2*m))
        return y

    def lpmv(self, l, m, x):
        """Associated Legendre function including Condon-Shortley phase.
        Args:
            m: int order 
            l: int degree
            x: float argument tensor
        Returns:
            tensor of x-shape
        """
        # Check memoized versions
        m_abs = abs(m)
        if (l,m) in self.leg:
            return self.leg[(l,m)]
        elif m_abs > l:
            return None
        elif l == 0:
            self.leg[(l,m)] = torch.ones_like(x)
            return self.leg[(l,m)]
        
        # Check if on boundary else recurse solution down to boundary
        if m_abs == l:
            # Compute P_m^m
            y = (-1)**m_abs * semifactorial(2*m_abs-1)
            y *= torch.pow(1-x*x, m_abs/2)
            self.leg[(l,m)] = self.negative_lpmv(l, m, y)
            return self.leg[(l,m)]
        else:
            # Recursively precompute lower degree harmonics
            self.lpmv(l-1, m, x)

        # Compute P_{l}^m from recursion in P_{l-1}^m and P_{l-2}^m
        # Inplace speedup
        y = ((2*l-1) / (l-m_abs)) * x * self.lpmv(l-1, m_abs, x)
        if l - m_abs > 1:
            y -= ((l+m_abs-1)/(l-m_abs)) * self.leg[(l-2, m_abs)]
        #self.leg[(l, m_abs)] = y
        
        if m < 0:
            y = self.negative_lpmv(l, m, y)
        self.leg[(l,m)] = y

        return self.leg[(l,m)]

    def get_element(self, l, m, theta, phi):
        """Tesseral spherical harmonic with Condon-Shortley phase.
        The Tesseral spherical harmonics are also known as the real spherical
        harmonics.
        Args:
            l: int for degree
            m: int for order, where -l <= m < l
            theta: collatitude or polar angle
            phi: longitude or azimuth
        Returns:
            tensor of shape theta
        """
        assert abs(m) <= l, "absolute value of order m must be <= degree l"

        N = np.sqrt((2*l+1) / (4*np.pi))
        leg = self.lpmv(l, abs(m), torch.cos(theta))
        if m == 0:
            return N*leg
        elif m > 0:
            Y = torch.cos(m*phi) * leg
        else:
            Y = torch.sin(abs(m)*phi) * leg
        N *= np.sqrt(2. / pochhammer(l-abs(m)+1, 2*abs(m)))
        Y *= N
        return Y

    def get(self, l, theta, phi, refresh=True):
        """Tesseral harmonic with Condon-Shortley phase.
        The Tesseral spherical harmonics are also known as the real spherical
        harmonics.
        Args:
            l: int for degree
            theta: collatitude or polar angle
            phi: longitude or azimuth
        Returns:
            tensor of shape [*theta.shape, 2*l+1]
        """
        results = []
        if refresh:
            self.clear()
        for m in range(-l, l+1):
            results.append(self.get_element(l, m, theta, phi))
        return torch.stack(results, -1)

def precompute_sh(r_ij, max_J):
    """
    pre-comput spherical harmonics up to order max_J
    :param r_ij: relative positions
    :param max_J: maximum order used in entire network
    :return: dict where each entry has shape [B,N,K,2J+1]
    """
    
    i_distance = 0
    i_alpha = 1
    i_beta = 2

    Y_Js = {}
    sh = SphericalHarmonics()

    for J in range(max_J+1):
        # dimension [B,N,K,2J+1]
        #Y_Js[J] = spherical_harmonics(order=J, alpha=r_ij[...,i_alpha], beta=r_ij[...,i_beta])
        Y_Js[J] = sh.get(J, theta=math.pi-r_ij[...,i_beta], phi=r_ij[...,i_alpha], refresh=False)

    sh.clear()
    return Y_Js

def irr_repr(order, alpha, beta, gamma, dtype=None):
    """
    irreducible representation of SO3
    - compatible with compose and spherical_harmonics
    """
    # from from_lielearn_SO3.wigner_d import wigner_D_matrix
    from lie_learn.representations.SO3.wigner_d import wigner_D_matrix
    # if order == 1:
    #     # change of basis to have vector_field[x, y, z] = [vx, vy, vz]
    #     A = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    #     return A @ wigner_D_matrix(1, alpha, beta, gamma) @ A.T

    # TODO (non-essential): try to do everything in torch
    # return torch.tensor(wigner_D_matrix(torch.tensor(order), alpha, beta, gamma), dtype=torch.get_default_dtype() if dtype is None else dtype)
    return torch.tensor(wigner_D_matrix(order, np.array(alpha), np.array(beta), np.array(gamma)), dtype=torch.get_default_dtype() if dtype is None else dtype)


def kron(a, b):
    """
    A part of the pylabyk library: numpytorch.py at https://github.com/yulkang/pylabyk
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)

def get_matrix_kernel(A, eps=1e-10):
    '''
    Compute an orthonormal basis of the kernel (x_1, x_2, ...)
    A x_i = 0
    scalar_product(x_i, x_j) = delta_ij
    :param A: matrix
    :return: matrix where each row is a basis vector of the kernel of A
    '''
    _u, s, v = torch.svd(A)

    # A = u @ torch.diag(s) @ v.t()
    kernel = v.t()[s < eps]
    return kernel


def get_matrices_kernel(As, eps=1e-10):
    '''
    Computes the commun kernel of all the As matrices
    '''
    return get_matrix_kernel(torch.cat(As, dim=0), eps)
    
class torch_default_dtype:

    def __init__(self, dtype):
        self.saved_dtype = None
        self.dtype = dtype

    def __enter__(self):
        self.saved_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_default_dtype(self.saved_dtype)

def basis_transformation_Q_J(J, order_in, order_out, version=3):  # pylint: disable=W0613
    """
    :param J: order of the spherical harmonics
    :param order_in: order of the input representation
    :param order_out: order of the output representation
    :return: one part of the Q^-1 matrix of the article
    """
    with torch_default_dtype(torch.float64):
        def _R_tensor(a, b, c): return kron(irr_repr(order_out, a, b, c), irr_repr(order_in, a, b, c))

        def _sylvester_submatrix(J, a, b, c):
            ''' generate Kronecker product matrix for solving the Sylvester equation in subspace J '''
            R_tensor = _R_tensor(a, b, c)  # [m_out * m_in, m_out * m_in]
            R_irrep_J = irr_repr(J, a, b, c)  # [m, m]
            return kron(R_tensor, torch.eye(R_irrep_J.size(0))) - \
                kron(torch.eye(R_tensor.size(0)), R_irrep_J.t())  # [(m_out * m_in) * m, (m_out * m_in) * m]

        random_angles = [
            [4.41301023, 5.56684102, 4.59384642],
            [4.93325116, 6.12697327, 4.14574096],
            [0.53878964, 4.09050444, 5.36539036],
            [2.16017393, 3.48835314, 5.55174441],
            [2.52385107, 0.2908958, 3.90040975]
        ]
        null_space = get_matrices_kernel([_sylvester_submatrix(J, a, b, c) for a, b, c in random_angles])
        assert null_space.size(0) == 1, null_space.size()  # unique subspace solution
        Q_J = null_space[0]  # [(m_out * m_in) * m]
        Q_J = Q_J.view((2 * order_out + 1) * (2 * order_in + 1), 2 * J + 1)  # [m_out * m_in, m]
        assert all(torch.allclose(_R_tensor(a, b, c) @ Q_J, Q_J @ irr_repr(J, a, b, c)) for a, b, c in torch.randn(4, 3))

    assert Q_J.dtype == torch.float64
    return Q_J  # [m_out * m_in, m]
