# -----------------------------------------------------------------------------
#
# Copyright (C) 2024 by Nils Schween, Brian Reville
#
#
#
# The python script "matrices.py" is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# It is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# the "matrices.py" script. If not, see <https://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------------

import numpy as np
import scipy as sp
import math
# max. degree of the spherical harmonic expansion
l_max = 2


# coefficient appearing in the matrxi representation of A^x
def c(l, m):
    return np.sqrt((l + m) * (l - m) / ((2 * l + 1) * (2 * l - 1)))


# coefficient appeaering in the matrix representation of A^y and A^z
def a(l, m):
    return np.sqrt((l + m) * (l + m - 1) / ((2 * l + 1) * (2 * l - 1)))


# Wigner-D function of a rotation by pi/2 about the z-axis appaering in the
# matrix representation of e^(iL^z pi/2)


# NOTE: For l_max > 6 this function procudes an overflow of Pythons integer
# type. Use the scipy method for expm to compute the matrix exponential or
# better alogrithm to compute the Wigner-D matrices.
def wigner_d_function(l_prime, m_prime, l, m):
    n = np.min([l - m_prime, l - m])
    matrix_element = 0
    for k in range(n + 1):
        if m_prime + m + k >= 0:
            matrix_element += (-1)**k * np.sqrt(
                math.factorial(l + m_prime) * math.factorial(l - m_prime) *
                math.factorial(l + m) * math.factorial(l - m)
            ) / (math.factorial(k) * math.factorial(l - m_prime - k) *
                 math.factorial(l - m - k) * math.factorial(m_prime + m + k))
    matrix_element *= (-1)**(l - m_prime) / 2**l

    return matrix_element


l_range = range(0, l_max + 1)

# Allocate memory for the arrays holding the representation matrices
system_size = (l_max + 1) * (l_max + 1)

# Representation matrix of the collision operator
C = np.zeros((system_size, system_size))

# Representation matrices of the angular momentum operators
Omegax = np.zeros((system_size, system_size))
Omegay = np.zeros((system_size, system_size))
Omegaz = np.zeros((system_size, system_size), complex)

# Representation matrices of the the direction operators
Ax = np.zeros((system_size, system_size))
Ay = np.zeros((system_size, system_size))
Az = np.zeros((system_size, system_size), complex)

# Representation matrices of the rotation operators
Ux = np.zeros((system_size, system_size), complex)
Uz = np.zeros((system_size, system_size))

for l_prime in l_range:
    for m_prime in range(-l_prime, l_prime + 1):
        i = l_prime * (l_prime + 1) - m_prime  # i starts with zero
        for l in l_range:
            for m in range(-l, l + 1):
                j = l * (l + 1) - m
                # Collision operator representation matrix
                if l_prime == l and m_prime == m:
                    C[i, j] = 0.5 * l * (l + 1)

                # Omega^x representation matrix
                if l_prime == l and m_prime == m:
                    Omegax[i, j] = m
                # Omega^y and Omega^z
                if l_prime == l and m_prime == m + 1:
                    Omegay[i, j] = 0.5 * np.sqrt((l + m + 1) * (l - m))
                    Omegaz[i, j] = -0.5j * np.sqrt((l + m + 1) * (l - m))
                if l_prime == l and m_prime == m - 1:
                    Omegay[i, j] = 0.5 * np.sqrt((l + m) * (l - m + 1))
                    Omegaz[i, j] = 0.5j * np.sqrt((l + m) * (l - m + 1))

                # A^x representation matrix
                if l_prime == l + 1 and m_prime == m:
                    Ax[i, j] = c(l + 1, m)
                if l_prime == l - 1 and m_prime == m:
                    Ax[i, j] = c(l, m)
                # A^y and A^z representation matrices
                if l_prime == l + 1 and m_prime == m + 1:
                    Ay[i, j] = -0.5 * a(l + 1, m + 1)
                    Az[i, j] = 0.5j * a(l + 1, m + 1)
                if l_prime == l - 1 and m_prime == m + 1:
                    Ay[i, j] = 0.5 * a(l, -m)
                    Az[i, j] = -0.5j * a(l, -m)
                if l_prime == l + 1 and m_prime == m - 1:
                    Ay[i, j] = 0.5 * a(l + 1, -m + 1)
                    Az[i, j] = 0.5j * a(l + 1, -m + 1)
                if l_prime == l - 1 and m_prime == m - 1:
                    Ay[i, j] = -0.5 * a(l, m)
                    Az[i, j] = -0.5j * a(l, m)

                # Rotation matrices
                if l_prime == l and m_prime == m:
                    Ux[i, j] = np.exp(1j * m * np.pi / 2)
                if l_prime == l:
                    Uz[i, j] = wigner_d_function(l_prime, m_prime, l, m)

RotatedOmegax = np.dot(Uz.transpose().conjugate(), np.dot(Omegax,
                                                          Uz))  # equals Omegay
RotatedOmegay = np.dot(Ux.transpose().conjugate(), np.dot(Omegay,
                                                          Ux))  # equals Omegaz

RotatedAx = np.dot(Uz.transpose().conjugate(), np.dot(Ax, Uz))  # equals Ay
RotatedAy = np.dot(Ux.transpose().conjugate(), np.dot(Ay, Ux))  # equals Az

# Basis transformation matrix S
S = np.zeros((system_size, system_size), complex)
for l in l_range:
    for m in range(-l, l + 1):
        for l_prime in l_range:
            i = l * (l + 1) - m
            for s_prime in range(2):
                for m_prime in range(s_prime, l_prime + 1):
                    j = l_prime * (l_prime +
                                   1) + (1 if s_prime else -1) * m_prime
                    if l_prime == l and m_prime == m and s_prime == 0:
                        S[i, j] += 1. / np.sqrt(2 * (2 if m_prime == 0 else 1))
                        # NOTE: The += covers the element in the center of the
                        # matrix, i.e. if s_prime = 0 and m_prime = 0 and m = 0
                        # two terms have to be added before writting into the
                        # matrix, in all the other cases only a single term is
                        # evaluated and written to the matrix
                    if l_prime == l and -m_prime == m and s_prime == 0:
                        S[i, j] += (-1)**m_prime / np.sqrt(
                            2 * (2 if m_prime == 0 else 1))
                    if l_prime == l and m_prime == m and s_prime == 1:
                        S[i, j] = -1j / np.sqrt(2)
                    if l_prime == l and -m_prime == m and s_prime == 1:
                        S[i, j] = 1j / np.sqrt(2) * (-1)**m_prime

# Real representation matrices
# TransformedMatrices
TransformedC = np.dot(S.transpose().conjugate(), np.dot(C, S)).real

TransformedOmegax = np.dot(S.transpose().conjugate(), np.dot(1j * Omegax,
                                                             S)).real
TransformedOmegay = np.dot(S.transpose().conjugate(), np.dot(1j * Omegay,
                                                             S)).real
TransformedOmegaz = np.dot(S.transpose().conjugate(), np.dot(1j * Omegaz,
                                                             S)).real

TransformedAx = np.dot(S.transpose().conjugate(), np.dot(Ax, S)).real
TransformedAy = np.dot(S.transpose().conjugate(), np.dot(Ay, S)).real
TransformedAz = np.dot(S.transpose().conjugate(), np.dot(Az, S)).real

TransformedUx = np.dot(S.transpose().conjugate(), np.dot(Ux, S)).real
TransformedUz = np.dot(S.transpose().conjugate(), np.dot(Uz, S)).real

# Direct implementation, see formulae in the Appendix
CR = np.zeros((system_size, system_size))

OmegaxR = np.zeros((system_size, system_size))
OmegayR = np.zeros((system_size, system_size))
OmegazR = np.zeros((system_size, system_size), complex)

AxR = np.zeros((system_size, system_size))
AyR = np.zeros((system_size, system_size))
AzR = np.zeros((system_size, system_size), complex)

UxR = np.zeros((system_size, system_size))
UzR = np.zeros((system_size, system_size))

for l_prime in l_range:
    for s_prime in range(2):
        for m_prime in range(s_prime, l_prime + 1):
            i = l_prime * (l_prime + 1) + (1 if s_prime else -1) * m_prime
            i_complex = l_prime * (l_prime + 1) - m_prime
            for l in l_range:
                for s in range(2):
                    for m in range(s, l + 1):
                        j = l * (l + 1) + (1 if s else -1) * m
                        j_complex = l * (l + 1) - m
                        # Collision operator
                        if l_prime == l and m_prime == m and s_prime == s:
                            CR[i, j] = 0.5 * l * (l + 1)

                        # Angular momentum operator
                        if s_prime == 0 and s == 1:
                            OmegaxR[i, j] = Omegax[i_complex, j_complex] * (
                                1 if m_prime else 1 / np.sqrt(2))
                        if s_prime == 1 and s == 0:
                            OmegaxR[i, j] = Omegax[i_complex, j_complex] * (
                                -1 if m else -1 / np.sqrt(2))

                        if s_prime == 0 and s == 1:
                            OmegayR[i, j] = Omegay[i_complex, j_complex] * (
                                1 if m_prime else 1 / np.sqrt(2))
                            if l_prime == l and m_prime == 0 and m == 1:
                                OmegayR[i,
                                        j] += 1 / (2 * np.sqrt(2)) * np.sqrt(
                                            l * (l + 1))
                        if s_prime == 1 and s == 0:
                            OmegayR[i, j] = Omegay[i_complex, j_complex] * (
                                -1 if m else -1 / np.sqrt(2))
                            if l_prime == l and m_prime == 1 and m == 0:
                                OmegayR[i,
                                        j] -= 1 / (2 * np.sqrt(2)) * np.sqrt(
                                            l * (l + 1))
                        if s_prime == 0 and s == 0:
                            OmegazR[i, j] = 1j * Omegaz[i_complex, j_complex]
                            if l_prime == l and m_prime == 0 and m == 1:
                                OmegazR[i, j] -= 0.5 * np.sqrt(l * (l + 1))
                            if l_prime == l and m_prime == 1 and m == 0:
                                OmegazR[i, j] += 0.5 * np.sqrt(l * (l + 1))
                            if m_prime == 0:
                                OmegazR[i, j] *= 1 / np.sqrt(2)
                            if m == 0:
                                OmegazR[i, j] *= 1 / np.sqrt(2)
                        if s_prime == 1 and s == 1:
                            OmegazR[i, j] = 1j * Omegaz[i_complex, j_complex]

                        # Direction operator
                        if s_prime == s:
                            AxR[i, j] = Ax[i_complex, j_complex]

                        if s_prime == 0 and s == 0:
                            AyR[i, j] = Ay[i_complex, j_complex]
                            if m_prime == 0 and m == 1:
                                if l_prime == l + 1:
                                    AyR[i, j] += 0.5 * a(l + 1, -m + 1)
                                if l_prime == l - 1:
                                    AyR[i, j] -= 0.5 * a(l, m)
                            if m_prime == 1 and m == 0:
                                if l_prime == l + 1:
                                    AyR[i, j] -= 0.5 * a(l + 1, -m + 1)
                                if l_prime == l - 1:
                                    AyR[i, j] += 0.5 * a(l, m)
                            if m_prime == 0:
                                AyR[i, j] *= 1 / np.sqrt(2)
                            if m == 0:
                                AyR[i, j] *= 1 / np.sqrt(2)

                        if s_prime == 1 and s == 1:
                            AyR[i, j] = Ay[i_complex, j_complex]

                        if s_prime == 0 and s == 1:
                            AzR[i, j] = -1j * Az[i_complex, j_complex]
                            if m_prime == 0 and m == 1:
                                if l_prime == l + 1:
                                    AzR[i, j] += 0.5 * a(l + 1, -m + 1)
                                if l_prime == l - 1:
                                    AzR[i, j] -= 0.5 * a(l, m)
                            if m_prime == 0:
                                AzR[i, j] *= 1 / np.sqrt(2)
                        if s_prime == 1 and s == 0:
                            AzR[i, j] = 1j * Az[i_complex, j_complex]
                            if m_prime == 1 and m == 0:
                                if l_prime == l + 1:
                                    AzR[i, j] -= 0.5 * a(l + 1, -m + 1)
                                if l_prime == l - 1:
                                    AzR[i, j] += 0.5 * a(l, m)
                            if m == 0:
                                AzR[i, j] *= 1 / np.sqrt(2)

                        # Rotation operators
                        if l_prime == l and m_prime == m:
                            if s_prime == 0 and s == 0:
                                UxR[i, j] = np.cos(m * np.pi / 2)
                                if m_prime == 0 and m == 0:
                                    UxR[i, j] += 1
                                if m_prime == 0:
                                    UxR[i, j] /= np.sqrt(2)
                                if m == 0:
                                    UxR[i, j] /= np.sqrt(2)
                            if s_prime == 0 and s == 1:
                                UxR[i, j] = np.sin(m * np.pi / 2)
                                if m_prime == 0:
                                    UxR[i, j] /= np.sqrt(2)
                            if s_prime == 1 and s == 0:
                                UxR[i, j] = -np.sin(m * np.pi / 2)
                                if m == 0:
                                    UxR[i, j] /= np.sqrt(2)
                            if s_prime == 1 and s == 1:
                                UxR[i, j] = np.cos(m * np.pi / 2)

                        if l_prime == l:
                            if s_prime == 0 and s == 0:
                                UzR[i, j] = wigner_d_function(
                                    l_prime, m_prime, l,
                                    m) + (-1)**m * wigner_d_function(
                                        l_prime, m_prime, l, -m)
                                if m_prime == 0:
                                    UzR[i, j] /= np.sqrt(2)
                                if m == 0:
                                    UzR[i, j] /= np.sqrt(2)
                            if s_prime == 1 and s == 1:
                                UzR[i, j] = wigner_d_function(
                                    l_prime, m_prime, l,
                                    m) - (-1)**m * wigner_d_function(
                                        l_prime, m_prime, l, -m)

RotatedOmegaxR = np.dot(UzR.transpose(), np.dot(OmegaxR,
                                                UzR))  # equals OmegayR
RotatedOmegayR = np.dot(UxR.transpose(), np.dot(OmegayR,
                                                UxR))  # equals OmegazR

RotatedAxR = np.dot(UzR.transpose(), np.dot(AxR, UzR))  # equals AyR
RotatedAyR = np.dot(UxR.transpose(), np.dot(AyR, UxR))  # equals AzR

# Matrix exponentials
ExponentialUx = sp.linalg.expm(1j * Omegax * np.pi / 2)
ExponentialUz = sp.linalg.expm(1j * Omegaz * np.pi / 2)

ExponentialUxR = sp.linalg.expm(OmegaxR * np.pi / 2)
ExponentialUzR = sp.linalg.expm(OmegazR * np.pi / 2)
