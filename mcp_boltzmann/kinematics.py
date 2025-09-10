import numpy as np
import scipy.special as sps

def tri(a,b,c):
	return a**2 + b**2 + c**2 - 2*(a*b + a*c + b*c)

def beta_t(m, t, eps=1e-6):
    arg_sqrt = (1 - 4*m**2/t)
    return np.sqrt(arg_sqrt*np.heaviside(np.abs(arg_sqrt) - eps, 0))

def E1_cm(s, m1, m2):
	return (s + m1**2 - m2**2)/(2*np.sqrt(s))

def E3_cm(s, m3, m4):
	return E1_cm(s, m3, m4)

def p1_cm(s, m1, m2):
	return np.sqrt(E1_cm(s,m1,m2)**2 - m1**2)

def p3_cm(s, m3, m4):
	return p1_cm(s, m3, m4)

def t0(s, m1, m2, m3, m4):
	return ((m1**2 - m3**2 - m2**2 + m4**2)/(2*np.sqrt(s)))**2 - (p1_cm(s,m1,m2) - p3_cm(s, m3, m4))**2

def t1(s, m1, m2, m3, m4):
	return ((m1**2 - m3**2 - m2**2 + m4**2)/(2*np.sqrt(s)))**2 - (p1_cm(s,m1,m2) + p3_cm(s, m3, m4))**2

def F12(s, m1, m2):
	arg_sqrt = np.heaviside(tri(s, m1**2, m2**2), 1)*tri(s, m1**2, m2**2)
	return 0.5*np.sqrt(arg_sqrt)

def s_min(m1, m2, m3, m4):
	return max((m1+m2)**2, (m3+m4)**2)

def s_low(m1, m2, E1, E2):
	p1 = np.sqrt(E1**2 - m1**2)
	p2 = np.sqrt(E2**2 - m2**2)

	return m1**2 + m2**2 + 2*E1*E2 - 2*p1*p2

def s_high(m1, m2, E1, E2):
	p1 = np.sqrt(E1**2 - m1**2)
	p2 = np.sqrt(E2**2 - m2**2)

	return m1**2 + m2**2 + 2*E1*E2 + 2*p1*p2

