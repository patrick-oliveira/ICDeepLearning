def ConvArithR1(i, k):
    """
        s = 1
        p = 0
    """
    o = (i - k) + 1
    return o

def ConvArithR2(i, k, p):
    """
        s = 1
    """
    o = (i - k) + 2*p + 1
    return o

def ConvArithR3(i, k):
    """
        s = 1
        k = 2*n + 1
        p = k//2 = n
    """
    o = i
    return o

def ConvArithR4(i, k):
    """
        p = k - 1
        s = 1
    """
    o = i + (k - 1)
    return o

def ConvArithR5(i, k, s): 
    """
        p = 0
    """
    o = (i - k)//s + 1
    return o

def ConvArithR6(i, k, p, s):
    o = (i + 2*p - k)//s + 1
    return o

def ConvArithP(i, k, s):
    o = (i - k)//s + 1
    return o
    

def ConvArithSelect(relationship, *args, **kwargs):
    functions = {
        'R1': ConvArithR1,
        'R2': ConvArithR2,
        'R3': ConvArithR3,
        'R4': ConvArithR4,
        'R5': ConvArithR5,
        'R6': ConvArithR6,
        'P' : ConvArithP
    }
    return functions[relationship](*args, **kwargs)