
def step_point_milieu(y, t, h, f):
    ynu = y + (h / 2) * f ( y, t )
    pn = f ( ynu, t + h / 2 )
    return y + h * pn

    