import numpy as np
from typing import Tuple

def poly(z: float, a: np.ndarray) -> float:
    """Evaluate polynomial a[0] + a[1]*z + a[2]*z² + ... at z."""
    return np.polyval(a[::-1], z)

def parabolapeak(a: np.ndarray) -> Tuple[float, float, float]:
    """Compute center, height, width of parabola a[0] + a[1]*z + a[2]*z²."""
    if a[2] == 0:
        return 0.0, 1.0, 1.0
    x0 = -a[1] / (2 * a[2])
    h = a[0] - a[1]**2 / (4 * a[2])
    w = 2 * np.sqrt(abs(h / a[2])) if h * a[2] < 0 else 0.0
    return x0, h, w

def quarticpeak(a: np.ndarray, x0: float, xl: float, xr: float) -> Tuple[float, float, float, bool]:
    """Compute center, height, width of quartic polynomial."""
    def p(z): return poly(z, a)
    def d(z): return a[1] + 2*a[2]*z + 3*a[3]*z**2 + 4*a[4]*z**3
    def c(z): return 2*a[2] + 6*a[3]*z + 12*a[4]*z**2
    
    # Find peak with Newton's method
    for _ in range(50):
        dx = d(x0) / max(c(x0), 1e-10)
        x0 -= dx
        if abs(dx) < 1e-3:
            break
    
    h = p(x0)
    half_h = h / 2
    noroot = False
    
    # Find half-max points
    for i, guess in enumerate([xl, xr]):
        x = guess
        found = False
        
        # Find sign change
        for s in np.linspace(0, 1, 30):
            test = x0 + s * (guess - x0)
            if p(test) < half_h:
                x = test
                found = True
                break
        
        if not found:
            noroot = True
            x = guess
        
        # Refine with Newton
        if found:
            a_temp = a.copy()
            a_temp[0] -= half_h  # Shift polynomial
            for _ in range(50):
                dx = poly(x, a_temp) / max(d(x), 1e-10)
                x -= dx
                if abs(dx) < 1e-3:
                    break
        
        if i == 0:
            xl = x
        else:
            xr = x
    
    return x0, h, xr - xl, noroot

def peakfit(cfn: np.ndarray, lz1: int = 0, lz2: int = -1) -> Tuple[float, float, float, np.ndarray, bool]:
    """
    Fit correlation peak with quartic (falls back to parabola).
    
    Args:
        cfn: Correlation function array
        lz1, lz2: Search range indices
    
    Returns:
        center, height, width, polynomial coefficients, error_flag
    """
    npt = len(cfn)
    error = False
    EXTRA = 0.2
    ERRMAX = 0.01
    
    # Find peak
    valid_indices = [i for i in range(npt) 
                     if (lz1 <= i <= lz2) or (lz2 < 0 and i >= npt + lz2)]
    imax = valid_indices[np.argmax(cfn[valid_indices])]
    cmax = cfn[imax]
    
    if cmax == 0:
        return 0.0, 0.0, 0.0, np.zeros(5), True
    
    # Check local maximum
    left = cfn[(imax - 1) % npt]
    right = cfn[(imax + 1) % npt]
    if not (cfn[imax] >= left and cfn[imax] >= right):
        return float(imax), cmax, 0.0, np.zeros(5), False
    
    # Find inflection points
    nsearch = npt // 4
    il = ir = None
    
    for i in range(1, nsearch):
        if il is None and cfn[(imax - i + 1) % npt] - cfn[(imax - i) % npt] < 0:
            il = imax - i + 1
        if ir is None and cfn[(imax + i - 1) % npt] - cfn[(imax + i) % npt] < 0:
            ir = imax + i - 1
        if il is not None and ir is not None:
            break
    
    if ir - il < 6:
        il, ir = imax - 3, imax + 3
    
    # Find half-peak points
    ilh = irh = None
    for i in range(1, nsearch):
        if irh is None and cfn[(imax + i) % npt] < cmax / 2:
            irh = imax + i
        if ilh is None and cfn[(imax - i) % npt] < cmax / 2:
            ilh = imax - i
        if ilh is not None and irh is not None:
            break
    
    # Find zero crossings
    ilz = irz = None
    for i in range(1, nsearch):
        if irz is None and cfn[(imax + i) % npt] < 0:
            irz = imax + i
        if ilz is None and cfn[(imax - i) % npt] < 0:
            ilz = imax - i
        if ilz is not None and irz is not None:
            break
    
    # Determine fit range
    ifit1 = max(il - int(EXTRA * (ir - il)), ilz or -np.inf, ilh or -np.inf)
    ifit2 = min(ir + int(EXTRA * (ir - il)), irz or np.inf, irh or np.inf)
    ifit1 = max(ifit1, 0)
    ifit2 = min(ifit2, npt - 1)
    
    # Fit polynomial
    x = np.arange(ifit1, ifit2 + 1)
    y = np.array([cfn[i % npt] for i in range(ifit1, ifit2 + 1)])
    nfit = len(x)
    
    if nfit >= 8:
        coeff = np.polyfit(x, y, 4)[::-1]  # Convert to a[0] + a[1]*x + ...
        ncoeff = 5
    else:
        coeff = np.polyfit(x, y, 2)[::-1]
        coeff = np.append(coeff, [0, 0])  # Pad to length 5
        ncoeff = 3
    
    # Check residuals
    residuals = y - poly(x, coeff[:ncoeff])
    rms = np.sqrt(np.mean(residuals**2))
    
    if rms / cmax > ERRMAX and nfit > 6:
        # Try narrower fit
        ifit1 = min(ifit1 + 2, ilh or ifit1)
        ifit2 = max(ifit2 - 2, irh or ifit2)
        return peakfit(cfn, lz1, lz2)  # Recursive refinement
    
    # Compute peak parameters
    if ncoeff == 5:
        center, height, width, noroot = quarticpeak(coeff, float(imax), 
                                                   float(ilh or imax-3), 
                                                   float(irh or imax+3))
        if noroot:
            center, height, width = parabolapeak(coeff[:3])
    else:
        
        center, height, width = parabolapeak(coeff[:3])
    
    return center, height, width, coeff, error
