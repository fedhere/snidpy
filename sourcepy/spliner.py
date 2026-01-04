import numpy as np

class CubicSpline:
    """Natural cubic spline implementation (translated from IDL/Vista)."""
    
    def __init__(self, max_knots=150):
        self.max_knots = max_knots
        self.x = None      # Knot positions
        self.y = None      # Knot values
        self.y2 = None     # Second derivatives
        self.h = None      # Knot spacing
        
    def fit(self, x, y):
        """
        Generate natural cubic spline on knot points (x, y).
        
        Args:
            x: Ordered knot positions
            y: Function values at knot points
            
        Returns:
            bool: True if error occurred
        """
        n = len(x)
        if n > self.max_knots:
            print(f"Number of spline points must be <= {self.max_knots}")
            return True
        
        # Check ordering
        if not np.all(np.diff(x) > 0):
            print("Spline X values are out of order...")
            return True
        
        self.x = np.array(x)
        self.y = np.array(y)
        n_points = n
        
        # Calculate knot spacing
        self.h = np.diff(self.x)
        
        # Calculate slopes between knots
        f = np.diff(self.y)
        r = f / self.h
        
        # Natural spline: second derivatives zero at ends
        self.y2 = np.zeros(n_points)
        
        if n_points > 2:
            # Tridiagonal system for second derivatives
            n_a = n_points - 2  # Number of interior points
            
            # Build tridiagonal matrix
            a = 2.0 * (self.h[1:] + self.h[:-1])  # Main diagonal
            c = self.h[1:-1]                      # Upper diagonal
            b = 6.0 * (r[1:] - r[:-1])            # Right-hand side
            
            # Solve tridiagonal system with Thomas algorithm
            u = np.zeros(n_a)
            l = np.zeros(n_a)
            yt = np.zeros(n_a)
            
            # Forward elimination
            u[0] = a[0]
            yt[0] = b[0]
            
            for i in range(1, n_a):
                l[i] = c[i-1] / u[i-1]
                u[i] = a[i] - l[i] * c[i-1]
                yt[i] = b[i] - l[i] * yt[i-1]
            
            # Back substitution
            self.y2[n_a] = yt[n_a-1] / u[n_a-1]
            for i in range(2, n_a+1):
                fac = yt[n_a-i] - c[n_a-i] * self.y2[n_a+2-i]
                self.y2[n_a+1-i] = fac / u[n_a-i]
        
        # Scale second derivatives
        self.y2 /= 6.0
        return False  # No error
    
    def evaluate(self, xp):
        """
        Evaluate spline at position xp.
        
        Args:
            xp: Evaluation point(s)
            
        Returns:
            float or ndarray: Spline value(s) at xp
        """
        xp = np.asarray(xp)
        scalar_input = xp.ndim == 0
        xp = np.atleast_1d(xp)
        
        result = np.zeros_like(xp)
        
        for i, x_val in enumerate(xp):
            # Find which interval x_val falls into
            if x_val < self.x[0]:  # Left extrapolation
                deriv = (-3.0 * self.y2[0] / self.h[0] * (self.x[1] - self.x[0])**2 +
                        (self.y[1] - self.y[0]) / self.h[0] + 
                        (self.y2[0] - self.y2[1]) * self.h[0])
                result[i] = (x_val - self.x[0]) * deriv + self.y[0]
                
            elif x_val > self.x[-1]:  # Right extrapolation
                deriv = ((self.y[-1] - self.y[-2]) / self.h[-1] +
                        (self.y2[-2] - self.y2[-1]) * self.h[-1] +
                        3.0 * self.y2[-1] / self.h[-1] * (self.x[-1] - self.x[-2])**2)
                result[i] = (x_val - self.x[-1]) * deriv + self.y[-1]
                
            else:  # Interior evaluation
                # Binary search for interval
                idx = np.searchsorted(self.x, x_val) - 1
                if idx < 0:
                    idx = 0
                if idx >= len(self.x) - 1:
                    idx = len(self.x) - 2
                
                # Cubic spline formula
                dx = self.x[idx+1] - self.x[idx]
                sp1 = self.y2[idx] / dx
                sp2 = self.y2[idx+1] / dx
                sp3 = self.y[idx+1] / dx - self.y2[idx+1] * dx
                sp4 = self.y[idx] / dx - self.y2[idx] * dx
                
                dxp = self.x[idx+1] - x_val
                dxm = x_val - self.x[idx]
                
                result[i] = (sp1 * dxp**3 + sp2 * dxm**3 + 
                            sp4 * dxp + sp3 * dxm)
        
        return result[0] if scalar_input else result
    
    def derivative(self, xp):
        """
        Evaluate spline derivative at position xp.
        
        Args:
            xp: Evaluation point(s)
            
        Returns:
            float or ndarray: Derivative value(s) at xp
        """
        xp = np.asarray(xp)
        scalar_input = xp.ndim == 0
        xp = np.atleast_1d(xp)
        
        result = np.zeros_like(xp)
        
        for i, x_val in enumerate(xp):
            if x_val < self.x[0]:  # Left extrapolation
                result[i] = (-3.0 * self.y2[0] / self.h[0] * (self.x[1] - self.x[0])**2 +
                            (self.y[1] - self.y[0]) / self.h[0] + 
                            (self.y2[0] - self.y2[1]) * self.h[0])
                
            elif x_val > self.x[-1]:  # Right extrapolation
                result[i] = ((self.y[-1] - self.y[-2]) / self.h[-1] +
                            (self.y2[-2] - self.y2[-1]) * self.h[-1] +
                            3.0 * self.y2[-1] / self.h[-1] * (self.x[-1] - self.x[-2])**2)
                
            else:  # Interior evaluation
                idx = np.searchsorted(self.x, x_val) - 1
                if idx < 0:
                    idx = 0
                if idx >= len(self.x) - 1:
                    idx = len(self.x) - 2
                
                dx = self.x[idx+1] - self.x[idx]
                sp1 = self.y2[idx] / dx
                sp2 = self.y2[idx+1] / dx
                sp3 = self.y[idx+1] / dx - self.y2[idx+1] * dx
                sp4 = self.y[idx] / dx - self.y2[idx] * dx
                
                dxp = self.x[idx+1] - x_val
                dxm = x_val - self.x[idx]
                
                result[i] = -3.0 * sp1 * dxp**2 + 3.0 * sp2 * dxm**2 - sp4 + sp3
        
        return result[0] if scalar_input else result

# Convenience functions matching original IDL interface
def spline(x, y, max_knots=150):
    """
    Generate natural cubic spline (IDL-compatible interface).
    
    Returns:
        tuple: (spline_object, error_flag)
    """
    spl = CubicSpline(max_knots)
    err = spl.fit(x, y)
    return spl, err

def splineval(xp, spl):
    """Evaluate spline at xp (IDL-compatible interface)."""
    return spl.evaluate(xp)

def splinedel(xp, spl):
    """Evaluate spline derivative at xp (IDL-compatible interface)."""
    return spl.derivative(xp)

# Example usage
if __name__ == "__main__":
    # Create test data
    x_knots = [0, 1, 2, 3, 4]
    y_knots = [0, 1, 4, 1, 0]
    
    # Fit spline
    spl, err = spline(x_knots, y_knots)
    
    if not err:
        # Evaluate at points
        x_eval = np.linspace(-0.5, 4.5, 20)
        y_eval = splineval(x_eval, spl)
        y_deriv = splinedel(x_eval, spl)
        
        print(f"Spline evaluation: {y_eval[:5]}")
        print(f"Spline derivative: {y_deriv[:5]}")
