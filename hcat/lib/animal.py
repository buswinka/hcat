
# https://en.wikipedia.org/wiki/Greenwood_function
class Animal:
    NAME: str
    A: float  # Scaling constant
    K: float  # constant of integration
    B: float  # slope of straight-line portion of the frequency-position curve


"""
public double fmouse(double d){ // d is fraction total distance
    //f(Hz) = (10 ^((1-d)*0.92) - 0.680)* 9.8
    return (Math.pow(10, (1-d)*0.92) - 0.680) * 9.8;
}
"""
class Mouse(Animal):
    NAME: str = 'mouse'
    A: float = 9.8
    B: float = 0.92
    K: float = 0.680

