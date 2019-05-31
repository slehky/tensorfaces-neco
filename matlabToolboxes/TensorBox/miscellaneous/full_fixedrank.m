function Xf = full_fixedrank(x)
Xf = x.U*x.S*x.V';
