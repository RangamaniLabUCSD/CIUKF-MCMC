using SIAN

# Run with full state measurements
# Fully observed system
ode = @ODEmodel(
    x1'(t) = (k1 * (S1t - x1(t)) * ((K1^10) / (K1^10 + x3(t)^10))) - (k2 * x1(t)),
    x2'(t) =  (k3 * (S2t - x2(t)) * x1(t) * (1 + ((alph * x3(t)^15) / (K2^15 + x3(t)^15)))) - (k4 * x2(t)),
    x3'(t) = (k5 * (S3t - x3(t)) * x2(t)) - (k6 * x3(t)),
    y1(t) = x1(t),
    y2(t) = x2(t),
    y3(t) = x3(t)
);

output = identifiability_ode(ode, get_parameters(ode), p_mod=p_mod=2^29 - 3, nthrds=12);

# Uncomment to run with y= x3
# Partially observed system
# ode = @ODEmodel(
#     x1'(t) = (k1 * (S1t - x1(t)) * ((K1^10) / (K1^10 + x3(t)^10))) - (k2 * x1(t)),
#     x2'(t) =  (k3 * (S2t - x2(t)) * x1(t) * (1 + ((alph * x3(t)^15) / (K2^15 + x3(t)^15)))) - (k4 * x2(t)),
#     x3'(t) = (k5 * (S3t - x3(t)) * x2(t)) - (k6 * x3(t)),
#     y1(t) = x3(t)
# );

# output = identifiability_ode(ode, get_parameters(ode), p_mod=p_mod=2^29 - 3, nthrds=12);