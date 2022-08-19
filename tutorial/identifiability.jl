# Nathaniel Linden
# UCSD MAE

# Script for the CIUKF-MCMC tutorial
# This script runs structural identifiability analysis of the MAPK model with SIAN-Julia

using SIAN

# Run with full state measurements
ode = @ODEmodel(
    x1'(t) = (k1 * (100 - x1(t)) * ((K1^10) / (K1^10 + x3(t)^10))) - (k2 * x1(t)),
    x2'(t) =  (k3 * (100 - x2(t)) * x1(t) * (1 + ((alph * x3(t)^15) / 
                    (K2^15 + x3(t)^15)))) - (k4 * x2(t)),
    x3'(t) = (k5 * (100 - x3(t)) * x2(t)) - (k6 * x3(t)),
    y1(t) = x1(t),
    y2(t) = x2(t),
    y3(t) = x3(t)
);

# run with 12 threads
output = identifiability_ode(ode, get_parameters(ode), p_mod=p_mod=2^29 - 3, nthrds=12);