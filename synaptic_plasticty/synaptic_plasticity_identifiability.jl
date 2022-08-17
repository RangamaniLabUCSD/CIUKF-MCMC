using SIAN

Fully observed system
ode = @ODEmodel(
    x1'(t) = k1*x1(t)*((Ktot-x1(t))/(Km1 + (Ktot-x1(t)))) - k2*(x2(t)+P0)*(x1(t)/(Km2+x1(t))) + k3*K0 + k4*(Ktot-x1(t))*((u(t)^4)/((Km3^4) + (u(t)^4))),
    x2'(t) = k5*x2(t)*((Ptot-x2(t))/(Km4+(Ptot-x2(t)))) - k6*(x1(t)+K0)*(x2(t)/(Km5+x2(t))) + k7*P0 + k8*(Ptot-x2(t))*((u(t)^3)/((Km3^3) + (u(t)^3))),
    x3'(t) = (c1*x1(t)+c3)*(Atot-x3(t)) - (c2*x2(t)-c4)*x3(t),
    y1(t) = x1(t),
    y2(t) = x2(t),
    y3(t) = x3(t)
);

output = identifiability_ode(ode, get_parameters(ode), p_mod=p_mod=2^29 - 3, nthrds=6);

# No included in manuscript
# # Partially observed system
# ode = @ODEmodel(
#     x1'(t) = k1*x1(t)*((Ktot-x1(t))/(Km1 + (Ktot-x1(t)))) - k2*(x2(t)+P0)*(x1(t)/(Km2+x1(t))) + k3*K0 + k4*(Ktot-x1(t))*((u(t)^4)/((Km3^4) + (u(t)^4))),
#     x2'(t) = k5*x2(t)*((Ptot-x2(t))/(Km4+(Ptot-x2(t)))) - k6*(x1(t)+K0)*(x2(t)/(Km5+x2(t))) + k7*P0 + k8*(Ptot-x2(t))*((u(t)^3)/((Km3^3) + (u(t)^3))),
#     x3'(t) = (c1*x1(t)+c3)*(Atot-x3(t)) - (c2*x2(t)-c4)*x3(t),
#     y1(t) = x3(t)
# );

# output = identifiability_ode(ode, get_parameters(ode), p_mod=p_mod=2^29 - 3, nthrds=12);