function xdot = synaptic_plasticity(time, x_values, p)
% function synaptic_plasticity takes
%
% either	1) no arguments
%       	    and returns a vector of the initial values
%
% or    	2) time - the elapsed time since the beginning of the reactions
%       	   x_values    - vector of the current values of the variables
%       	    and returns a vector of the rate of change of value of each of the variables
%
% synaptic_plasticity can be used with MATLABs odeN functions as 
%
%	[t,x] = ode23(@synaptic_plasticity, [0, t_end], synaptic_plasticity)
%
%			where  t_end is the end time of the simulation
%
%The variables in this model are related to the output vectors with the following indices
%	Index	Variable name
%	  1  	  x1
%	  2  	  x2
%	  3  	  x3
%
%--------------------------------------------------------
% output vector

xdot = zeros(3, 1);

%--------------------------------------------------------
% compartment values

compartment = 1;

%--------------------------------------------------------
% parameter values 
 
k1 = p(1);
k2 = p(2);
k3 = p(3);
k4 = p(4);
k5 = p(5);
k6 = p(6);
k7 = p(7);
k8 = p(8);
c1 = p(9);
c2 = p(10);
c3 = p(11);
c4 = p(12);
Km1 = p(13);
Km2 = p(14);
Km3 = p(15);
Km4 = p(16);
Km5 = p(17);
K0 = p(18);
P0 = p(19);
Ktot = p(20);
Atot = p(21);
Ptot = p(22);
Ca = p(23);

%--------------------------------------------------------
% unpack xvalues 
 
x1 = x_values(1);
x2 = x_values(2);
x3 = x_values(3);

%--------------------------------------------------------
% assignment rules

%--------------------------------------------------------
% algebraic rules

%--------------------------------------------------------

% rate equations
xdot(1) = k1*x1*((Ktot-x1)/(Ktot-x1+Km1))-k2*(x2+P0)*(x1/(Km2+x1))+k3*K0+k4*(Ktot-x1)*(power(Ca,4)/(power(Km3,4)+power(Ca,4)));
xdot(2) = k5*x2*((Ptot-x2)/(Ptot-x2+Km4))-k6*(x1+K0)*(x2/(Km5+x2))+k7*P0+k8*(Ptot-x2)*(power(Ca,3)/(power(Km3,3)+power(Ca,3)));
xdot(3) = (c1*x1+c3)*(Atot-x3)-(c2*x2+c4)*x3;
