function xdot = two_state(time, x_values, p)
    % function two_state takes
    %
    % either	1) no arguments
    %       	    and returns a vector of the initial values
    %
    % or    	2) time - the elapsed time since the beginning of the reactions
    %       	   x_values    - vector of the current values of the variables
    %       	    and returns a vector of the rate of change of value of each of the variables
    %
    % two_state can be used with MATLABs odeN functions as 
    %
    %	[t,x] = ode23(@two_state, [0, t_end], two_state)
    %
    %			where  t_end is the end time of the simulation
    %
    %The variables in this model are related to the output vectors with the following indices
    %	Index	Variable name
    %	  1  	  x1
    %	  2  	  x2
    %
    %--------------------------------------------------------
    % output vector
    
    xdot = zeros(2, 1);
    
    %--------------------------------------------------------
    % compartment values
    
    compartment = 1;
    
    %--------------------------------------------------------
    % parameter values 
     
    k1e = p(1);
    k12 = p(2);
    k21 = p(3);
    b = p(4);
    u = p(5);
    
    %--------------------------------------------------------
    % unpack xvalues 
     
    x1 = x_values(1);
    x2 = x_values(2);
    
    %--------------------------------------------------------
    % assignment rules
    
    %--------------------------------------------------------
    % algebraic rules
    
    %--------------------------------------------------------
    
    % rate equations
    xdot(1) = -(k1e+k12)*x1+k21*x2+b*u;
    xdot(2) = k12*x1-k21*x2;
    