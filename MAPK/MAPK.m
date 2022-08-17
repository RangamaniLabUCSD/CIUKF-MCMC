function xdot = MAPK(time, x_values, p)
    % function MAPK takes
    %
    % either	1) no arguments
    %       	    and returns a vector of the initial values
    %
    % or    	2) time - the elapsed time since the beginning of the reactions
    %       	   x_values    - vector of the current values of the variables
    %       	    and returns a vector of the rate of change of value of each of the variables
    %
    % MAPK can be used with MATLABs odeN functions as 
    %
    %	[t,x] = ode23(@MAPK, [0, t_end], MAPK)
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
     
    S1t = p(1);
    S2t = p(2);
    S3t = p(3);
    k1 = p(4);
    k2 = p(5);
    k3 = p(6);
    k4 = p(7);
    k5 = p(8);
    k6 = p(9);
    K1 = p(10);
    n1 = p(11);
    K2 = p(12);
    n2 = p(13);
    alpha = p(14);
    
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
    xdot(1) = k1*(S1t-x1)*(power(K1,n1)/(power(K1,n1)+power(x3,n1)))-k2*x1;
    xdot(2) = k3*(S2t-x2)*x1*(1+alpha*power(x3,n2)/(power(K2,n2)+power(x3,n2)))-k4*x2;
    xdot(3) = k5*(S3t-x3)*x2-k6*x3;
    