function jac = MAPK_Jacobian(x, p)

     % Unpack parameters for clarity
     S1t   =   p(1);
     S2t   =   p(2);
     S3t   =   p(3);
     k1    =   p(4);
     k2    =   p(5);
     k3    =   p(6);
     k4    =   p(7);
     k5    =   p(8);
     k6    =   p(9);
     n1    =   p(10);
     K1    =   p(11);
     n2    =   p(12);
     K2    =   p(13);
     alph =   p(14);
 
     % conservation law
     S1 = S1t - x(1);
     S2 = S2t - x(2);
     S3 = S3t - x(3);

     % non-zero elements
     J11 = -( (k1 * K1^n1) / (K1^n1 + x(3)^n1) ) - k2;
     J13 = -( n1 * k1 * (K1^n1) * S1 * (x(3)^(n1-1)) ) / ( (K1^n1 + x(3)^n1)^2 );

     J21 = k3*S2 * (1+ ((alph * x(3)^n1) / (K2^n2 + x(3)^n2)) );
     J22 = -k3*x(1) * (1+ ((alph * x(3)^n1) / (K2^n2 + x(3)^n2)) ) - k4;
     J23 = k3*S1*x(1) * ( (alph*n2*(x(3)^(n2-1)) / (K2^n2 + x(3)^n2) ) - ( (alph*n1* (x(3)^(2*n2-1)) ) / ((K2^n2 + x(3)^n2)^2) ) );

     J32 = k5*S3;
     J33 = -k5*x(3) - k6;

     jac = [J11,   0, J13;
            J21, J22, J23;
              0, J32, J33];
end