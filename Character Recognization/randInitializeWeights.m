function W = randInitializeWeights(L_in, L_out)

    W = zeros(L_out, 1 + L_in);
    epsiloninit = 0.12;
    W = rand(L_out, 1 + L_in)*2*epsiloninit - epsiloninit;

end
