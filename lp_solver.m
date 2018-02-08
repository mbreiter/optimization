%% mie376 homework 2
% matthew reiter (1002246722)

%% questions 9 and 15
clear clc;

c = [-1; -2; 0; 0];
A = [1 -2 -1 0; 1 1 0 1];
b = [2; 4];

optimal = solve_lp(c,A,b);

function f = solve_lp(c,A,b)
    x = [];
    bfs = [];
    B = [];
    
    permute = transpose(combnk(1:size(A,2), size(A,1)));
    
    for i=1:size(permute,2)
        temp = zeros(size(A,2), 1);

        for j=1:size(permute,1)
            B = cat(2, B, A(:,permute(j,i)));
        end

        if det(B) ~= 0
            x_ext = B\b;

            if x_ext > 0   
                for k=1:size(permute,1)
                    temp(permute(k,i)) = x_ext(k);
                end
                
                bfs = cat(2, bfs, temp);        
            end    
        end

        B=[];

    end

    optimal = bfs(:,1);

    for i=1:size(bfs,2)
        soln = transpose(c)*bfs(:,i);

        if soln < optimal
            optimal = soln;
        end
    end
    
    f = optimal;
end