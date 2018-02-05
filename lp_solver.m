clear clc;

% c = [-1; -2; 0; 0; 0];
% A = [-2 1 1 0 0; -1 1 0 1 0; 1 0 0 0 1];
% b = [2; 3; 2];

c = [-1; -2; 0; 0];
A = [1 1 1 0; 2 1 0 1];
b = [20; 30];

optimal = solve_lp(c,A,b);

function f = solve_lp(c,A,b)
    x = [];
    bfs = [];
    B = [];

    permute = combinatoric(A,x,size(A,1),1);
    
    for i=1:size(permute,2)
        temp = zeros(size(A,2), 1);

        for j=1:size(permute,1)
            B = cat(2, B, A(:,permute(j,i)));
        end

        if det(B) ~= 0
            x_ext = B\b;

            for k=1:size(permute,1)
                temp(permute(k,i)) = x_ext(k);
            end

            if x_ext > 0      
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

function g = combinatoric(A,x,n,initial)
    
    if size(x,1) == n
        g = x;
    
    else
        if initial ~= 1
            xnew = [];
            
            for i=1:size(x,2)
                for j=1:size(A,2)
                    if ~ismember(j,x(:,i))
                        xnew = cat(2,xnew,cat(1,x(:,i),j));
                    end
                end
            end
            
            g = combinatoric(A,xnew,n,0); 
        
        else
            for i=1:size(A,2)
                for j=i:size(A,2)
                    if initial == 1 && i~=j
                        x = cat(2, x, [i;j]);
                    end
                end
            end
            g = combinatoric(A,x,n,0);    
        end      
    end
end