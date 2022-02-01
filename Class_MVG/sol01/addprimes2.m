function [ result ] = addprimes2( s, e )
    A = s:1:e;
    result = sum(isprime(A) .* A);
    % isprime函数，如果是质数返回下标1，否则返回下标0
end

