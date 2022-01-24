% Lecture "Basic Math Tools"
% Supplementary Material
%
% Technische Universit�t M�nchen, Fakult�t f�r Informatik
% Dr. Tobias Lasser, Richard Brosig, Jakob Vogel
% 2011

% NEWTONITERATION
% Computes and visualize the newton iteration in one dimension.

function [ x1 ] = newtonIteration(fun, x0, maxSteps, threshold, varargin )
    xstart = x0;
    plotrange=[-10,10];
    nrOfEvaluations=100;
    h=0.0001; % value for the approximation of the differential quotient
    if(nargin==5)
        plotrange=varargin{1};
    end
    % First compute the function in the on the given points 
    x=linspace(plotrange(1),plotrange(2),nrOfEvaluations);
    y=fun(x);
    %Plot the result
    fig=figure();
    plot(x,y);
    hold on
    grid on
    
    steps=0;
    x1=x0;
    x0=inf;
    i = 0;
    while(abs(x0-x1)>threshold && steps<maxSteps && isfinite(x1))
        steps=steps+1;
        x0=x1;
        % Compute differential with central differences
        fprime=(fun(x0+h)-fun(x0-h))/(2*h);
        % Compute the next Newton iteration
        x1=x0-fun(x0)/fprime;
        
        % Visualize the result
%         line([x0, x1],[fun(x0),0],'Color','r');
%         line([x1, x1],[0,fun(x1)],'Color','k');
        if(mod(i,2))
            line([x0, x1],[fun(x0),fun(x1)],'Color','k', 'LineWidth', 2);
        else
            line([x0, x1],[fun(x0),fun(x1)],'Color','r', 'LineWidth', 2);
        end
        fprintf('Current value: %f\n',x1);
        title("x_o: " + xstart + ", Current value: " + x1);
        pause;
        i = i+1;
    end
    close(fig);
end

