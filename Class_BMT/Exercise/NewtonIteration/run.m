% Lecture "Basic Math Tools"
% Supplementary Material
%
% Technische Universit�t M�nchen, Fakult�t f�r Informatik
% Dr. Tobias Lasser, Richard Brosig, Jakob Vogel
% 2011

% the pr ocess finds the unique root 
% fun1 = @(x) (x.^3);
% newtonIteration(fun1,6.5,20,0.01,[-7,7])
% pause();
%  
% the process starts to oscillate (but recovers due to rounding errors)
fun2 = @(x) (1./(x.^2+1))-1;
newtonIteration(fun2,0.8,10,0.01,[-5,5])
pause();

fun2 = @(x) (1./(x.^2+1))-1;
newtonIteration(fun2,sqrt(3),20,0.01,[-5,5])
pause();

fun2 = @(x) (1./(x.^2+1))-1;
newtonIteration(fun2,1.7320,10,0.01,[-5,5])
pause();

fun2 = @(x) (1./(x.^2+1))-1;
newtonIteration(fun2,1.7322,10,0.01,[-5,5])
pause();
 
fun3 = @(x) (x.^2-2);
newtonIteration(fun3,9,10,0.01,[-10,10])
 