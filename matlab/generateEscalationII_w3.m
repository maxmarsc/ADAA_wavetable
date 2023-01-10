% Generate one of Massive wavetables: the third in M2-Basic/EscalationII.wav
% these wav files contain one or more 2048 point tables. This is the third.

function [X,m,q,wt] = generateEscalationII_w3()

    tabL = 2048;

    % data points
    X = [0      1       2       3       4       5       6       7       8]/8;
    m = [   1       -1      1       -1      -1      1       -1      1]*2;
    q = [0      0       0       0       2       -2      2     -2];

    % generate plot
    segmL = tabL / (length(X)-1);
    y = zeros(1,tabL);
    for i = 1:length(X)-1
        x = linspace(X(i),X(i+1),segmL);
        y(1+(segmL*(i-1)):segmL*(i)) = m(i)*x + q(i);
    end
    wt = y;
    %figure, plot(y)
    
end