
function [X, m, q, wt] = generateWavetableSaw()
    function m = compute_m(x0, x1, y0, y1)
        m = (y1 - y0) / (x1 - x0);
    end

    function q = compute_q(x0, x1, y0, y1)
        q = (y0 * (x1 - x0) - x0 * (y1 - y0)) / (x1 - x0);
    end
    
    FRAMES = 2048;
    X = linspace(0.0, 1.0, FRAMES + 1);
    wt = zeros(1,FRAMES);
    m = zeros(1,FRAMES);
    q = zeros(1,FRAMES);

    steps = 1.0/FRAMES;
    % phase = 0.5 + steps;
    phase = 0;

    for i = 2:FRAMES
        wt(i) = 2.0 * phase - 1.0;

        m(i-1) = compute_m(X(i-1), X(i), wt(i-1), wt(i));
        q(i-1) = compute_q(X(i-1), X(i), wt(i-1), wt(i));

        phase = mod((phase + steps), 1.0);
    end

    m(end) = compute_m(X(FRAMES - 1), X(FRAMES), wt(FRAMES - 1), wt(FRAMES));
    q(end) = compute_q(X(FRAMES - 1), X(FRAMES), wt(FRAMES - 1), wt(FRAMES));
end